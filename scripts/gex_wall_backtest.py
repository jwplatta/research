"""GEX Wall Stability Backtest

For each trading day in Dec 2025–Jun 2026:
  1. Load all SPXW snapshots (all expiries, not just 0DTE) from MinIO
  2. At each snapshot timestamp, compute 4 wall/zone methods
  3. Measure intraday wall drift per method
  4. Classify each wall level against SPX 1-min price action:
       - Tested (price within 10 pts)
       - Bounce at 30 min / 1 hr after first approach
       - EOD outcome: held, crossed, or crossed-and-returned

Outputs:
  out/gex_walls/events.csv    — per wall-level event
  out/gex_walls/stability.csv — per-day drift per method
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_aggregate_wall_strikes, find_decision_zones, find_raw_wall_strikes
from util.minio import get_duckdb_conn, option_parquet_uris
from util.spx_candles import load_spx_1min

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE   = date(2025, 12, 1)
END_DATE     = date(2026, 10, 17)
PROVIDER     = "schwab"
TZ           = "America/New_York"

STRIKE_RANGE = 300.0          # pts either side of spot
APPROACH_PTS = 10.0           # price within this many pts = "tested"
BOUNCE_MINS  = [30, 60]       # time horizons for bounce check (cols: bounce_30m, bounce_60m)

OUT_DIR = Path("out/gex_walls")

COLUMNS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_day_options(con, trade_date: date) -> pd.DataFrame:
    """Load all SPXW snapshots for a single day (all expiries)."""
    uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
    uri = uris[0]
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    cols = ", ".join(COLUMNS)
    df = con.execute(f"""
        SELECT {cols}
        FROM read_parquet('{uri}', union_by_name=true)
    """).fetchdf()

    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Wall computation
# ---------------------------------------------------------------------------

def compute_walls(snap: pd.DataFrame, spot: float, snap_ts: pd.Timestamp) -> list[dict]:
    """Compute all wall methods for a single snapshot. Returns list of wall records."""
    anchor = pd.Timestamp(snap_ts.date())
    records = []

    # Method 1: raw
    raw_call, raw_put = find_raw_wall_strikes(snap, spot=spot, strike_range=STRIKE_RANGE)
    for wall_type, level in (("call", raw_call), ("put", raw_put)):
        if level is not None:
            records.append({"method": "raw", "wall_type": wall_type,
                            "wall_level": level, "spot": spot, "snapshot_ts": snap_ts})

    # Method 2: distance-weighted aggregate
    wtd_call, wtd_put = find_aggregate_wall_strikes(
        snap, spot=spot, strike_range=STRIKE_RANGE,
        method="distance_weighted_aggregate", anchor_date=anchor,
    )
    for wall_type, level in (("call", wtd_call), ("put", wtd_put)):
        if level is not None:
            records.append({"method": "weighted", "wall_type": wall_type,
                            "wall_level": level, "spot": spot, "snapshot_ts": snap_ts})

    # Method 3: per-expiry clustering
    clust_call, clust_put = find_aggregate_wall_strikes(
        snap, spot=spot, strike_range=STRIKE_RANGE,
        method="per_expiry_clustering", anchor_date=anchor,
    )
    for wall_type, level in (("call", clust_call), ("put", clust_put)):
        if level is not None:
            records.append({"method": "clustered", "wall_type": wall_type,
                            "wall_level": level, "spot": spot, "snapshot_ts": snap_ts})

    # Method 4: decision zones (top zone center per side)
    resistance_zones, support_zones = find_decision_zones(
        snap, spot=spot, strike_range=STRIKE_RANGE, anchor_date=anchor, top_n=1,
    )
    if resistance_zones:
        records.append({"method": "zones", "wall_type": "call",
                        "wall_level": resistance_zones[0]["center"],
                        "spot": spot, "snapshot_ts": snap_ts})
    if support_zones:
        records.append({"method": "zones", "wall_type": "put",
                        "wall_level": support_zones[0]["center"],
                        "spot": spot, "snapshot_ts": snap_ts})

    return records


# ---------------------------------------------------------------------------
# Price behavior classification
# ---------------------------------------------------------------------------

def classify_wall(
    wall_level: float,
    wall_type: str,
    snap_ts: pd.Timestamp,
    prices: pd.DataFrame,
) -> dict:
    """Check SPX 1-min behavior at a wall level after snap_ts."""
    post = prices[prices.index > snap_ts].copy()
    if post.empty:
        return {"tested": False, "bounce_30m": None, "bounce_1h": None,
                "eod_held": None, "eod_crossed": None, "eod_returned": None,
                "first_approach_ts": None}

    # Determine which side of the wall spot is on
    # call wall = resistance above spot; put wall = support below spot
    side_above = (wall_type == "call")  # True = wall is above current spot

    # Tested: price comes within APPROACH_PTS of wall
    if side_above:
        approached = post[post["high"] >= wall_level - APPROACH_PTS]
    else:
        approached = post[post["low"] <= wall_level + APPROACH_PTS]

    if approached.empty:
        return {"tested": False, "bounce_30m": None, "bounce_1h": None,
                "eod_held": None, "eod_crossed": None, "eod_returned": None,
                "first_approach_ts": None}

    first_ts = approached.index[0]
    result: dict = {"tested": True, "first_approach_ts": first_ts}

    # Bounce checks: is price still on the original side N minutes after approach?
    for mins in BOUNCE_MINS:
        cutoff = first_ts + pd.Timedelta(minutes=mins)
        window = post[(post.index > first_ts) & (post.index <= cutoff)]
        if window.empty:
            result[f"bounce_{mins}m"] = None
        else:
            last_close = float(window["close"].iloc[-1])
            if side_above:
                result[f"bounce_{mins}m"] = last_close < wall_level
            else:
                result[f"bounce_{mins}m"] = last_close > wall_level

    # EOD outcome
    eod_close = float(post["close"].iloc[-1]) if not post.empty else None
    if eod_close is None:
        result.update({"eod_held": None, "eod_crossed": None, "eod_returned": None})
    else:
        crossed = (side_above and eod_close >= wall_level) or (not side_above and eod_close <= wall_level)
        # "Returned": crossed at some point intraday but closed back on original side
        if side_above:
            crossed_intraday = post["high"].max() >= wall_level
            eod_back = eod_close < wall_level
        else:
            crossed_intraday = post["low"].min() <= wall_level
            eod_back = eod_close > wall_level

        result["eod_held"] = not crossed
        result["eod_crossed"] = crossed and not (crossed_intraday and eod_back)
        result["eod_returned"] = crossed_intraday and eod_back

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SPX 1-min candles {START_DATE} → {END_DATE} ...")
    spx = load_spx_1min(START_DATE, END_DATE)
    print(f"  {len(spx)} bars loaded\n")

    con = get_duckdb_conn()

    # Build list of trading days (days with SPX candle data)
    trading_days = sorted({d for d in spx.index.date if START_DATE <= d < END_DATE})
    print(f"Found {len(trading_days)} trading days\n")

    all_events: list[dict] = []
    all_stability: list[dict] = []

    for trade_date in tqdm(trading_days, desc="Days", unit="day"):
        day_opts = load_day_options(con, trade_date)
        if day_opts.empty:
            continue

        day_prices = spx[spx.index.date == trade_date].sort_index()
        if day_prices.empty:
            continue

        eod_close = float(day_prices["close"].iloc[-1])

        # All unique snapshot timestamps for the day
        snapshots = sorted(day_opts["sampled_at"].unique())

        # --- Wall computation at each snapshot ---
        day_wall_records: list[dict] = []
        for snap_ts in snapshots:
            snap = day_opts[day_opts["sampled_at"] == snap_ts]
            spot_vals = snap["underlying_price"].dropna()
            if spot_vals.empty:
                continue
            spot = float(spot_vals.median())
            walls = compute_walls(snap, spot, snap_ts)
            day_wall_records.extend(walls)

        if not day_wall_records:
            continue

        # --- Stability: drift per method/wall_type across all snapshots ---
        wall_df = pd.DataFrame(day_wall_records)
        for (method, wall_type), grp in wall_df.groupby(["method", "wall_type"]):
            levels = grp["wall_level"].dropna()
            if len(levels) < 2:
                continue
            open_snap = grp.loc[grp["snapshot_ts"].idxmin()]
            close_snap = grp.loc[grp["snapshot_ts"].idxmax()]
            all_stability.append({
                "date": trade_date,
                "method": method,
                "wall_type": wall_type,
                "open_wall": open_snap["wall_level"],
                "close_wall": close_snap["wall_level"],
                "drift_pts": float(levels.max() - levels.min()),
                "n_snapshots": len(levels),
                "eod_spx": eod_close,
            })

        # --- Price behavior: classify each wall record ---
        for rec in day_wall_records:
            behavior = classify_wall(
                wall_level=rec["wall_level"],
                wall_type=rec["wall_type"],
                snap_ts=rec["snapshot_ts"],
                prices=day_prices,
            )
            all_events.append({
                "date": trade_date,
                "snapshot_ts": rec["snapshot_ts"],
                "method": rec["method"],
                "wall_type": rec["wall_type"],
                "wall_level": rec["wall_level"],
                "spot": rec["spot"],
                **behavior,
            })

    con.close()

    # --- Save outputs ---
    events_df = pd.DataFrame(all_events)
    stability_df = pd.DataFrame(all_stability)

    events_path = OUT_DIR / "events.csv"
    stability_path = OUT_DIR / "stability.csv"
    events_df.to_csv(events_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    print(f"\nSaved {len(events_df)} events → {events_path}")
    print(f"Saved {len(stability_df)} stability records → {stability_path}")

    # --- Summary ---
    if events_df.empty:
        print("No events to summarize.")
        return

    tested = events_df[events_df["tested"]]
    print("\n=== Wall Behavior Summary ===")
    print(f"{'Method':<12} {'Side':<6} {'Total':>7} {'Tested%':>8} "
          f"{'Bnc30m%':>8} {'Bnc1h%':>8} {'EOD Held%':>10} {'EOD Ret%':>9}")
    print("-" * 75)

    for (method, wall_type), grp in events_df.groupby(["method", "wall_type"]):
        total = len(grp)
        n_tested = grp["tested"].sum()
        tst = tested[(tested["method"] == method) & (tested["wall_type"] == wall_type)]
        b30 = tst["bounce_30m"].dropna()
        b1h = tst["bounce_60m"].dropna()
        held = tst["eod_held"].dropna()
        ret = tst["eod_returned"].dropna()
        print(
            f"{method:<12} {wall_type:<6} {total:>7} "
            f"{n_tested/total*100:>7.1f}% "
            f"{b30.mean()*100 if len(b30) else float('nan'):>7.1f}% "
            f"{b1h.mean()*100 if len(b1h) else float('nan'):>7.1f}% "
            f"{held.mean()*100 if len(held) else float('nan'):>9.1f}% "
            f"{ret.mean()*100 if len(ret) else float('nan'):>8.1f}%"
        )

    print("\n=== Wall Stability (median intraday drift) ===")
    print(f"{'Method':<12} {'Side':<6} {'Median Drift':>14} {'Mean Drift':>12}")
    print("-" * 48)
    for (method, wall_type), grp in stability_df.groupby(["method", "wall_type"]):
        print(
            f"{method:<12} {wall_type:<6} "
            f"{grp['drift_pts'].median():>13.1f}  "
            f"{grp['drift_pts'].mean():>11.1f}"
        )


if __name__ == "__main__":
    main()
