"""GEX Wall Solidity — Feature Set 2

Adds per-day options-derived features to the wall approach events:
  - net_gex_at_spot:       signed sum of GEX within ±25pt of spot (regime signal)
  - dist_wall_to_zgl:      wall_level minus zero-gamma level (positive = wall beyond ZGL)
  - gex_at_wall_strike:    |GEX| magnitude at the specific wall strike being tested
  - wall_breached_at_open: did price trade beyond wall_level in the first 15 min?
  - approach_momentum:     SPX close minus close 20 bars prior at first_approach_ts
  - repeat_test_count:     number of prior zone entries+exits before this approach
  - total_vanna:           aggregate vanna exposure at open snapshot
  - total_charm:           aggregate charm exposure at open snapshot

Outputs:
  out/gex_walls/features2.csv      — events with new features joined
  out/gex_walls/correlations2.csv  — correlations with hold/break outcomes
  out/gex_walls/buckets_regime.csv — net_gex_at_spot bucket table
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_zero_gamma_level, gex_at_strike, net_gex_at_spot
from calc.vanna_charm import aggregate_vanna_charm
from util.minio import get_duckdb_conn, option_parquet_uris

TZ       = "America/New_York"
PROVIDER = "schwab"
OUT_DIR  = Path("out/gex_walls")

COLUMNS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price", "volatility",
]

OUTCOMES = ["bounce_30m", "bounce_60m", "eod_held"]
APPROACH_PTS = 10.0   # zone radius for repeat-test counting
EXIT_PTS     = 15.0   # must move this far away to count as "exited" zone


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_day_options(con, trade_date: date) -> pd.DataFrame:
    uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
    uri = uris[0]
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    cols = ", ".join(COLUMNS)
    df = con.execute(f"SELECT {cols} FROM read_parquet('{uri}', union_by_name=true)").fetchdf()
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    for c in ["gamma", "open_interest", "underlying_price", "volatility"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_open_snapshot(day_opts: pd.DataFrame) -> pd.DataFrame | None:
    cutoff = day_opts["sampled_at"].iloc[0].replace(hour=9, minute=40, second=0, microsecond=0)
    valid = day_opts[day_opts["sampled_at"] >= cutoff]
    if valid.empty:
        return None
    snap_ts = valid["sampled_at"].min()
    return day_opts[day_opts["sampled_at"] == snap_ts]


def load_spx_1min(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT datetime, open, high, low, close
        FROM read_parquet('s3://tractatus-lakehouse/candles/schwab/1min/SPX.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
    return df.set_index("datetime")


# ---------------------------------------------------------------------------
# Per-day option-derived features
# ---------------------------------------------------------------------------

def compute_day_features(snap: pd.DataFrame, spot: float, trade_date: date) -> dict:
    """Compute all open-snapshot scalar features for a single day."""
    zgl = find_zero_gamma_level(snap, spot=spot, strike_range=300.0)
    net_gex = net_gex_at_spot(snap, spot=spot, window=25.0)
    vc = aggregate_vanna_charm(snap, spot=spot, strike_range=300.0)
    return {
        "spot_open":    spot,
        "zgl":          zgl,
        "net_gex_spot": net_gex,
        "total_vanna":  vc["total_vanna"],
        "total_charm":  vc["total_charm"],
    }


# ---------------------------------------------------------------------------
# Repeat test count
# ---------------------------------------------------------------------------

def count_repeat_tests(
    price_series: pd.Series,
    wall_level: float,
    first_approach_ts: pd.Timestamp,
) -> int:
    """Count how many prior zone entries occurred before first_approach_ts.

    A new test = price enters ±APPROACH_PTS zone, then exits >EXIT_PTS away,
    then re-enters. The first entry of the day counts as test 0 (not a repeat).
    Returns the number of *prior* completed test cycles before first_approach_ts.
    """
    day_prices = price_series.sort_index()
    pre = day_prices[day_prices.index < first_approach_ts]
    if pre.empty:
        return 0

    in_zone = False
    repeat_count = 0
    exited_after_entry = False

    for price in pre:
        near = abs(price - wall_level) <= APPROACH_PTS
        far  = abs(price - wall_level) > EXIT_PTS

        if near and not in_zone:
            if exited_after_entry:
                repeat_count += 1
            in_zone = True
            exited_after_entry = False
        elif far and in_zone:
            in_zone = False
            exited_after_entry = True

    return repeat_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading events...")
    events = pd.read_csv("out/gex_walls/features.csv")
    events = events[events["first_approach_ts"].notna()].copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    events["first_approach_ts"] = pd.to_datetime(events["first_approach_ts"], utc=True).dt.tz_convert(TZ)
    for col in OUTCOMES:
        events[col] = events[col].map({"True": True, "False": False, True: True, False: False})
    print(f"  {len(events):,} events, {events['date'].nunique()} days")

    con = get_duckdb_conn()
    spx = load_spx_1min(con)
    print(f"  SPX 1-min: {len(spx):,} bars ({spx.index.min().date()} → {spx.index.max().date()})")

    trading_days = sorted(events["date"].unique())

    # --- Per-day options features ---
    print("\nComputing per-day options features (ZGL, net GEX at spot, vanna, charm)...")
    day_features: dict[date, dict] = {}

    for trade_date in tqdm(trading_days, desc="Days"):
        day_opts = load_day_options(con, trade_date)
        if day_opts.empty:
            continue
        snap = get_open_snapshot(day_opts)
        if snap is None:
            continue
        spot_vals = snap["underlying_price"].dropna()
        if spot_vals.empty:
            continue
        spot = float(spot_vals.median())
        day_features[trade_date] = compute_day_features(snap, spot, trade_date)

    con.close()
    print(f"  Computed for {len(day_features)} / {len(trading_days)} days")

    # Join day-level features
    for feat in ["zgl", "net_gex_spot", "total_vanna", "total_charm", "spot_open"]:
        events[feat] = events["date"].map({d: v[feat] for d, v in day_features.items()})

    events["dist_wall_to_zgl"] = events["wall_level"] - events["zgl"]

    # --- Per-event options features: GEX at wall strike ---
    print("\nComputing GEX at wall strike per event...")
    # We already have net_gex_spot; for per-strike we need to rejoin options.
    # Build a lookup: (date, wall_level) -> gex_at_wall_strike
    # Reload once per day and compute for all events that day.
    con = get_duckdb_conn()
    gex_wall_map: dict[tuple, float] = {}
    gex_breach_map: dict[tuple, bool] = {}

    for trade_date in tqdm(trading_days, desc="GEX at wall"):
        day_events = events[events["date"] == trade_date]
        if day_events.empty:
            continue
        day_opts = load_day_options(con, trade_date)
        if day_opts.empty:
            continue
        snap = get_open_snapshot(day_opts)
        if snap is None:
            continue
        spot_vals = snap["underlying_price"].dropna()
        if spot_vals.empty:
            continue
        spot = float(spot_vals.median())

        for wall_level in day_events["wall_level"].unique():
            key = (trade_date, wall_level)
            gex_wall_map[key] = gex_at_strike(snap, spot=spot, strike=wall_level)

        # Wall breached at open: did SPX trade beyond wall_level in first 15 min?
        day_spx = spx[spx.index.date == trade_date].sort_index()
        open_bars = day_spx.between_time("09:30", "09:45")
        for _, ev in day_events.iterrows():
            wall = ev["wall_level"]
            wall_type = ev["wall_type"]
            key = (trade_date, wall, wall_type)
            if open_bars.empty:
                gex_breach_map[key] = False
            elif wall_type == "call":
                gex_breach_map[key] = bool(open_bars["high"].max() >= wall)
            else:
                gex_breach_map[key] = bool(open_bars["low"].min() <= wall)

    con.close()

    events["gex_at_wall_strike"] = events.apply(
        lambda r: gex_wall_map.get((r["date"], r["wall_level"]), np.nan), axis=1
    )
    events["wall_breached_at_open"] = events.apply(
        lambda r: gex_breach_map.get((r["date"], r["wall_level"], r["wall_type"]), np.nan), axis=1
    )

    # --- Momentum at approach ---
    print("\nComputing approach momentum...")
    def approach_momentum(row) -> float | None:
        ts = row["first_approach_ts"]
        if pd.isna(ts):
            return None
        day_spx = spx[spx.index.date == row["date"]].sort_index()
        if day_spx.empty:
            return None
        idx = day_spx.index.get_indexer([ts], method="nearest")[0]
        if idx < 20:
            return None
        return float(day_spx["close"].iloc[idx] - day_spx["close"].iloc[idx - 20])

    events["approach_momentum"] = events.apply(approach_momentum, axis=1)

    # --- Repeat test count ---
    print("Computing repeat test counts...")
    def repeat_tests(row) -> int:
        ts = row["first_approach_ts"]
        if pd.isna(ts):
            return 0
        day_spx = spx[spx.index.date == row["date"]].sort_index()
        if day_spx.empty:
            return 0
        return count_repeat_tests(day_spx["close"], row["wall_level"], ts)

    events["repeat_test_count"] = events.apply(repeat_tests, axis=1)

    # --- Save ---
    feat_path = OUT_DIR / "features2.csv"
    events.to_csv(feat_path, index=False)
    print(f"\nSaved {len(events):,} rows → {feat_path}")

    # --- Correlations ---
    feature_cols = [
        "net_gex_spot", "dist_wall_to_zgl", "gex_at_wall_strike",
        "wall_breached_at_open", "approach_momentum", "repeat_test_count",
        "total_vanna", "total_charm",
    ]

    print("\n=== Point-biserial correlations (feature set 2) ===")
    rows = []
    for feat in feature_cols:
        row = {"feature": feat}
        for outcome in OUTCOMES:
            sub = events[[feat, outcome]].dropna()
            sub = sub[sub[outcome].isin([True, False])]
            if len(sub) < 50:
                row[f"r_{outcome}"] = np.nan
                row[f"p_{outcome}"] = np.nan
                continue
            r, p = stats.pointbiserialr(sub[outcome].astype(int), sub[feat].astype(float))
            row[f"r_{outcome}"] = round(r, 4)
            row[f"p_{outcome}"] = round(p, 4)
        rows.append(row)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_DIR / "correlations2.csv", index=False)

    print(f"\n{'Feature':<26} {'r_bounce30':>11} {'r_bounce60':>11} {'r_eod_held':>11}  {'p_bounce30':>11}")
    print("-" * 78)
    for _, r in corr_df.iterrows():
        print(
            f"{r['feature']:<26} "
            f"{r.get('r_bounce_30m', r.get('r_bounce30m', np.nan)):>10.4f}  "
            f"{r.get('r_bounce_60m', r.get('r_bounce60m', np.nan)):>10.4f}  "
            f"{r['r_eod_held']:>10.4f}  "
            f"{r.get('p_bounce_30m', r.get('p_bounce30m', np.nan)):>10.4f}"
        )

    # --- Net GEX at spot buckets ---
    print("\n=== EOD Held by net GEX at spot regime ===")
    events["gex_regime"] = pd.cut(
        events["net_gex_spot"],
        bins=[-np.inf, -1e9, 0, 1e9, np.inf],
        labels=["large negative", "small negative", "small positive", "large positive"],
    )
    summary_rows = []
    for bucket, grp in events.dropna(subset=["net_gex_spot"]).groupby("gex_regime", observed=False):
        n = len(grp)
        row = {"bucket": bucket, "n": n}
        for col in OUTCOMES:
            sub = grp[col].dropna()
            sub = sub[sub.isin([True, False])]
            row[f"{col}_pct"] = round(sub.mean() * 100, 1) if len(sub) > 0 else np.nan
        summary_rows.append(row)
        print(f"  {str(bucket):<18} n={n:>6}  bounce30={row.get('bounce_30m_pct', 'N/A'):>5}%  eod_held={row.get('eod_held_pct', 'N/A'):>5}%")

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "buckets_regime.csv", index=False)
    print(f"\nSaved correlations2.csv, buckets_regime.csv → {OUT_DIR}")


if __name__ == "__main__":
    main()
