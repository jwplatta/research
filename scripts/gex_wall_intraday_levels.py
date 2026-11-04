"""Compute call and put GEX wall levels at three points during the trading day.

For each trading day Dec 2025 – Aug 2026, find the nearest option chain snapshot to:
  - Morning:  09:45 ET
  - Midday:   13:00 ET
  - Close:    15:30 ET

GEX is aggregated across ALL expirations in the chain (standard community approach —
no DTE cutoff, no weighting; raw gamma × OI × spot²).

Call wall:  OTM strike with highest net call GEX above spot
Put wall:   OTM strike with most negative net put GEX below spot
(SpotGamma "raw wall" method)

Output: out/gex_walls/intraday_levels.csv
Columns:
  date, spot_morning, spot_midday, spot_close,
  call_wall_morning, put_wall_morning,
  call_wall_midday,  put_wall_midday,
  call_wall_close,   put_wall_close,
  snap_morning, snap_midday, snap_close   (actual snapshot timestamps used)
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_raw_wall_strikes
from util.minio import get_duckdb_conn, option_parquet_uris

TZ       = ZoneInfo("America/New_York")
PROVIDER = "schwab"
OUT_DIR  = Path("out/gex_walls")

START_DATE = date(2025, 12, 1)
END_DATE   = date(2026, 8, 31)   # Sept–Oct 2026 reserved for live test

# Target times (hour, minute) in ET
TARGETS = [
    ("morning", 9,  45),
    ("midday",  13,  0),
    ("close",   15, 30),
]


def load_day(con, trade_date: date) -> pd.DataFrame | None:
    """Load full option chain for trade_date. Returns None if unavailable."""
    uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
    if not uris:
        return None
    uri = uris[0]
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
    except Exception:
        return None

    df = con.execute(f"""
        SELECT sampled_at, expiration_date, strike, contract_type,
               gamma, open_interest, underlying_price
        FROM read_parquet('{uri}', union_by_name=true)
        WHERE gamma IS NOT NULL
          AND open_interest IS NOT NULL
          AND gamma > 0
          AND open_interest > 0
    """).fetchdf()

    if df.empty:
        return None

    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert("America/New_York")
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")

    # Keep only RTH snapshots
    rth = df[
        (df["sampled_at"].dt.hour >= 9) & (df["sampled_at"].dt.hour < 16)
        & (df["sampled_at"].dt.date == trade_date)
    ]
    return rth if not rth.empty else None


def nearest_snapshot(df: pd.DataFrame, trade_date: date, target_hour: int, target_min: int) -> pd.Timestamp | None:
    """Return the sampled_at timestamp nearest to (target_hour, target_min) on trade_date."""
    target = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=target_hour, minute=target_min, tzinfo=TZ)
    snaps = df["sampled_at"].unique()
    if len(snaps) == 0:
        return None
    diffs = [(abs((s - target).total_seconds()), s) for s in snaps]
    best = min(diffs, key=lambda x: x[0])
    # Only accept if within 90 minutes of target (avoids using wrong-day data)
    if best[0] > 90 * 60:
        return None
    return best[1]


def compute_walls(df: pd.DataFrame, snap_ts: pd.Timestamp) -> tuple[float | None, float | None, float | None]:
    """Compute (call_wall, put_wall, spot) for a given snapshot across ALL expirations."""
    snap = df[df["sampled_at"] == snap_ts].copy()
    if snap.empty:
        return None, None, None

    spot = float(snap["underlying_price"].dropna().median())
    if pd.isna(spot) or spot <= 0:
        return None, None, None

    call_wall, put_wall = find_raw_wall_strikes(snap, spot)
    return call_wall, put_wall, spot


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = get_duckdb_conn()

    # Collect all trading days in range (we'll discover them from available parquets)
    all_dates = []
    d = START_DATE
    while d <= END_DATE:
        if d.weekday() < 5:  # Mon–Fri
            all_dates.append(d)
        d += timedelta(days=1)

    rows = []
    skipped = 0

    for trade_date in tqdm(all_dates, desc="Intraday GEX walls"):
        day_df = load_day(con, trade_date)
        if day_df is None:
            skipped += 1
            continue

        row = {"date": trade_date}
        all_ok = True

        for label, hour, minute in TARGETS:
            snap_ts = nearest_snapshot(day_df, trade_date, hour, minute)
            if snap_ts is None:
                row[f"snap_{label}"]       = None
                row[f"call_wall_{label}"]  = None
                row[f"put_wall_{label}"]   = None
                row[f"spot_{label}"]       = None
                all_ok = False
                continue

            call_wall, put_wall, spot = compute_walls(day_df, snap_ts)
            row[f"snap_{label}"]       = snap_ts
            row[f"call_wall_{label}"]  = call_wall
            row[f"put_wall_{label}"]   = put_wall
            row[f"spot_{label}"]       = spot

        rows.append(row)

    con.close()

    out = pd.DataFrame(rows)
    # Reorder columns for readability
    cols = ["date",
            "spot_morning", "call_wall_morning", "put_wall_morning", "snap_morning",
            "spot_midday",  "call_wall_midday",  "put_wall_midday",  "snap_midday",
            "spot_close",   "call_wall_close",   "put_wall_close",   "snap_close"]
    out = out[[c for c in cols if c in out.columns]]

    out_path = OUT_DIR / "intraday_levels.csv"
    out.to_csv(out_path, index=False)

    print(f"\nDone.")
    print(f"  Days processed: {len(rows)} | Skipped (no data): {skipped}")
    print(f"  Saved → {out_path}")

    # Quick summary: how often do walls move?
    for side in ["call", "put"]:
        m = out[f"{side}_wall_morning"]
        c = out[f"{side}_wall_close"]
        valid = out.dropna(subset=[f"{side}_wall_morning", f"{side}_wall_close"])
        moved = (valid[f"{side}_wall_morning"] != valid[f"{side}_wall_close"]).sum()
        pct = moved / len(valid) * 100 if len(valid) > 0 else 0
        print(f"  {side.capitalize()} wall moved morning→close: {moved}/{len(valid)} days ({pct:.1f}%)")


if __name__ == "__main__":
    main()
