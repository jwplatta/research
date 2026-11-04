"""GEX Wall Solidity Feature Study

Joins candidate features onto wall approach events and measures
correlation with hold/break outcome (bounce_30m, bounce_60m, eod_held).

Features built here:
  - dist_to_wall:        abs(wall_level - spot) at snapshot time
  - wall_side:           call=1, put=-1
  - approach_hour:       hour of first_approach_ts (time-of-day)
  - approach_minute:     minute of day
  - vix_open:            VIX close at 9:30 ET on trade date
  - realized_vol_5d:     annualized realized vol over prior 5 trading days (SPX 1-min)
  - spy_vol_ratio:       SPY volume at approach bar / median SPY volume at that minute-of-day
  - is_method_raw:       one-hot for wall method

Outputs:
  out/gex_walls/features.csv    — events with features, tested-only rows
  out/gex_walls/correlations.csv — point-biserial correlations with outcomes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from util.minio import get_duckdb_conn

TZ = "America/New_York"
OUT_DIR = Path("out/gex_walls")


# ---------------------------------------------------------------------------
# Load candles from lakehouse
# ---------------------------------------------------------------------------

def load_spy_1min(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT datetime, open, high, low, close, volume
        FROM read_parquet('s3://tractatus-lakehouse/candles/ibkr-paper/1min/SPY.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
    df = df.set_index("datetime")
    return df


def load_vix_1min(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT datetime, close
        FROM read_parquet('s3://tractatus-lakehouse/candles/ibkr-paper/1min/VIX.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
    df = df.set_index("datetime")
    return df


def load_spx_1min(con) -> pd.DataFrame:
    """Load SPX 1-min from schwab (Feb 2026+) with ibkr fallback."""
    df = con.execute("""
        SELECT datetime, close
        FROM read_parquet('s3://tractatus-lakehouse/candles/schwab/1min/SPX.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
    df = df.set_index("datetime")
    return df


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def build_vix_open(vix: pd.DataFrame) -> pd.Series:
    """First VIX bar at or after 9:30 ET per date."""
    rth = vix.between_time("09:30", "09:35")
    return rth.groupby(rth.index.date)["close"].first().rename("vix_open")


def build_realized_vol_5d(spx: pd.DataFrame) -> pd.Series:
    """
    Annualized realized vol (5-day lookback) per date.
    Computed from RTH 1-min log returns, annualized assuming 252 days * 390 bars.
    """
    rth = spx.between_time("09:30", "16:00").copy()
    rth["log_ret"] = np.log(rth["close"] / rth["close"].shift(1))
    # Group by date, compute daily realized vol
    daily_var = rth.groupby(rth.index.date)["log_ret"].var() * 390
    # Rolling 5-day mean of variance, then sqrt and annualize
    rv5 = daily_var.rolling(5).mean().apply(np.sqrt) * np.sqrt(252)
    return rv5.rename("realized_vol_5d")


def build_spy_vol_ratio(spy: pd.DataFrame, approach_ts_series: pd.Series) -> pd.Series:
    """
    For each approach timestamp, compute SPY volume at that bar /
    median SPY volume for that minute-of-day across all days.
    Returns a Series aligned to the input index.
    """
    rth = spy.between_time("09:30", "16:00").copy()
    rth["minute_of_day"] = rth.index.hour * 60 + rth.index.minute
    median_by_minute = rth.groupby("minute_of_day")["volume"].median()

    ratios = []
    for ts in approach_ts_series:
        if pd.isna(ts):
            ratios.append(np.nan)
            continue
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(TZ)
        else:
            ts = ts.tz_convert(TZ)
        bar = spy.index.get_indexer([ts], method="nearest")
        if bar[0] < 0:
            ratios.append(np.nan)
            continue
        vol = spy.iloc[bar[0]]["volume"]
        mod = ts.hour * 60 + ts.minute
        med = median_by_minute.get(mod, np.nan)
        ratios.append(vol / med if med and med > 0 else np.nan)
    return pd.Series(ratios, index=approach_ts_series.index, name="spy_vol_ratio")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading events...")
    events = pd.read_csv("out/gex_walls/events.csv")
    # Keep only tested rows with a valid approach timestamp
    events = events[events["tested"] == True].copy()
    events = events[events["first_approach_ts"].notna()].copy()
    print(f"  {len(events):,} tested events")

    # Parse timestamps
    events["first_approach_ts"] = pd.to_datetime(events["first_approach_ts"], utc=True).dt.tz_convert(TZ)
    events["date"] = pd.to_datetime(events["date"]).dt.date

    # Outcome columns — coerce to bool
    for col in ["bounce_30m", "bounce_60m", "eod_held"]:
        events[col] = events[col].map({"True": True, "False": False, True: True, False: False})

    print("Loading candles from lakehouse...")
    con = get_duckdb_conn()
    spy = load_spy_1min(con)
    vix = load_vix_1min(con)
    spx = load_spx_1min(con)
    con.close()
    print(f"  SPY: {len(spy):,} bars  VIX: {len(vix):,} bars  SPX: {len(spx):,} bars")

    # --- Build features ---
    print("Building features...")

    # 1. Distance from spot to wall
    events["dist_to_wall"] = (events["wall_level"] - events["spot"]).abs()

    # 2. Wall side: call=1, put=-1
    events["wall_side"] = events["wall_type"].map({"call": 1, "put": -1})

    # 3. Time of day
    events["approach_hour"] = events["first_approach_ts"].dt.hour
    events["approach_minute_of_day"] = (
        events["first_approach_ts"].dt.hour * 60 + events["first_approach_ts"].dt.minute
    )

    # 4. VIX at open
    vix_open = build_vix_open(vix)
    events["vix_open"] = events["date"].map(vix_open)
    print(f"  VIX open: {events['vix_open'].notna().sum():,} / {len(events):,} events matched")

    # 5. Realized vol 5d (SPX)
    rv5 = build_realized_vol_5d(spx)
    events["realized_vol_5d"] = events["date"].map(rv5)
    print(f"  Realized vol 5d: {events['realized_vol_5d'].notna().sum():,} / {len(events):,} events matched")

    # 6. SPY volume ratio at approach
    print("  Building SPY volume ratios (may be slow)...")
    events["spy_vol_ratio"] = build_spy_vol_ratio(spy, events["first_approach_ts"])
    print(f"  SPY vol ratio: {events['spy_vol_ratio'].notna().sum():,} / {len(events):,} events matched")

    # 7. Method one-hots
    for method in ["raw", "weighted", "clustered", "zones"]:
        events[f"is_{method}"] = (events["method"] == method).astype(int)

    # --- Save features ---
    feat_path = OUT_DIR / "features.csv"
    events.to_csv(feat_path, index=False)
    print(f"\nSaved {len(events):,} rows → {feat_path}")

    # --- Correlation analysis ---
    feature_cols = [
        "dist_to_wall", "wall_side",
        "approach_hour", "approach_minute_of_day",
        "vix_open", "realized_vol_5d", "spy_vol_ratio",
        "is_raw", "is_weighted", "is_clustered", "is_zones",
    ]
    outcome_cols = ["bounce_30m", "bounce_60m", "eod_held"]

    print("\n=== Point-biserial correlations with hold/break outcomes ===")
    rows = []
    for feat in feature_cols:
        row = {"feature": feat}
        for outcome in outcome_cols:
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
    corr_path = OUT_DIR / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    # Print table
    print(f"\n{'Feature':<26} {'r_bounce30':>11} {'r_bounce60':>11} {'r_eod_held':>11}  {'p_bounce30':>11}")
    print("-" * 75)
    for _, r in corr_df.iterrows():
        print(
            f"{r['feature']:<26} "
            f"{r['r_bounce_30m']:>10.4f}  "
            f"{r['r_bounce_60m']:>10.4f}  "
            f"{r['r_eod_held']:>10.4f}  "
            f"{r['p_bounce_30m']:>10.4f}"
        )

    print(f"\nSaved correlations → {corr_path}")


if __name__ == "__main__":
    main()
