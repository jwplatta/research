"""Build master feature dataset for GEX wall solidity classifiers.

Merges:
  - features2.csv          (base: per-event features from options + candles)
  - daily_rr.csv           (25Δ risk reversal + iv_25d_call/put per day)
  - daily_skew.csv         (0DTE GEX call/put skew per day)
  - daily_atm_iv.csv       (0DTE ATM IV per day, from gex_wall_iv_zscore.py)

Derives:
  - atm_iv_zscore_{5,10,20}d  rolling z-scores of ATM IV
  - vix_zscore_{5,10,20}d     rolling z-scores of VIX open

Output: out/gex_walls/master_features.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

OUTCOMES  = ["bounce_30m", "bounce_60m", "eod_held"]
BOOL_COLS = ["bounce_30m", "bounce_60m", "eod_held", "eod_crossed", "eod_returned",
             "wall_breached_at_open", "is_raw", "is_weighted", "is_clustered", "is_zones"]
LOOKBACKS = [5, 10, 20]

OUT_DIR = Path("out/gex_walls")


def load_base() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "features2.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


def load_daily(filename: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / filename)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df.set_index(date_col)


def rolling_zscore(series: pd.Series, lb: int) -> pd.Series:
    mu = series.rolling(lb).mean()
    sd = series.rolling(lb).std()
    return (series - mu) / sd


def build_daily_features(atm_iv_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling z-score features that are daily (not per-event)."""
    atm_iv = atm_iv_daily["atm_iv"].copy()
    atm_iv.index = pd.to_datetime(atm_iv.index)

    # VIX open from features2 — reconstruct daily series
    # (already joined onto events; we recompute from daily_atm_iv dates)
    rows = {}
    for lb in LOOKBACKS:
        rows[f"atm_iv_zscore_{lb}d"] = rolling_zscore(atm_iv, lb)
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading base features (features2.csv)...")
    base = load_base()
    print(f"  {len(base):,} events, {base['date'].nunique()} days")

    # --- Daily data sources ---
    print("Loading daily sources...")
    rr     = load_daily("daily_rr.csv")       # rr, iv_25d_call, iv_25d_put, strike_25d_call, strike_25d_put
    skew   = load_daily("daily_skew.csv")      # gex_skew_0dte
    atm_iv = load_daily("daily_atm_iv.csv")    # atm_iv

    # --- Rolling z-scores from ATM IV and VIX ---
    # Use ALL event dates as the index so gap days (missing SPXW parquet) don't
    # cause NaN z-scores on surrounding dates. Forward-fill ATM IV across gaps
    # (carry-forward is fine here — no future leakage, just using last known IV estimate).
    all_event_dates = pd.to_datetime(
        sorted(base["date"].unique())
    )
    full_idx = pd.DatetimeIndex(all_event_dates)

    atm_iv_ts = (
        atm_iv["atm_iv"]
        .copy()
        .pipe(lambda s: s.set_axis(pd.to_datetime(s.index)))
        .reindex(full_idx)
        .ffill()          # carry-forward for days with no SPXW parquet
        .sort_index()
    )

    # VIX open: reconstruct daily series from base events, reindex to full index
    vix_daily = (
        base[["date", "vix_open"]]
        .dropna(subset=["vix_open"])
        .drop_duplicates("date")
        .set_index("date")
        ["vix_open"]
        .pipe(lambda s: s.set_axis(pd.to_datetime(s.index)))
        .reindex(full_idx)
        .ffill()          # forward-fill isolated missing days so they don't poison rolling windows
        .sort_index()
    )

    zscore_df = pd.DataFrame(index=full_idx)
    for lb in LOOKBACKS:
        zscore_df[f"atm_iv_zscore_{lb}d"] = rolling_zscore(atm_iv_ts, lb)
        zscore_df[f"vix_zscore_{lb}d"]    = rolling_zscore(vix_daily, lb)
    zscore_df.index = zscore_df.index.date  # back to date objects for join

    # --- Join onto base ---
    base = base.join(rr[["rr", "iv_25d_call", "iv_25d_put"]],       on="date", how="left")
    base = base.join(skew[["gex_skew_0dte"]],                         on="date", how="left")
    base = base.join(atm_iv[["atm_iv"]],                              on="date", how="left")
    base = base.join(zscore_df,                                        on="date", how="left")

    # --- Validation ---
    n_total = len(base)
    missing_rr    = base["rr"].isna().sum()
    missing_iv    = base["atm_iv"].isna().sum()
    missing_skew  = base["gex_skew_0dte"].isna().sum()
    missing_z5    = base["atm_iv_zscore_5d"].isna().sum()

    print(f"\nMerge validation:")
    print(f"  Total events:          {n_total:,}")
    print(f"  Missing rr:            {missing_rr:,} ({missing_rr/n_total*100:.1f}%)")
    print(f"  Missing atm_iv:        {missing_iv:,} ({missing_iv/n_total*100:.1f}%)")
    print(f"  Missing gex_skew:      {missing_skew:,} ({missing_skew/n_total*100:.1f}%)")
    print(f"  Missing atm_iv_z5d:    {missing_z5:,} ({missing_z5/n_total*100:.1f}%)")

    # Warn if key columns have unexpected values
    for col in OUTCOMES:
        invalid = base[col].dropna().apply(lambda x: x not in [True, False]).sum()
        if invalid > 0:
            print(f"  WARNING: {col} has {invalid} non-boolean values")

    # --- Save ---
    out_path = OUT_DIR / "master_features.csv"
    base["date"] = pd.to_datetime(base["date"])  # serialize cleanly
    base.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"  Shape: {base.shape}")
    print(f"  Columns: {list(base.columns)}")


if __name__ == "__main__":
    main()
