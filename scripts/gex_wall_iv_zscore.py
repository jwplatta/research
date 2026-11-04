"""IV z-score feature analysis for GEX wall solidity.

Three variants:
  1. VIX z-score over 5/10/20/30/60d rolling window
  2. 0DTE ATM IV z-score over same windows
  3. IV/VIX ratio (0DTE ATM IV / VIX) — term structure proxy
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, "src")
from util.minio import get_duckdb_conn, option_parquet_uris

TZ       = "America/New_York"
PROVIDER = "schwab"
LOOKBACKS = [5, 10, 20, 30, 60]
OUTCOMES  = ["bounce_30m", "bounce_60m", "eod_held"]


def load_vix_open(con) -> pd.Series:
    df = con.execute("""
        SELECT datetime, close
        FROM read_parquet('s3://tractatus-lakehouse/candles/ibkr-paper/1min/VIX.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
    df = df.set_index("datetime")
    rth = df.between_time("09:30", "09:35")
    # group by date (plain date objects)
    return rth.groupby(rth.index.date)["close"].first()


def compute_atm_iv(con, trade_dates: list) -> pd.Series:
    atm_iv = {}
    for trade_date in tqdm(trade_dates, desc="ATM IV"):
        uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
        uri = uris[0]
        try:
            con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
        except Exception:
            continue
        df = con.execute(f"""
            SELECT sampled_at, expiration_date, strike, contract_type, volatility, underlying_price
            FROM read_parquet('{uri}', union_by_name=true)
        """).fetchdf()
        df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
        df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
        for c in ["volatility", "underlying_price", "strike"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        cutoff = df["sampled_at"].iloc[0].replace(hour=9, minute=40, second=0, microsecond=0)
        snap = df[df["sampled_at"] >= cutoff]
        if snap.empty:
            continue
        snap = snap[snap["sampled_at"] == snap["sampled_at"].min()]
        snap0 = snap[snap["expiration_date"] == trade_date].copy()
        if snap0.empty:
            continue

        spot = float(snap0["underlying_price"].dropna().median())
        snap0 = snap0.dropna(subset=["volatility", "strike"])
        snap0 = snap0[snap0["volatility"] > 0]
        if snap0.empty:
            continue

        snap0["dist_atm"] = (snap0["strike"] - spot).abs()
        atm_snap = snap0[snap0["dist_atm"] == snap0["dist_atm"].min()]
        atm_iv[trade_date] = float(atm_snap["volatility"].mean())

    return pd.Series(atm_iv)


def rolling_zscore(series: pd.Series, lb: int) -> pd.Series:
    mu = series.rolling(lb).mean()
    sd = series.rolling(lb).std()
    return (series - mu) / sd


def print_buckets(events: pd.DataFrame, col: str, bins, labels: list) -> None:
    events = events.copy()
    events["_bucket"] = pd.cut(events[col], bins=bins, labels=labels)
    print(f"{'Bucket':<14} {'N':>7}  {'bounce30':>9}  {'eod_held':>9}")
    for bucket, grp in events.dropna(subset=[col]).groupby("_bucket", observed=False):
        n = len(grp)
        b30 = grp["bounce_30m"].dropna(); b30 = b30[b30.isin([True, False])]
        eod = grp["eod_held"].dropna();   eod = eod[eod.isin([True, False])]
        b30_pct = f"{b30.mean()*100:>8.1f}%" if len(b30) > 0 else "       N/A"
        eod_pct = f"{eod.mean()*100:>8.1f}%" if len(eod) > 0 else "       N/A"
        print(f"{str(bucket):<14} {n:>7}  {b30_pct}  {eod_pct}")


def main() -> None:
    print("Loading events...")
    events = pd.read_csv("out/gex_walls/features.csv")
    events["date"] = pd.to_datetime(events["date"]).dt.date
    for col in OUTCOMES:
        events[col] = events[col].map({"True": True, "False": False, True: True, False: False})

    con = get_duckdb_conn()

    print("Loading VIX open series...")
    vix_open = load_vix_open(con)
    print(f"  {len(vix_open)} days, {vix_open.index.min()} → {vix_open.index.max()}")

    trade_dates = sorted(events["date"].unique().tolist())
    atm_iv = compute_atm_iv(con, trade_dates)
    con.close()
    print(f"ATM IV: {len(atm_iv)} days, range {atm_iv.min():.3f}–{atm_iv.max():.3f}, median {atm_iv.median():.3f}")

    # Save daily ATM IV for use by other scripts
    atm_iv_df = atm_iv.reset_index()
    atm_iv_df.columns = ["date", "atm_iv"]
    atm_iv_df["date"] = pd.to_datetime(atm_iv_df["date"])
    atm_iv_df.to_csv("out/gex_walls/daily_atm_iv.csv", index=False)
    print("Saved daily ATM IV → out/gex_walls/daily_atm_iv.csv")

    # Build all feature series
    features = {}
    for lb in LOOKBACKS:
        features[f"vix_zscore_{lb}d"]    = rolling_zscore(vix_open, lb)
        features[f"atm_iv_zscore_{lb}d"] = rolling_zscore(atm_iv, lb)

    # IV/VIX ratio: ATM IV is decimal (0.12 = 12%), VIX is percentage (12.0)
    features["iv_vix_ratio"] = atm_iv * 100 / vix_open.reindex(atm_iv.index)
    features["atm_iv_open"]  = atm_iv

    # Join onto events
    for name, series in features.items():
        events[name] = events["date"].map(series)

    # Correlations
    feat_names = list(features.keys())
    print(f"\n{'Feature':<24} {'r_bounce30':>11} {'r_bounce60':>11} {'r_eod_held':>11}   n")
    print("-" * 68)
    for feat in feat_names:
        row_vals = {}
        ns = []
        for outcome in OUTCOMES:
            sub = events[[feat, outcome]].dropna()
            sub = sub[sub[outcome].isin([True, False])]
            if len(sub) < 50:
                row_vals[outcome] = np.nan
            else:
                r, _ = stats.pointbiserialr(sub[outcome].astype(int), sub[feat].astype(float))
                row_vals[outcome] = r
            ns.append(len(sub))
        print(
            f"{feat:<24} "
            f"{row_vals.get('bounce_30m', np.nan):>10.4f}  "
            f"{row_vals.get('bounce_60m', np.nan):>10.4f}  "
            f"{row_vals.get('eod_held', np.nan):>10.4f}   "
            f"{min(ns)}"
        )

    # Bucket tables for best candidates
    print("\n=== VIX z-score 20d buckets ===")
    print_buckets(events, "vix_zscore_20d",
                  bins=[-np.inf, -1, 0, 1, np.inf],
                  labels=["<-1 (vol cooling)", "-1 to 0", "0 to +1", ">+1 (vol rising)"])

    print("\n=== ATM IV z-score 20d buckets ===")
    print_buckets(events, "atm_iv_zscore_20d",
                  bins=[-np.inf, -1, 0, 1, np.inf],
                  labels=["<-1 (IV cooling)", "-1 to 0", "0 to +1", ">+1 (IV rising)"])

    print("\n=== IV/VIX ratio buckets ===")
    print(f"  (ratio median: {events['iv_vix_ratio'].median():.3f}, range: {events['iv_vix_ratio'].min():.3f}–{events['iv_vix_ratio'].max():.3f})")
    print_buckets(events, "iv_vix_ratio",
                  bins=[0, 0.6, 0.8, 1.0, 1.2, 99],
                  labels=["<0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", ">1.2"])


if __name__ == "__main__":
    main()
