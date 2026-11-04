"""25-delta risk reversal skew vs wall solidity.

For each trading day compute 25Δ RR from the 0DTE open snapshot:
  RR = IV(25Δ call) - IV(25Δ put)

Positive RR = calls more expensive than puts (upside fear / unusual)
Negative RR = puts more expensive (downside fear / normal SPX skew)

Then bucket by RR level and by call vs put wall to see if skew
predicts wall hold/break.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, "src")
from util.minio import get_duckdb_conn, option_parquet_uris

# Reuse the risk reversal calc from options-monitor
sys.path.insert(0, str(Path.home() / "repos/options-monitor/src"))
from options_monitor.calc.risk_reversal import compute_risk_reversal

TZ       = "America/New_York"
PROVIDER = "schwab"
OUTCOMES = ["bounce_30m", "bounce_60m", "eod_held"]


def load_events() -> pd.DataFrame:
    df = pd.read_csv("out/gex_walls/features.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in OUTCOMES:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


def compute_daily_rr(con, trade_dates: list[date]) -> pd.DataFrame:
    """Compute 25Δ RR from 0DTE open snapshot for each trading day."""
    rows = []
    for trade_date in tqdm(trade_dates, desc="RR skew"):
        uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
        uri = uris[0]
        try:
            con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
        except Exception:
            continue

        df = con.execute(f"""
            SELECT sampled_at, expiration_date, strike, contract_type,
                   delta, volatility, underlying_price
            FROM read_parquet('{uri}', union_by_name=true)
        """).fetchdf()
        df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
        df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
        for c in ["delta", "volatility", "underlying_price", "strike"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Open snapshot
        cutoff = df["sampled_at"].iloc[0].replace(hour=9, minute=40, second=0, microsecond=0)
        snap = df[df["sampled_at"] >= cutoff]
        if snap.empty:
            continue
        snap = snap[snap["sampled_at"] == snap["sampled_at"].min()]

        # 0DTE only
        snap0 = snap[snap["expiration_date"] == trade_date].copy()
        if snap0.empty:
            continue

        result = compute_risk_reversal(snap0)
        if result is None:
            continue

        rows.append({
            "date":          trade_date,
            "rr":            result.rr,
            "iv_25d_call":   result.iv_25d_call,
            "iv_25d_put":    result.iv_25d_put,
            "strike_25d_call": result.strike_25d_call,
            "strike_25d_put":  result.strike_25d_put,
        })

    return pd.DataFrame(rows)


def print_buckets(events: pd.DataFrame, col: str, bins, labels: list, title: str) -> None:
    events = events.copy()
    events["_b"] = pd.cut(events[col], bins=bins, labels=labels)
    print(f"\n=== {title} ===")
    print(f"{'Bucket':<22} {'N':>6}  {'bounce30':>9}  {'bounce60':>9}  {'eod_held':>9}")
    print("-" * 65)
    for bucket, grp in events.dropna(subset=[col]).groupby("_b", observed=False):
        n = len(grp)
        vals = []
        for col2 in OUTCOMES:
            sub = grp[col2].dropna(); sub = sub[sub.isin([True, False])]
            vals.append(f"{sub.mean()*100:>8.1f}%" if len(sub) > 0 else "       N/A")
        print(f"{str(bucket):<22} {n:>6}  {'  '.join(vals)}")


def main() -> None:
    print("Loading events...")
    events = load_events()
    print(f"  {len(events):,} tested events, {events['date'].nunique()} days")

    con = get_duckdb_conn()
    trade_dates = sorted(events["date"].unique().tolist())
    rr_df = compute_daily_rr(con, trade_dates)
    con.close()

    print(f"\nRR computed for {len(rr_df)} / {len(trade_dates)} days")
    print(f"RR range: {rr_df['rr'].min():.4f} → {rr_df['rr'].max():.4f}")
    print(f"RR median: {rr_df['rr'].median():.4f}  mean: {rr_df['rr'].mean():.4f}")
    print(f"RR < 0 (put skew): {(rr_df['rr'] < 0).mean()*100:.0f}% of days")

    # Join onto events
    rr_map = rr_df.set_index("date")
    events["rr"]          = events["date"].map(rr_map["rr"])
    events["iv_25d_call"] = events["date"].map(rr_map["iv_25d_call"])
    events["iv_25d_put"]  = events["date"].map(rr_map["iv_25d_put"])

    print(f"\nEvents matched: {events['rr'].notna().sum():,} / {len(events):,}")

    # Correlation
    print("\n=== Correlations: RR vs outcomes ===")
    for feat in ["rr", "iv_25d_call", "iv_25d_put"]:
        for outcome in OUTCOMES:
            sub = events[[feat, outcome]].dropna()
            sub = sub[sub[outcome].isin([True, False])]
            if len(sub) < 50:
                continue
            r, p = stats.pointbiserialr(sub[outcome].astype(int), sub[feat].astype(float))
            print(f"  {feat} vs {outcome}: r={r:.4f}  p={p:.4f}  n={len(sub):,}")

    # Overall RR buckets
    bins   = [-np.inf, -0.05, -0.02, 0.0, 0.02, np.inf]
    labels = ["<-5% (strong put skew)", "-5% to -2%", "-2% to 0%", "0% to +2%", ">+2% (call skew)"]
    print_buckets(events, "rr", bins, labels, "All walls: hold rate by 25Δ RR")

    # Split by wall type
    print_buckets(
        events[events["wall_type"] == "call"], "rr", bins, labels,
        "Call walls only: hold rate by 25Δ RR"
    )
    print_buckets(
        events[events["wall_type"] == "put"], "rr", bins, labels,
        "Put walls only: hold rate by 25Δ RR"
    )

    # Save daily RR
    rr_df.to_csv("out/gex_walls/daily_rr.csv", index=False)
    print(f"\nSaved daily RR → out/gex_walls/daily_rr.csv")


if __name__ == "__main__":
    main()
