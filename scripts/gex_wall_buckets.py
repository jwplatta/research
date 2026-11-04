"""GEX Wall Solidity — Bucket Analysis

1. VIX buckets: bounce/hold rates by VIX level at open
2. 0DTE GEX skew: (call_gex - put_gex) / total_gex per day, then bucket analysis

Outputs:
  out/gex_walls/buckets_vix.csv
  out/gex_walls/buckets_skew.csv
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from calc.gex import net_gex_by_strike
from util.minio import get_duckdb_conn, option_parquet_uris

TZ       = "America/New_York"
PROVIDER = "schwab"
OUT_DIR  = Path("out/gex_walls")

COLUMNS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price",
]

OUTCOMES = ["bounce_30m", "bounce_60m", "eod_held"]


# ---------------------------------------------------------------------------
# Load features (built by gex_wall_features.py)
# ---------------------------------------------------------------------------

def load_events() -> pd.DataFrame:
    df = pd.read_csv("out/gex_walls/features.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["first_approach_ts"] = pd.to_datetime(df["first_approach_ts"], utc=True).dt.tz_convert(TZ)
    for col in OUTCOMES:
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


# ---------------------------------------------------------------------------
# 0DTE GEX skew
# ---------------------------------------------------------------------------

def compute_0dte_skew(con, trading_days: list[date]) -> pd.Series:
    """
    For each trading day, load the open snapshot (~9:40 ET) for the 0DTE expiry only,
    compute total call GEX and put GEX, return normalized skew per day.

    skew = (call_gex - |put_gex|) / (call_gex + |put_gex|)
    Positive = call-dominant = more resistance above spot
    Negative = put-dominant  = more support below spot
    """
    skews = {}
    for trade_date in trading_days:
        uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
        uri = uris[0]
        try:
            con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
        except Exception:
            continue

        cols = ", ".join(COLUMNS)
        df = con.execute(f"""
            SELECT {cols} FROM read_parquet('{uri}', union_by_name=true)
        """).fetchdf()
        df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
        df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
        for c in ["gamma", "open_interest", "underlying_price"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Open snapshot: first at or after 9:40 ET
        cutoff = df["sampled_at"].iloc[0].replace(hour=9, minute=40, second=0, microsecond=0)
        snap = df[df["sampled_at"] >= cutoff]
        if snap.empty:
            continue
        snap = snap[snap["sampled_at"] == snap["sampled_at"].min()]

        # 0DTE only
        snap_0dte = snap[snap["expiration_date"] == trade_date]
        if snap_0dte.empty:
            continue

        spot_vals = snap_0dte["underlying_price"].dropna()
        if spot_vals.empty:
            continue
        spot = float(spot_vals.median())

        gex_df = net_gex_by_strike(snap_0dte, spot=spot, strike_range=300.0).dropna(subset=["net_gex"])
        if gex_df.empty:
            continue

        call_gex = gex_df[gex_df["net_gex"] > 0]["net_gex"].sum()
        put_gex  = gex_df[gex_df["net_gex"] < 0]["net_gex"].abs().sum()
        total    = call_gex + put_gex
        if total == 0:
            continue

        skews[trade_date] = (call_gex - put_gex) / total

    return pd.Series(skews, name="gex_skew_0dte")


# ---------------------------------------------------------------------------
# Bucket analysis helper
# ---------------------------------------------------------------------------

def bucket_summary(df: pd.DataFrame, bucket_col: str, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for bucket, grp in df.groupby(bucket_col):
        row = {"bucket": bucket, "n": len(grp)}
        for col in outcomes:
            sub = grp[col].dropna()
            sub = sub[sub.isin([True, False])]
            row[f"{col}_pct"] = round(sub.mean() * 100, 1) if len(sub) > 0 else np.nan
            row[f"{col}_n"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def print_bucket_table(title: str, df: pd.DataFrame, outcomes: list[str]) -> None:
    print(f"\n=== {title} ===")
    hdr = f"{'Bucket':<20} {'N':>6}"
    for o in outcomes:
        hdr += f"  {o[:12]:>12}"
    print(hdr)
    print("-" * (len(hdr) + 4))
    for _, r in df.iterrows():
        row = f"{str(r['bucket']):<20} {int(r['n']):>6}"
        for o in outcomes:
            val = r.get(f"{o}_pct", np.nan)
            row += f"  {val:>11.1f}%" if not np.isnan(val) else f"  {'N/A':>12}"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading events with features...")
    events = load_events()
    print(f"  {len(events):,} tested events, {events['date'].nunique()} days")

    # -----------------------------------------------------------------------
    # 1. VIX buckets
    # -----------------------------------------------------------------------
    vix_bins   = [0, 15, 20, 25, 30, 999]
    vix_labels = ["<15", "15-20", "20-25", "25-30", ">30"]
    events["vix_bucket"] = pd.cut(
        events["vix_open"], bins=vix_bins, labels=vix_labels, right=False
    )

    vix_summary = bucket_summary(events.dropna(subset=["vix_open"]), "vix_bucket", OUTCOMES)
    print_bucket_table("Bounce/Hold Rate by VIX Bucket", vix_summary, OUTCOMES)
    vix_summary.to_csv(OUT_DIR / "buckets_vix.csv", index=False)

    # Also show event counts per bucket
    print("\nVIX bucket distribution:")
    print(events["vix_bucket"].value_counts().sort_index().to_string())

    # -----------------------------------------------------------------------
    # 2. 0DTE GEX skew
    # -----------------------------------------------------------------------
    print("\nComputing 0DTE GEX skew per day...")
    con = get_duckdb_conn()
    trading_days = sorted(events["date"].unique())
    skew_series = compute_0dte_skew(con, trading_days)
    con.close()
    print(f"  Computed skew for {len(skew_series)} / {len(trading_days)} days")
    print(f"  skew range: {skew_series.min():.3f} → {skew_series.max():.3f}  median: {skew_series.median():.3f}")

    events["gex_skew_0dte"] = events["date"].map(skew_series)

    skew_bins   = [-1.01, -0.5, -0.2, 0.0, 0.2, 0.5, 1.01]
    skew_labels = ["<-0.5 (put dom)", "-0.5 to -0.2", "-0.2 to 0", "0 to 0.2", "0.2 to 0.5", ">0.5 (call dom)"]
    events["skew_bucket"] = pd.cut(
        events["gex_skew_0dte"], bins=skew_bins, labels=skew_labels, right=False
    )

    skew_summary = bucket_summary(events.dropna(subset=["gex_skew_0dte"]), "skew_bucket", OUTCOMES)
    print_bucket_table("Bounce/Hold Rate by 0DTE GEX Skew", skew_summary, OUTCOMES)
    skew_summary.to_csv(OUT_DIR / "buckets_skew.csv", index=False)

    # Also: separate call wall vs put wall by skew
    print("\n=== Call walls only: Hold Rate by 0DTE GEX Skew ===")
    call_ev = events[events["wall_type"] == "call"].dropna(subset=["gex_skew_0dte"])
    cs = bucket_summary(call_ev, "skew_bucket", OUTCOMES)
    print_bucket_table("Call walls", cs, OUTCOMES)

    print("\n=== Put walls only: Hold Rate by 0DTE GEX Skew ===")
    put_ev = events[events["wall_type"] == "put"].dropna(subset=["gex_skew_0dte"])
    ps = bucket_summary(put_ev, "skew_bucket", OUTCOMES)
    print_bucket_table("Put walls", ps, OUTCOMES)

    # Save combined
    events[["date", "gex_skew_0dte"]].drop_duplicates().to_csv(
        OUT_DIR / "daily_skew.csv", index=False
    )
    print(f"\nSaved → {OUT_DIR}/buckets_vix.csv, buckets_skew.csv, daily_skew.csv")


if __name__ == "__main__":
    main()
