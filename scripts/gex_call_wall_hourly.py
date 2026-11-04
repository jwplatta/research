"""Build hourly call wall dataset for the training window (Dec 2025–Aug 2026).

For each trading day, finds the nearest SPXW option chain snapshot to each
hour boundary during RTH (10:00, 11:00, 12:00, 13:00, 14:00, 15:00 ET)
and computes call wall features at that snapshot.

Output: out/gex_walls/call_wall_hourly.csv

Columns:
  date, hour, snap_ts,
  call_wall,         — strike with highest positive net GEX (all expiries)
  gex_at_wall,       — net GEX at that strike
  spot,              — underlying price at snapshot
  dist_to_wall,      — call_wall - spot (positive = wall above spot)
  dist_to_zgl,       — spot - zero_gamma_level (positive = spot above ZGL)
  vix,               — VIX close at nearest 1-min bar
  atm_iv,            — 0DTE ATM implied vol (decimal)
  iv_25d_call,       — 0DTE 25-delta call IV (percentage points)
  iv_25d_put,        — 0DTE 25-delta put IV (percentage points)
  gex_wall_p10,      — net GEX at call_wall + 10 pts
  gex_wall_m10,      — net GEX at call_wall - 10 pts
  spy_vol_zscore,    — SPY volume at snap bar / median vol at that minute-of-day
  minute_of_day,     — hour*60 + minute of snap_ts
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_raw_wall_strikes, net_gex_by_strike, find_zero_gamma_level, gex_at_strike
from util.minio import get_duckdb_conn, option_parquet_uris

TZ       = ZoneInfo("America/New_York")
PROVIDER = "schwab"
OUT_DIR  = Path("out/gex_walls")

START_DATE = date(2025, 12, 1)
END_DATE   = date(2026, 8, 31)

# Hour targets (ET) — beginning of each hour during RTH
HOUR_TARGETS = [10, 11, 12, 13, 14, 15]
MAX_SNAP_GAP_MIN = 30  # reject snapshot if >30 min from target


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

LOCAL_VIX_DEC2025 = Path("data/candles/vix_1min_dec2025.parquet")
LOCAL_SPY_DEC2025 = Path("data/candles/spy_1min_dec2025.parquet")


def load_vix_1min(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT datetime, close
        FROM read_parquet('s3://tractatus-lakehouse/candles/ibkr-paper/1min/VIX.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")

    # Prepend locally-fetched Dec 2025 bars (not in the lakehouse parquet)
    if LOCAL_VIX_DEC2025.exists():
        local = pd.read_parquet(LOCAL_VIX_DEC2025, columns=["datetime", "close"])
        local["datetime"] = pd.to_datetime(local["datetime"], utc=True).dt.tz_convert("America/New_York")
        df = pd.concat([local, df], ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    else:
        print("WARNING: data/candles/vix_1min_dec2025.parquet not found — Dec 2025 VIX will be NaN")

    return df.set_index("datetime")


def load_spy_1min(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT datetime, volume
        FROM read_parquet('s3://tractatus-lakehouse/candles/ibkr-paper/1min/SPY.parquet')
        ORDER BY datetime
    """).fetchdf()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")

    if LOCAL_SPY_DEC2025.exists():
        local = pd.read_parquet(LOCAL_SPY_DEC2025, columns=["datetime", "volume"])
        local["datetime"] = pd.to_datetime(local["datetime"], utc=True).dt.tz_convert("America/New_York")
        df = pd.concat([local, df], ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    else:
        print("WARNING: data/candles/spy_1min_dec2025.parquet not found — Dec 2025 SPY vol will be NaN")

    return df.set_index("datetime")


def build_spy_vol_zscore_lookup(spy: pd.DataFrame) -> dict[int, float]:
    """Median SPY volume by minute-of-day across all RTH bars."""
    rth = spy.between_time("09:30", "16:00").copy()
    rth["mod"] = rth.index.hour * 60 + rth.index.minute
    return rth.groupby("mod")["volume"].median().to_dict()


def load_day_options(con, trade_date: date) -> pd.DataFrame | None:
    """Load full RTH option chain for trade_date. Returns None if unavailable."""
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
               gamma, open_interest, underlying_price,
               delta, volatility
        FROM read_parquet('{uri}', union_by_name=true)
        WHERE gamma IS NOT NULL
          AND open_interest IS NOT NULL
          AND gamma > 0
          AND open_interest > 0
    """).fetchdf()

    if df.empty:
        return None

    df["sampled_at"]       = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert("America/New_York")
    df["expiration_date"]  = pd.to_datetime(df["expiration_date"]).dt.date
    df["strike"]           = pd.to_numeric(df["strike"], errors="coerce")
    df["gamma"]            = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"]    = pd.to_numeric(df["open_interest"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["delta"]            = pd.to_numeric(df["delta"], errors="coerce")
    df["volatility"]       = pd.to_numeric(df["volatility"], errors="coerce")

    rth = df[
        (df["sampled_at"].dt.hour >= 9) & (df["sampled_at"].dt.hour < 16)
        & (df["sampled_at"].dt.date == trade_date)
    ]
    return rth if not rth.empty else None


# ---------------------------------------------------------------------------
# Per-snapshot computations
# ---------------------------------------------------------------------------

def nearest_snapshot(df: pd.DataFrame, trade_date: date, target_hour: int) -> pd.Timestamp | None:
    target = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=target_hour, minute=0, tzinfo=TZ)
    snaps = df["sampled_at"].unique()
    if len(snaps) == 0:
        return None
    diffs = [(abs((s - target).total_seconds()), s) for s in snaps]
    best = min(diffs, key=lambda x: x[0])
    if best[0] > MAX_SNAP_GAP_MIN * 60:
        return None
    return best[1]


def nearest_0dte_snap(day_df: pd.DataFrame, trade_date: date, snap_ts: pd.Timestamp) -> pd.DataFrame:
    """Return the 0DTE subset from the snapshot nearest to snap_ts that has 0DTE data.

    Searches all snapshots in day_df, picks the one with minimum time distance
    to snap_ts that has at least one 0DTE row with valid volatility.
    """
    snaps_with_0dte = (
        day_df[day_df["expiration_date"] == trade_date]
        .dropna(subset=["volatility"])
        .groupby("sampled_at")
        .filter(lambda g: len(g) > 0)
        ["sampled_at"].unique()
    )
    if len(snaps_with_0dte) == 0:
        return pd.DataFrame()
    diffs = [(abs((s - snap_ts).total_seconds()), s) for s in snaps_with_0dte]
    best_ts = min(diffs, key=lambda x: x[0])[1]
    return day_df[(day_df["sampled_at"] == best_ts) & (day_df["expiration_date"] == trade_date)].copy()


def compute_atm_iv(snap0dte: pd.DataFrame, spot: float) -> float | None:
    """ATM IV from 0DTE chain: mean volatility at the nearest-to-ATM strike."""
    s = snap0dte.dropna(subset=["volatility", "strike"])
    s = s[s["volatility"] > 0]
    if s.empty:
        return None
    s = s.copy()
    s["dist"] = (s["strike"] - spot).abs()
    atm = s[s["dist"] == s["dist"].min()]
    return float(atm["volatility"].mean())


def compute_25d_ivs(snap0dte: pd.DataFrame) -> tuple[float | None, float | None]:
    """25-delta call and put IVs from the 0DTE chain.

    Finds the call with delta nearest to +0.25 and put with delta nearest to -0.25.
    Returns IVs as percentage points (multiply decimal vol × 100).
    """
    s = snap0dte.dropna(subset=["delta", "volatility"])
    s = s[s["volatility"] > 0]
    if s.empty:
        return None, None

    calls = s[s["contract_type"].str.upper() == "CALL"].copy()
    puts  = s[s["contract_type"].str.upper() == "PUT"].copy()

    iv_call = None
    if not calls.empty:
        calls["d25_dist"] = (calls["delta"].abs() - 0.25).abs()
        best = calls.loc[calls["d25_dist"].idxmin()]
        iv_call = float(best["volatility"])

    iv_put = None
    if not puts.empty:
        puts["d25_dist"] = (puts["delta"].abs() - 0.25).abs()
        best = puts.loc[puts["d25_dist"].idxmin()]
        iv_put = float(best["volatility"])

    return iv_call, iv_put


def lookup_vix(vix: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Last VIX bar at or before ts (pad/ffill — never looks forward)."""
    idx = vix.index.get_indexer([ts], method="pad")
    if idx[0] < 0:
        return None
    return float(vix.iloc[idx[0]]["close"])


def lookup_spy_vol_zscore(
    spy: pd.DataFrame,
    median_by_mod: dict[int, float],
    ts: pd.Timestamp,
) -> float | None:
    idx = spy.index.get_indexer([ts], method="nearest")
    if idx[0] < 0:
        return None
    gap = abs((spy.index[idx[0]] - ts).total_seconds())
    if gap > 10 * 60:
        return None
    vol = float(spy.iloc[idx[0]]["volume"])
    mod = ts.hour * 60 + ts.minute
    med = median_by_mod.get(mod)
    if not med or med <= 0:
        return None
    return vol / med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = get_duckdb_conn()

    print("Loading VIX and SPY 1-min candles...")
    vix = load_vix_1min(con)
    spy = load_spy_1min(con)
    spy_median_by_mod = build_spy_vol_zscore_lookup(spy)

    all_dates = []
    d = START_DATE
    while d <= END_DATE:
        if d.weekday() < 5:
            all_dates.append(d)
        d += timedelta(days=1)

    rows = []
    skipped = 0

    for trade_date in tqdm(all_dates, desc="Call wall hourly"):
        day_df = load_day_options(con, trade_date)
        if day_df is None:
            skipped += 1
            continue

        for hour in HOUR_TARGETS:
            snap_ts = nearest_snapshot(day_df, trade_date, hour)
            if snap_ts is None:
                continue

            snap = day_df[day_df["sampled_at"] == snap_ts].copy()
            if snap.empty:
                continue

            spot = float(snap["underlying_price"].dropna().median())
            if pd.isna(spot) or spot <= 0:
                continue

            # Call wall and GEX structure
            call_wall, _ = find_raw_wall_strikes(snap, spot)
            if call_wall is None:
                continue

            gex_wall  = gex_at_strike(snap, spot, call_wall)
            zgl       = find_zero_gamma_level(snap, spot)
            gex_p10   = gex_at_strike(snap, spot, call_wall + 10)
            gex_m10   = gex_at_strike(snap, spot, call_wall - 10)

            # 0DTE chain for IV features — use nearest snapshot that has 0DTE data
            snap0 = nearest_0dte_snap(day_df, trade_date, snap_ts)
            atm_iv         = compute_atm_iv(snap0, spot) if not snap0.empty else None
            iv_25c, iv_25p = compute_25d_ivs(snap0) if not snap0.empty else (None, None)

            # VIX and SPY vol at this bar
            snap_ts_aware = snap_ts if snap_ts.tzinfo is not None else snap_ts.tz_localize("America/New_York")
            vix_val  = lookup_vix(vix, snap_ts_aware)
            spy_zsc  = lookup_spy_vol_zscore(spy, spy_median_by_mod, snap_ts_aware)

            rows.append({
                "date":          trade_date,
                "hour":          hour,
                "snap_ts":       snap_ts,
                "call_wall":     call_wall,
                "gex_at_wall":   gex_wall,
                "spot":          round(spot, 2),
                "dist_to_wall":  round(call_wall - spot, 2),
                "dist_to_zgl":   round(spot - zgl, 2) if zgl is not None else None,
                "vix":           round(vix_val, 3) if vix_val is not None else None,
                "atm_iv":        round(atm_iv, 4) if atm_iv is not None else None,
                "iv_25d_call":   round(iv_25c, 2) if iv_25c is not None else None,
                "iv_25d_put":    round(iv_25p, 2) if iv_25p is not None else None,
                "gex_wall_p10":  gex_p10,
                "gex_wall_m10":  gex_m10,
                "spy_vol_zscore": round(spy_zsc, 3) if spy_zsc is not None else None,
                "minute_of_day": snap_ts.hour * 60 + snap_ts.minute,
            })

    con.close()

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "call_wall_hourly.csv"
    out.to_csv(out_path, index=False)

    print(f"\nDone.")
    print(f"  Rows: {len(out):,} | Trading days skipped: {skipped}")
    print(f"  Saved → {out_path}")
    if not out.empty:
        print(f"  Dates: {out['date'].min()} – {out['date'].max()}")
        print(f"  Hours covered: {sorted(out['hour'].unique())}")
        for col in ["vix", "atm_iv", "iv_25d_call", "iv_25d_put", "dist_to_zgl"]:
            cov = out[col].notna().mean() * 100
            print(f"  {col} coverage: {cov:.1f}%")


if __name__ == "__main__":
    main()
