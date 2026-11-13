"""GEX Bear Call Spread Backtest

Self-contained backtest — all data pulled live from MinIO SPXW parquet files.
No dependency on pre-computed wall CSV files.

For each trading day:
  1. Load the SPXW option chain for the entry window (10:00–10:30 ET)
  2. Compute call wall and ZGL directly from the chain
  3. Get spot from underlying_price in the chain
  4. Place K1 = call_wall + 10pts (rounded to nearest 5pt strike)
  5. Read credit (mark spread) and delta from the chain
  6. Load the EOD snapshot (15:30–16:00 ET) for settlement proxy
  7. Compute P&L

Output: out/gex_walls/call_spread_backtest.csv + stdout summary
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_raw_wall_strikes, find_zero_gamma_level, net_gex_by_strike
from util.minio import get_duckdb_conn, option_parquet_uris

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TZ           = ZoneInfo("America/New_York")
PROVIDER     = "schwab"
START_DATE   = date(2025, 12, 1)
END_DATE     = date(2026, 11, 1)
OUT_DIR      = Path("out/gex_walls")
OUT_CSV      = OUT_DIR / "call_spread_backtest.csv"

SPREAD_WIDTH  = 20.0
CONTRACT_MULT = 100

# Entry window: 10:00–10:30 ET
ENTRY_START_H, ENTRY_START_M = 10, 0
ENTRY_END_H,   ENTRY_END_M   = 10, 30

# Fallback entry window: 10:30–11:00 ET
FALLBACK_START_H, FALLBACK_START_M = 10, 30
FALLBACK_END_H,   FALLBACK_END_M   = 11, 0

# EOD window: 15:30–16:00 ET
EOD_START_H, EOD_START_M = 15, 30
EOD_END_H,   EOD_END_M   = 16, 0

CHAIN_COLS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price", "volatility", "delta", "mark",
]

FILTERS = [
    ("cr≥$0.50", lambda c: c >= 0.50),
    ("cr≥$0.75", lambda c: c >= 0.75),
    ("cr≥$1.00", lambda c: c >= 1.00),
    ("cr≥$1.50", lambda c: c >= 1.50),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_day_chain(con, trade_date: date) -> pd.DataFrame | None:
    """Load all SPXW snapshots for trade_date. Returns None if file missing."""
    uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
    uri = uris[0]
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
    except Exception:
        return None

    cols = ", ".join(CHAIN_COLS)
    df = con.execute(f"""
        SELECT {cols}
        FROM read_parquet('{uri}', union_by_name=true)
        WHERE open_interest IS NOT NULL AND open_interest > 0
    """).fetchdf()
    if df.empty:
        return None

    df["sampled_at"]       = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
    df["expiration_date"]  = pd.to_datetime(df["expiration_date"]).dt.date
    df["strike"]           = pd.to_numeric(df["strike"],           errors="coerce")
    df["gamma"]            = pd.to_numeric(df["gamma"],            errors="coerce")
    df["open_interest"]    = pd.to_numeric(df["open_interest"],    errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df["volatility"]       = pd.to_numeric(df["volatility"],       errors="coerce")
    df["delta"]            = pd.to_numeric(df["delta"],            errors="coerce")
    df["mark"]             = pd.to_numeric(df["mark"],             errors="coerce")
    return df


def window_snapshot(day_df: pd.DataFrame, trade_date: date,
                    start_h: int, start_m: int,
                    end_h: int,   end_m: int) -> pd.DataFrame | None:
    """Return latest snapshot per (expiry, strike, contract_type) within a time window."""
    bstart = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=start_h, minute=start_m, tzinfo=TZ)
    bend   = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=end_h,   minute=end_m,   tzinfo=TZ)
    window = day_df[(day_df["sampled_at"] >= bstart) & (day_df["sampled_at"] < bend)]
    if window.empty:
        return None
    snap = (
        window.sort_values("sampled_at")
        .groupby(["expiration_date", "strike", "contract_type"], as_index=False)
        .last()
    )
    return snap if not snap.empty else None


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_iv_smile_residual(chain: pd.DataFrame, spot: float,
                               call_wall: float, trade_date: date) -> float | None:
    same_day = chain[
        (chain["expiration_date"] == trade_date)
        & (chain["contract_type"].str.upper() == "CALL")
        & (chain["strike"] > spot)
        & (chain["volatility"] > 0)
    ].dropna(subset=["volatility", "strike"])
    if len(same_day) < 5:
        return None
    log_m = np.log(same_day["strike"].values / spot)
    iv    = same_day["volatility"].values
    try:
        coeffs = np.polyfit(log_m, iv, 2)
    except Exception:
        return None
    wall_log_m = float(np.log(call_wall / spot))
    fitted_iv  = float(np.polyval(coeffs, wall_log_m))
    near = same_day[(same_day["strike"] - call_wall).abs() <= 2.5]
    if near.empty:
        return None
    return float(near["volatility"].median()) - fitted_iv


def find_mark_price(chain_0dte_calls: pd.DataFrame, strike: float) -> float | None:
    near = chain_0dte_calls[(chain_0dte_calls["strike"] - strike).abs() <= 2.5]
    if near.empty:
        return None
    vals = near["mark"].dropna()
    return float(vals.median()) if not vals.empty else None


def find_delta(chain_0dte_calls: pd.DataFrame, strike: float) -> float | None:
    near = chain_0dte_calls[(chain_0dte_calls["strike"] - strike).abs() <= 2.5]
    if near.empty:
        return None
    vals = near["delta"].dropna()
    return float(vals.median()) if not vals.empty else None


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------

def compute_pnl(K1: float, credit: float, settlement: float) -> float:
    K2 = K1 + SPREAD_WIDTH
    if settlement <= K1:
        return credit * CONTRACT_MULT
    if settlement >= K2:
        return -(SPREAD_WIDTH - credit) * CONTRACT_MULT
    return (credit - (settlement - K1)) * CONTRACT_MULT


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = []
    d = START_DATE
    while d < END_DATE:
        all_dates.append(d)
        d += timedelta(days=1)

    con = get_duckdb_conn()
    rows_by_filter: dict[str, list[dict]] = {label: [] for label, _ in FILTERS}

    for trade_date in tqdm(all_dates, desc="Days", unit="day"):
        day_df = load_day_chain(con, trade_date)
        if day_df is None:
            continue  # weekend / holiday / no data

        # --- Entry snapshot: try 10:00–10:30, fall back to 10:30–11:00 ---
        entry_snap = window_snapshot(day_df, trade_date, ENTRY_START_H, ENTRY_START_M,
                                     ENTRY_END_H, ENTRY_END_M)
        fallback_used = False
        if entry_snap is None:
            entry_snap = window_snapshot(day_df, trade_date, FALLBACK_START_H, FALLBACK_START_M,
                                         FALLBACK_END_H, FALLBACK_END_M)
            fallback_used = True
        if entry_snap is None:
            continue

        # --- EOD snapshot for settlement ---
        eod_snap = window_snapshot(day_df, trade_date, EOD_START_H, EOD_START_M,
                                   EOD_END_H, EOD_END_M)
        if eod_snap is None:
            continue
        eod_spot = float(eod_snap["underlying_price"].dropna().median())

        # --- Spot from entry chain ---
        spot = float(entry_snap["underlying_price"].dropna().median())
        if pd.isna(spot) or spot <= 0:
            continue

        # --- Call wall from entry chain ---
        call_wall, _ = find_raw_wall_strikes(entry_snap, spot)
        if call_wall is None:
            continue

        # --- ZGL from entry chain ---
        gex_chain = entry_snap[entry_snap["gamma"] > 0].copy()
        zgl = find_zero_gamma_level(gex_chain, spot=spot)
        dist_wall_to_zgl = (call_wall - zgl) if zgl is not None else None

        # --- IV smile residual ---
        residual = compute_iv_smile_residual(entry_snap, spot, call_wall, trade_date)

        # --- Strike placement ---
        K1 = round((call_wall + 10) / 5) * 5
        K2 = K1 + SPREAD_WIDTH

        # --- 0DTE calls at entry ---
        chain_0dte_calls = entry_snap[
            (entry_snap["expiration_date"] == trade_date)
            & (entry_snap["contract_type"].str.upper() == "CALL")
        ].copy()

        mark_K1 = find_mark_price(chain_0dte_calls, K1)
        mark_K2 = find_mark_price(chain_0dte_calls, K2)
        if mark_K1 is None or mark_K2 is None:
            continue

        credit   = mark_K1 - mark_K2
        delta_K1 = find_delta(chain_0dte_calls, K1)
        pnl      = compute_pnl(K1, credit, eod_spot)

        base = {
            "date":             trade_date,
            "fallback_used":    fallback_used,
            "call_wall":        call_wall,
            "spot":             spot,
            "zgl":              zgl,
            "dist_wall_to_zgl": dist_wall_to_zgl,
            "iv_smile_residual": residual,
            "K1":               K1,
            "K2":               K2,
            "credit":           credit,
            "delta_K1":         delta_K1,
            "eod_spot":         eod_spot,
            "pnl_dollars":      pnl,
        }

        for label, credit_filter in FILTERS:
            skip = "" if credit_filter(credit) else "insufficient_credit"
            rows_by_filter[label].append({**base, "skip_reason": skip})

    con.close()

    # --- Save CSV for cr≥$1.00 reference ---
    ref_df = pd.DataFrame(rows_by_filter["cr≥$1.00"])
    ref_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(ref_df)} rows → {OUT_CSV}\n")

    # --- Summary table ---
    labels = [label for label, _ in FILTERS]
    print(f"{'Metric':<28}", end="")
    for label in labels:
        print(f"  {label:>10}", end="")
    print()
    print("-" * (28 + len(labels) * 12))

    summary: dict[str, dict] = {}
    for label in labels:
        df   = pd.DataFrame(rows_by_filter[label])
        trad = df[df["skip_reason"] == ""]
        pnl  = trad["pnl_dollars"].dropna()
        wins = (pnl > 0).sum()
        summary[label] = {
            "total_days": len(df),
            "traded":     len(trad),
            "wins":       int(wins),
            "losses":     int((pnl <= 0).sum()),
            "win_rate":   wins / len(pnl) * 100 if len(pnl) else float("nan"),
            "avg_pnl":    pnl.mean() if len(pnl) else float("nan"),
            "total":      pnl.sum()  if len(pnl) else float("nan"),
            "max_win":    pnl.max()  if len(pnl) else float("nan"),
            "max_loss":   pnl.min()  if len(pnl) else float("nan"),
        }

    def row_print(row_label, key, fmt):
        print(f"{row_label:<28}", end="")
        for label in labels:
            print(f"  {fmt.format(summary[label][key]):>10}", end="")
        print()

    row_print("Days with data",  "total_days", "{:d}")
    row_print("Days traded",     "traded",     "{:d}")
    row_print("Wins",            "wins",       "{:d}")
    row_print("Losses",          "losses",     "{:d}")
    row_print("Win rate",        "win_rate",   "{:.1f}%")
    row_print("Avg P&L",         "avg_pnl",    "${:.2f}")
    row_print("Total P&L",       "total",      "${:.0f}")
    row_print("Max win",         "max_win",    "${:.2f}")
    row_print("Max loss",        "max_loss",   "${:.2f}")
    print()


if __name__ == "__main__":
    main()
