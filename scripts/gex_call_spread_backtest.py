"""GEX Call Spread Backtest

At bucket 0 (09:30–10:00 ET) each day, check a set of conditions and sell
a 20-pt wide 0DTE call spread just above the GEX call wall.  Exit at EOD
using the last bucket's spot as a settlement proxy.

Inputs
------
- out/gex_walls/wall_stability_30min.csv   — call wall, spot per bucket
- out/gex_walls/master_features.csv        — vix_open per event date
- Schwab SPXW parquet option chain via MinIO (get_duckdb_conn / option_parquet_uris)

Output
------
- out/gex_walls/call_spread_backtest.csv   — full trade log
- Stdout summary
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import find_zero_gamma_level
from util.minio import get_duckdb_conn, option_parquet_uris

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_CSV        = Path("out/gex_walls/wall_stability_30min.csv")
FEATURES_CSV    = Path("out/gex_walls/master_features.csv")
OUT_CSV         = Path("out/gex_walls/call_spread_backtest.csv")

TZ              = ZoneInfo("America/New_York")
PROVIDER        = "schwab"
SPREAD_WIDTH    = 20.0
MIN_CREDIT      = 1.0
CONTRACT_MULT   = 100

# Entry bucket: 1 = 10:00–10:30 ET (30 min after open)
ENTRY_BUCKET    = 1
ENTRY_BUCKET_START_H = 10
ENTRY_BUCKET_START_M = 0
ENTRY_BUCKET_END_H   = 10
ENTRY_BUCKET_END_M   = 30

# T for entry at ~10:45, ~5.25 hrs remaining in a 6.5-hr RTH session
# IV smile residual filter: skip if residual > this (market pricing a break)
MAX_IV_RESIDUAL = 8.0

# Columns to pull from parquet
CHAIN_COLS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price", "volatility", "delta", "mark",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_stability() -> pd.DataFrame:
    df = pd.read_csv(WALL_CSV, parse_dates=["sample_datetime"])
    df["date"] = pd.to_datetime(df["sample_datetime"], utc=True).dt.tz_convert("America/New_York").dt.date
    return df


def load_vix_by_date() -> dict[date, float]:
    """Return {trade_date: vix_open} from master_features.csv.

    The CSV may have multiple rows per date (different events/buckets).
    We use the first vix_open value per date.
    """
    feat = pd.read_csv(FEATURES_CSV)
    # Normalise date column — may be 'date', 'event_date', or similar
    date_col = next((c for c in feat.columns if "date" in c.lower()), feat.columns[0])
    feat[date_col] = pd.to_datetime(feat[date_col]).dt.date
    vix_col = next((c for c in feat.columns if "vix_open" in c.lower()), None)
    if vix_col is None:
        print("WARNING: vix_open column not found in master_features.csv; VIX set to NaN")
        return {}
    vix_map = (
        feat.dropna(subset=[vix_col])
        .groupby(date_col)[vix_col]
        .first()
        .to_dict()
    )
    return {d: float(v) for d, v in vix_map.items()}


def load_bucket0_chain(con, trade_date: date) -> pd.DataFrame:
    """Load the latest SPXW snapshot per expiry within the entry bucket window ET on trade_date.

    Returns DataFrame with CHAIN_COLS, filtered to open_interest > 0.
    """
    uris = option_parquet_uris("SPXW", trade_date, trade_date + timedelta(days=1), provider=PROVIDER)
    uri = uris[0]
    try:
        con.execute(f"SELECT 1 FROM read_parquet('{uri}') LIMIT 1")
    except Exception:
        return pd.DataFrame(columns=CHAIN_COLS)

    cols = ", ".join(CHAIN_COLS)
    # Window: 09:30–10:00 ET expressed as UTC strings for DuckDB filter
    # DuckDB will return timestamps; we filter in Python for safety.
    df = con.execute(f"""
        SELECT {cols}
        FROM read_parquet('{uri}', union_by_name=true)
    """).fetchdf()

    if df.empty:
        return df

    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date

    # Entry bucket window ET
    bstart = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=ENTRY_BUCKET_START_H, minute=ENTRY_BUCKET_START_M, tzinfo=TZ)
    bend   = pd.Timestamp(year=trade_date.year, month=trade_date.month, day=trade_date.day,
                          hour=ENTRY_BUCKET_END_H,   minute=ENTRY_BUCKET_END_M,   tzinfo=TZ)
    bucket0 = df[(df["sampled_at"] >= bstart) & (df["sampled_at"] < bend)].copy()

    if bucket0.empty:
        return pd.DataFrame(columns=CHAIN_COLS)

    # Numeric coercions
    for col in ["gamma", "open_interest", "underlying_price", "volatility", "strike"]:
        bucket0[col] = pd.to_numeric(bucket0[col], errors="coerce")

    bucket0 = bucket0[bucket0["open_interest"] > 0].copy()

    # Latest snapshot per (expiry, strike, contract_type) in bucket 0
    bucket0 = (
        bucket0.sort_values("sampled_at")
        .groupby(["expiration_date", "strike", "contract_type"], as_index=False)
        .last()
    )
    return bucket0


# ---------------------------------------------------------------------------
# Signal filters
# ---------------------------------------------------------------------------

def compute_iv_smile_residual(chain: pd.DataFrame, spot: float, call_wall: float) -> float | None:
    """Fit a quadratic in log-moneyness to 0DTE OTM call IVs.

    Returns the residual of the call_wall IV vs the smile fit.
    Returns None if insufficient data.
    """
    today = chain["expiration_date"].min()  # 0DTE = earliest / same-day expiry
    # Actually use sampled_at date for 0DTE identification
    same_day = chain[
        (chain["expiration_date"] == today)
        & (chain["contract_type"].str.upper() == "CALL")
        & (chain["strike"] > spot)
        & (chain["volatility"] > 0)
    ].dropna(subset=["volatility", "strike"])

    if len(same_day) < 5:
        return None

    log_m = np.log(same_day["strike"].values / spot)
    iv    = same_day["volatility"].values

    # Fit quadratic: iv ~ a*log_m² + b*log_m + c
    try:
        coeffs = np.polyfit(log_m, iv, 2)
    except Exception:
        return None

    wall_log_m = float(np.log(call_wall / spot))
    fitted_iv  = float(np.polyval(coeffs, wall_log_m))

    # Wall IV: nearest 0DTE call strike within 2.5 pts
    near = same_day[(same_day["strike"] - call_wall).abs() <= 2.5]
    if near.empty:
        return None
    wall_iv = float(near["volatility"].median())

    return wall_iv - fitted_iv


def find_mark(chain_0dte_calls: pd.DataFrame, strike: float) -> float | None:
    """Return median mark price of 0DTE calls nearest to strike (within 2.5 pts)."""
    near = chain_0dte_calls[(chain_0dte_calls["strike"] - strike).abs() <= 2.5]
    if near.empty:
        return None
    vals = near["mark"].dropna()
    return float(vals.median()) if not vals.empty else None


def find_delta(chain_0dte_calls: pd.DataFrame, strike: float) -> float | None:
    """Return median delta of 0DTE calls nearest to strike (within 2.5 pts)."""
    near = chain_0dte_calls[(chain_0dte_calls["strike"] - strike).abs() <= 2.5]
    if near.empty:
        return None
    vals = near["delta"].dropna()
    return float(vals.median()) if not vals.empty else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- Load static data ---
    stability = load_stability()
    vix_by_date = load_vix_by_date()

    trading_days = sorted(stability["date"].unique())
    print(f"Loaded {len(trading_days)} trading days from stability CSV")
    print(f"VIX available for {len(vix_by_date)} dates\n")

    con = get_duckdb_conn()

    # --- Per-day EOD spot (last bucket's spot) ---
    eod_spot_map: dict[date, float] = {}
    for d, grp in stability.groupby("date"):
        last_bucket = grp.loc[grp["bucket_num"].idxmax()]
        eod_spot_map[d] = float(last_bucket["spot"])

    FILTERS = [
        ("cr≥$0.50",   lambda c: c >= 0.50),
        ("cr≥$0.75",   lambda c: c >= 0.75),
        ("cr≥$1.00",   lambda c: c >= 1.00),
        ("cr/wid≥8%",  lambda c: c / SPREAD_WIDTH >= 0.08),
    ]

    def compute_pnl(K1: float, credit: float, settlement: float | None) -> float | None:
        K2 = K1 + SPREAD_WIDTH
        if settlement is None:
            return None
        if settlement <= K1:
            return credit * CONTRACT_MULT
        if settlement >= K2:
            return -(SPREAD_WIDTH - credit) * CONTRACT_MULT
        return (credit - (settlement - K1)) * CONTRACT_MULT

    # rows_by_thresh[min_credit] = list of row dicts
    rows_by_thresh: dict[str, list[dict]] = {label: [] for label, _ in FILTERS}

    for trade_date in tqdm(trading_days, desc="Days", unit="day"):
        day_stab = stability[stability["date"] == trade_date]
        b0       = day_stab[day_stab["bucket_num"] == ENTRY_BUCKET]
        eod_spot = eod_spot_map.get(trade_date)

        base: dict = {"date": trade_date, "eod_spot": eod_spot}

        if b0.empty:
            for label, _ in FILTERS:
                rows_by_thresh[label].append({**base, "skip_reason": "no_bucket0_stability",
                    "call_wall": None, "spot": None, "zgl": None, "dist_wall_to_zgl": None,
                    "vix": None, "iv_smile_residual": None, "K1": None, "K2": None,
                    "credit": None, "delta_K1": None, "pnl_dollars": None})
            continue

        call_wall = float(b0.iloc[0]["call_wall"])
        spot      = float(b0.iloc[0]["spot"])
        vix       = vix_by_date.get(trade_date)
        base.update({"call_wall": call_wall, "spot": spot, "vix": vix})

        chain = load_bucket0_chain(con, trade_date)
        if chain.empty:
            for label, _ in FILTERS:
                rows_by_thresh[label].append({**base, "skip_reason": "no_chain_data",
                    "zgl": None, "dist_wall_to_zgl": None, "iv_smile_residual": None,
                    "K1": None, "K2": None, "credit": None, "delta_K1": None, "pnl_dollars": None})
            continue

        zgl_chain        = chain[chain["gamma"] > 0].copy()
        zgl              = find_zero_gamma_level(zgl_chain, spot=spot)
        dist_wall_to_zgl = (call_wall - zgl) if zgl is not None else None
        chain_0dte       = chain[chain["expiration_date"] == trade_date]
        residual         = compute_iv_smile_residual(chain_0dte if not chain_0dte.empty else chain, spot, call_wall)
        base.update({"zgl": zgl, "dist_wall_to_zgl": dist_wall_to_zgl, "iv_smile_residual": residual})

        # structural skip reason (same across all thresholds)
        struct_skip = ""
        if dist_wall_to_zgl is not None and dist_wall_to_zgl < 0:
            struct_skip = "wall_below_zgl"
        elif residual is not None and residual > MAX_IV_RESIDUAL:
            struct_skip = "high_iv_residual"

        K1 = round((call_wall + 10) / 5) * 5
        K2 = K1 + SPREAD_WIDTH
        base.update({"K1": K1, "K2": K2})

        chain_0dte_calls = chain_0dte[chain_0dte["contract_type"].str.upper() == "CALL"].copy()
        mark_K1 = find_mark(chain_0dte_calls, K1)
        mark_K2 = find_mark(chain_0dte_calls, K2)

        if mark_K1 is None or mark_K2 is None:
            for label, _ in FILTERS:
                rows_by_thresh[label].append({**base, "skip_reason": struct_skip or "no_mark_at_strikes",
                    "credit": None, "delta_K1": None, "pnl_dollars": None})
            continue

        credit   = mark_K1 - mark_K2
        delta_K1 = find_delta(chain_0dte_calls, K1)
        base["credit"]   = credit
        base["delta_K1"] = delta_K1
        pnl = compute_pnl(K1, credit, eod_spot)

        for label, credit_filter in FILTERS:
            skip = struct_skip or ("insufficient_credit" if not credit_filter(credit) else "")
            rows_by_thresh[label].append({**base,
                "skip_reason": skip,
                "pnl_dollars": pnl if not skip else None})

    con.close()

    # --- Save CSV for cr≥$1.00 (reference) ---
    ref_df = pd.DataFrame(rows_by_thresh["cr≥$1.00"])
    ref_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(ref_df)} rows (cr≥$1.00) → {OUT_CSV}\n")

    # ---------------------------------------------------------------------------
    # Comparison table across filters
    # ---------------------------------------------------------------------------
    labels = [label for label, _ in FILTERS]
    print(f"{'Metric':<28}", end="")
    for label in labels:
        print(f"  {label:>10}", end="")
    print()
    print("-" * (28 + len(labels) * 12))

    summary: dict[str, dict] = {}
    for label in labels:
        df   = pd.DataFrame(rows_by_thresh[label])
        trad = df[df["skip_reason"].isna() | (df["skip_reason"] == "")]
        pnl  = trad["pnl_dollars"].dropna()
        wins = (pnl > 0).sum()
        summary[label] = {
            "traded":   len(trad),
            "wins":     int(wins),
            "losses":   int((pnl <= 0).sum()),
            "win_rate": wins / len(pnl) * 100 if len(pnl) else float("nan"),
            "avg_pnl":  pnl.mean() if len(pnl) else float("nan"),
            "total":    pnl.sum()  if len(pnl) else float("nan"),
            "max_win":  pnl.max()  if len(pnl) else float("nan"),
            "max_loss": pnl.min()  if len(pnl) else float("nan"),
        }

    def row_print(row_label, key, fmt):
        print(f"{row_label:<28}", end="")
        for label in labels:
            print(f"  {fmt.format(summary[label][key]):>10}", end="")
        print()

    row_print("Days traded",    "traded",   "{:d}")
    row_print("Wins",           "wins",     "{:d}")
    row_print("Losses",         "losses",   "{:d}")
    row_print("Win rate",       "win_rate", "{:.1f}%")
    row_print("Avg P&L",        "avg_pnl",  "${:.2f}")
    row_print("Total P&L",      "total",    "${:.0f}")
    row_print("Max win",        "max_win",  "${:.2f}")
    row_print("Max loss",       "max_loss", "${:.2f}")
    print()


if __name__ == "__main__":
    main()
