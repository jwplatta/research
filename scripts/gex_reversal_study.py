"""GEX Reverse-Engineering Study

For each trading day (Dec 2025–Jun 2026):
  1. Detect actual SPX intraday inflection points (swing highs/lows)
  2. Compute GEX levels from the open snapshot (~9:40 ET)
  3. For each inflection, measure distance to nearest GEX level
  4. Compare against baselines: low-GEX strikes, round numbers, random

Outputs:
  out/gex_reversal/inflections.csv  — per-inflection distances
  out/gex_reversal/summary.csv      — % within thresholds per method
"""

import sys
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from calc.gex import (
    find_aggregate_wall_strikes,
    find_decision_zones,
    find_raw_wall_strikes,
    net_gex_by_strike,
)
from util.minio import get_duckdb_conn, option_parquet_uris
from util.reversal import find_inflection_points
from util.spx_candles import load_spx_1min, load_vix_5min

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

START_DATE    = date(2025, 12, 1)
END_DATE      = date(2026, 6, 30)
PROVIDER      = "schwab"
TZ            = "America/New_York"

OPEN_SNAP_ET  = time(9, 40)    # earliest snapshot to use as "open" GEX
STRIKE_RANGE  = 300.0
TOP_N_STRIKES = 10             # top N high-GEX and low-GEX strikes per side

# Reversal detection parameters to sweep
LOOKBACK_BARS   = [5, 10, 15]
MIN_MOVE_SWEEP  = [5.0, 8.0, 12.0]

# Default params for the main run
DEFAULT_LOOKBACK  = 10
DEFAULT_MIN_MOVE  = 8.0

THRESHOLDS = [5.0, 10.0, 20.0]  # distance thresholds for % within analysis

OUT_DIR = Path("out/gex_reversal")

COLUMNS = [
    "sampled_at", "expiration_date", "strike", "contract_type",
    "gamma", "open_interest", "underlying_price",
]

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
    df = con.execute(f"""
        SELECT {cols} FROM read_parquet('{uri}', union_by_name=true)
    """).fetchdf()
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True).dt.tz_convert(TZ)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date
    for col in ["gamma", "open_interest", "underlying_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_open_snapshot(day_opts: pd.DataFrame) -> pd.DataFrame | None:
    """Return first snapshot at or after OPEN_SNAP_ET."""
    cutoff = day_opts["sampled_at"].iloc[0].replace(
        hour=OPEN_SNAP_ET.hour, minute=OPEN_SNAP_ET.minute, second=0, microsecond=0
    )
    valid = day_opts[day_opts["sampled_at"] >= cutoff]
    if valid.empty:
        return None
    snap_ts = valid["sampled_at"].min()
    return day_opts[day_opts["sampled_at"] == snap_ts]


# ---------------------------------------------------------------------------
# GEX level extraction
# ---------------------------------------------------------------------------

def extract_gex_levels(
    snap: pd.DataFrame,
    spot: float,
    trade_date: date,
) -> dict:
    """Extract all GEX-based level sets from the open snapshot."""
    anchor = pd.Timestamp(trade_date)

    # Full net GEX by strike
    gex_df = net_gex_by_strike(snap, spot=spot, strike_range=STRIKE_RANGE)
    gex_df = gex_df.dropna(subset=["net_gex"])

    call_gex = gex_df[gex_df["net_gex"] > 0].sort_values("net_gex", ascending=False)
    put_gex  = gex_df[gex_df["net_gex"] < 0].sort_values("net_gex", ascending=True)

    high_gex_calls = call_gex.head(TOP_N_STRIKES)["strike"].to_numpy()
    high_gex_puts  = put_gex.head(TOP_N_STRIKES)["strike"].to_numpy()
    high_gex_all   = gex_df.reindex(gex_df["net_gex"].abs().nlargest(TOP_N_STRIKES * 2).index)["strike"].to_numpy()

    # Low-GEX baseline: bottom N by |net_gex|
    low_gex_all = gex_df.reindex(gex_df["net_gex"].abs().nsmallest(TOP_N_STRIKES * 2).index)["strike"].to_numpy()

    # Wall methods
    raw_call, raw_put     = find_raw_wall_strikes(snap, spot=spot, strike_range=STRIKE_RANGE)
    wtd_call, wtd_put     = find_aggregate_wall_strikes(
        snap, spot=spot, strike_range=STRIKE_RANGE,
        method="distance_weighted_aggregate", anchor_date=anchor,
    )
    clust_call, clust_put = find_aggregate_wall_strikes(
        snap, spot=spot, strike_range=STRIKE_RANGE,
        method="per_expiry_clustering", anchor_date=anchor,
    )
    res_zones, sup_zones  = find_decision_zones(
        snap, spot=spot, strike_range=STRIKE_RANGE, anchor_date=anchor, top_n=1,
    )
    zone_call = res_zones[0]["center"] if res_zones else None
    zone_put  = sup_zones[0]["center"] if sup_zones else None

    return {
        "high_gex_calls": high_gex_calls,
        "high_gex_puts":  high_gex_puts,
        "high_gex_all":   high_gex_all,
        "low_gex_all":    low_gex_all,
        "raw_call": raw_call, "raw_put": raw_put,
        "wtd_call": wtd_call, "wtd_put": wtd_put,
        "clust_call": clust_call, "clust_put": clust_put,
        "zone_call": zone_call,  "zone_put": zone_put,
    }


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def nearest_dist(price: float, levels: np.ndarray) -> float | None:
    """Distance from price to nearest level in array. None if empty."""
    if len(levels) == 0:
        return None
    return float(np.abs(levels - price).min())


def nearest_round_dist(price: float, increment: float = 25.0) -> float:
    """Distance to nearest round-number increment."""
    rounded = round(price / increment) * increment
    return abs(price - rounded)


def random_baseline_dist(
    price: float,
    spot: float,
    n_levels: int,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean distance to nearest random level over n_trials Monte Carlo samples."""
    if rng is None:
        rng = np.random.default_rng(42)
    dists = []
    for _ in range(n_trials):
        rand_levels = rng.uniform(spot - STRIKE_RANGE, spot + STRIKE_RANGE, size=n_levels)
        dists.append(float(np.abs(rand_levels - price).min()))
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(lookback: int = DEFAULT_LOOKBACK, min_move: float = DEFAULT_MIN_MOVE) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SPX 1-min candles {START_DATE} → {END_DATE} ...")
    spx = load_spx_1min(START_DATE, END_DATE)
    vix = load_vix_5min(START_DATE, END_DATE)
    print(f"  {len(spx)} SPX bars, {len(vix)} VIX bars\n")

    con = get_duckdb_conn()
    trading_days = sorted({d for d in spx.index.date if START_DATE <= d < END_DATE})
    print(f"Found {len(trading_days)} trading days\n")

    rng = np.random.default_rng(42)
    all_rows: list[dict] = []

    for trade_date in tqdm(trading_days, desc="Days", unit="day"):
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

        day_prices = spx[spx.index.date == trade_date].sort_index()
        if day_prices.empty:
            continue

        # VIX at open
        vix_day = vix[vix.index.date == trade_date]
        vix_open = float(vix_day["close"].iloc[0]) if not vix_day.empty else None

        # Inflection points
        inflections = find_inflection_points(day_prices, lookback=lookback, min_move_pts=min_move)
        if inflections.empty:
            continue

        # GEX levels
        levels = extract_gex_levels(snap, spot, trade_date)
        n_gex = len(levels["high_gex_all"])

        for _, inf_row in inflections.iterrows():
            price     = float(inf_row["price"])
            inf_type  = str(inf_row["type"])

            row: dict = {
                "date":       trade_date,
                "timestamp":  inf_row["timestamp"],
                "price":      price,
                "type":       inf_type,
                "spot_open":  spot,
                "vix_open":   vix_open,
                "lookback":   lookback,
                "min_move":   min_move,
            }

            # High-GEX distances
            row["dist_high_gex_call"] = nearest_dist(price, levels["high_gex_calls"])
            row["dist_high_gex_put"]  = nearest_dist(price, levels["high_gex_puts"])
            row["dist_high_gex_any"]  = nearest_dist(price, levels["high_gex_all"])

            # Low-GEX baseline
            row["dist_low_gex_any"]   = nearest_dist(price, levels["low_gex_all"])

            # Wall methods
            for name, level in (
                ("raw_call",   levels["raw_call"]),
                ("raw_put",    levels["raw_put"]),
                ("wtd_call",   levels["wtd_call"]),
                ("wtd_put",    levels["wtd_put"]),
                ("clust_call", levels["clust_call"]),
                ("clust_put",  levels["clust_put"]),
                ("zone_call",  levels["zone_call"]),
                ("zone_put",   levels["zone_put"]),
            ):
                row[f"dist_{name}"] = abs(price - level) if level is not None else None
                row[f"level_{name}"] = level

            # Baselines
            row["dist_round_25"]  = nearest_round_dist(price, 25.0)
            row["dist_round_5"]   = nearest_round_dist(price, 5.0)
            row["dist_random"]    = random_baseline_dist(price, spot, n_gex, n_trials=100, rng=rng)

            all_rows.append(row)

    con.close()

    df = pd.DataFrame(all_rows)
    path = OUT_DIR / f"inflections_lb{lookback}_mm{int(min_move)}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} inflection events → {path}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print % within distance thresholds for each method."""
    methods = {
        "high_gex_any":  "dist_high_gex_any",
        "high_gex_call": "dist_high_gex_call",
        "high_gex_put":  "dist_high_gex_put",
        "low_gex_any":   "dist_low_gex_any",
        "raw_call":      "dist_raw_call",
        "raw_put":       "dist_raw_put",
        "wtd_call":      "dist_wtd_call",
        "wtd_put":       "dist_wtd_put",
        "zone_call":     "dist_zone_call",
        "zone_put":      "dist_zone_put",
        "round_25":      "dist_round_25",
        "round_5":       "dist_round_5",
        "random":        "dist_random",
    }

    hdr = f"{'Method':<18} {'N':>6} {'Median':>8}"
    for t in THRESHOLDS:
        hdr += f"  {'<'+str(int(t))+'pt':>7}"
    print("\n=== Distance to Nearest Level ===")
    print(hdr)
    print("-" * (len(hdr) + 4))

    for name, col in methods.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        n = len(series)
        median = series.median()
        row = f"{name:<18} {n:>6} {median:>7.1f}"
        for t in THRESHOLDS:
            pct = (series <= t).mean() * 100
            row += f"  {pct:>6.1f}%"
        print(row)

    # Stratify by inflection type
    print("\n=== High-GEX (any side) by Inflection Type ===")
    for inf_type in ("high", "low"):
        sub = df[df["type"] == inf_type]["dist_high_gex_any"].dropna()
        if sub.empty:
            continue
        row = f"{'gex_any ('+inf_type+')':<18} {len(sub):>6} {sub.median():>7.1f}"
        for t in THRESHOLDS:
            row += f"  {(sub <= t).mean()*100:>6.1f}%"
        print(row)


def main() -> None:
    # Default run
    df = run(lookback=DEFAULT_LOOKBACK, min_move=DEFAULT_MIN_MOVE)
    print_summary(df)

    # Sweep lookback and min_move to show sensitivity
    print("\n=== Sensitivity: median dist_high_gex_any by params ===")
    print(f"{'lookback':>10} {'min_move':>10} {'n_inflections':>14} {'median_dist':>12}")
    for lb in LOOKBACK_BARS:
        for mm in MIN_MOVE_SWEEP:
            if lb == DEFAULT_LOOKBACK and mm == DEFAULT_MIN_MOVE:
                continue  # already ran
            sub = run(lookback=lb, min_move=mm)
            med = sub["dist_high_gex_any"].dropna().median()
            print(f"{lb:>10} {mm:>10.1f} {len(sub):>14} {med:>11.1f}")


if __name__ == "__main__":
    main()
