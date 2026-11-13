"""Compare Backtest 1 (8-delta IC) vs Backtest 2 (GEX wall IC).

Loads trade CSVs produced by ic_delta.py and ic_gex_wall.py and prints:
  - Performance summary: Sharpe, ann_return, ann_vol, max_drawdown, max_drawdown_duration
  - Trade counts and trade-day overlap analysis
  - Side-by-side credit and PnL per day

Usage:
    uv run python backtests/gex_wall_ic/compare.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("./tmp/backtests/gex_wall_ic")
DELTA_CSV     = OUT_DIR / "ic_delta_trades.csv"
DELTA_15_CSV  = OUT_DIR / "ic_delta_15_trades.csv"
DELTA_20_CSV  = OUT_DIR / "ic_delta_20_trades.csv"
GEX_CSV       = OUT_DIR / "ic_gex_wall_trades.csv"
GEX_5PT_CSV  = OUT_DIR / "ic_gex_wall_5pt_trades.csv"
GEX_10PT_CSV = OUT_DIR / "ic_gex_wall_10pt_trades.csv"
GEX_15PT_CSV = OUT_DIR / "ic_gex_wall_15pt_trades.csv"
GEX_20PT_CSV = OUT_DIR / "ic_gex_wall_20pt_trades.csv"
MULTIPLIER = 100
SPREAD_WIDTH = 20.0
TRADING_DAYS_PER_YEAR = 252


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trade log not found: {path}\nRun the backtest first.")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _daily_pnl(df: pd.DataFrame) -> pd.Series:
    """Return a Series of PnL indexed by date (only traded days)."""
    return df.set_index("date")["pnl"]


def _metrics(daily_pnl: pd.Series, df: pd.DataFrame, label: str) -> dict:
    n = len(daily_pnl)
    if n == 0:
        return {"strategy": label}
    ann_return = daily_pnl.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = daily_pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > 0 else float("nan")

    # Drawdown on cumulative equity
    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = float(dd.min())

    # Max drawdown duration
    in_dd = dd < 0
    duration = 0
    max_dur = 0
    for v in in_dd:
        if v:
            duration += 1
            max_dur = max(max_dur, duration)
        else:
            duration = 0

    return {
        "strategy": label,
        "trades": n,
        "total_pnl": round(daily_pnl.sum(), 2),
        "ann_return": round(ann_return, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_duration_days": max_dur,
        "avg_credit": round(df["credit"].mean(), 3),
        "win_rate": round((daily_pnl > 0).mean(), 3),
    }


def _overlap(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    s1 = set(df1["date"])
    s2 = set(df2["date"])
    both = s1 & s2
    only1 = s1 - s2
    only2 = s2 - s1

    print(f"\n{'='*60}")
    print("TRADE DAY OVERLAP")
    print(f"{'='*60}")
    print(f"  Both traded:          {len(both)} days")
    print(f"  Only delta IC traded: {len(only1)} days")
    print(f"  Only GEX wall traded: {len(only2)} days")
    total_days = len(s1 | s2)
    print(f"  Total days with any trade: {total_days}")

    if only1:
        print(f"\n  Days only delta IC traded (first 10): {sorted(only1)[:10]}")
    if only2:
        print(f"  Days only GEX wall traded (first 10): {sorted(only2)[:10]}")


def _side_by_side(df1: pd.DataFrame, df2: pd.DataFrame, label1: str = "s1", label2: str = "s2") -> None:
    """Print per-day comparison for days where both traded."""
    df2_cols = ["date", "sc", "sp", "sc_delta", "sp_delta", "credit", "pnl"]
    extra = [c for c in ["call_wall", "put_wall"] if c in df2.columns]
    df2_cols = ["date"] + extra + ["sc", "sp", "sc_delta", "sp_delta", "credit", "pnl"]

    merged = pd.merge(
        df1[["date", "sc", "sp", "sc_delta", "sp_delta", "credit", "pnl"]].rename(
            columns={"sc": "s1_sc", "sp": "s1_sp", "sc_delta": "s1_sc_delta",
                     "sp_delta": "s1_sp_delta", "credit": "s1_credit", "pnl": "s1_pnl"}
        ),
        df2[df2_cols].rename(
            columns={"sc": "s2_sc", "sp": "s2_sp", "sc_delta": "s2_sc_delta",
                     "sp_delta": "s2_sp_delta", "credit": "s2_credit", "pnl": "s2_pnl"}
        ),
        on="date", how="inner",
    )
    if merged.empty:
        print("\nNo days where both strategies traded.")
        return

    print(f"\n{'='*60}")
    print(f"SIDE-BY-SIDE: days both traded ({len(merged)} days)")
    print(f"{'='*60}")
    display_cols = ["date", "s1_sc", "s1_sp", "s1_credit", "s1_pnl", "s2_sc", "s2_sp", "s2_credit", "s2_pnl"]
    with pd.option_context("display.max_rows", 50, "display.width", 200):
        print(merged[display_cols].to_string(index=False))

    print(f"\n  Avg {label1} credit: ${merged['s1_credit'].mean():.3f}")
    print(f"  Avg {label2} credit: ${merged['s2_credit'].mean():.3f}")
    print(f"  {label1} won more: {(merged['s1_pnl'] > merged['s2_pnl']).sum()} days")
    print(f"  {label2} won more: {(merged['s2_pnl'] > merged['s1_pnl']).sum()} days")


def main() -> None:
    datasets = [
        (DELTA_CSV,    "8-Delta IC"),
        (DELTA_15_CSV, "15-Delta IC"),
        (DELTA_20_CSV, "20-Delta IC"),
        (GEX_CSV,      "GEX Wall (0pt)"),
        (GEX_5PT_CSV,  "GEX Wall +5pt"),
        (GEX_10PT_CSV, "GEX Wall +10pt"),
        (GEX_15PT_CSV, "GEX Wall +15pt"),
        (GEX_20PT_CSV, "GEX Wall +20pt"),
    ]

    loaded = []
    for path, label in datasets:
        try:
            loaded.append((_load(path), label))
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    if not loaded:
        print("No trade logs found — run the backtests first.")
        return

    metrics = [_metrics(_daily_pnl(df), df, label) for df, label in loaded]

    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    summary = pd.DataFrame(metrics).set_index("strategy")
    print(summary.to_string())

    if len(loaded) >= 2:
        df1, label1 = loaded[0]
        for df2, label2 in loaded[1:]:
            print(f"\n--- Overlap: {label1} vs {label2} ---")
            _overlap(df1, df2)

        print(f"\n--- Side-by-side: {loaded[0][1]} vs {loaded[1][1]} ---")
        _side_by_side(loaded[0][0], loaded[1][0], loaded[0][1], loaded[1][1])


if __name__ == "__main__":
    main()
