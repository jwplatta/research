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
DELTA_CSV    = OUT_DIR / "ic_delta_trades.csv"
GEX_CSV      = OUT_DIR / "ic_gex_wall_trades.csv"
GEX_5PT_CSV  = OUT_DIR / "ic_gex_wall_5pt_trades.csv"
GEX_10PT_CSV = OUT_DIR / "ic_gex_wall_10pt_trades.csv"
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
    ann_return = daily_pnl.sum() / TRADING_DAYS_PER_YEAR * TRADING_DAYS_PER_YEAR
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


def _side_by_side(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Print per-day comparison for days where both traded."""
    merged = pd.merge(
        df1[["date", "sc", "sp", "sc_delta", "sp_delta", "credit", "pnl"]].rename(
            columns={"sc": "d_sc", "sp": "d_sp", "sc_delta": "d_sc_delta",
                     "sp_delta": "d_sp_delta", "credit": "d_credit", "pnl": "d_pnl"}
        ),
        df2[["date", "call_wall", "put_wall", "sc", "sp", "sc_delta", "sp_delta", "credit", "pnl"]].rename(
            columns={"sc": "g_sc", "sp": "g_sp", "sc_delta": "g_sc_delta",
                     "sp_delta": "g_sp_delta", "credit": "g_credit", "pnl": "g_pnl"}
        ),
        on="date", how="inner",
    )
    if merged.empty:
        print("\nNo days where both strategies traded.")
        return

    print(f"\n{'='*60}")
    print(f"SIDE-BY-SIDE: days both traded ({len(merged)} days)")
    print(f"{'='*60}")
    with pd.option_context("display.max_rows", 50, "display.width", 200):
        display_cols = ["date", "d_sc", "d_sp", "d_credit", "d_pnl", "g_sc", "g_sp", "g_credit", "g_pnl"]
        print(merged[display_cols].to_string(index=False))

    print(f"\n  Avg delta IC credit: ${merged['d_credit'].mean():.3f}")
    print(f"  Avg GEX wall credit: ${merged['g_credit'].mean():.3f}")
    print(f"  Delta IC won more: {(merged['d_pnl'] > merged['g_pnl']).sum()} days")
    print(f"  GEX wall won more: {(merged['g_pnl'] > merged['d_pnl']).sum()} days")


def main() -> None:
    datasets = [
        (DELTA_CSV,    "8-Delta IC"),
        (GEX_CSV,      "GEX Wall (0pt)"),
        (GEX_5PT_CSV,  "GEX Wall +5pt"),
        (GEX_10PT_CSV, "GEX Wall +10pt"),
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
        _side_by_side(loaded[0][0], loaded[1][0])


if __name__ == "__main__":
    main()
