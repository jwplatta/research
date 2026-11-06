"""Backtest drawdown chart."""

from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def plot_drawdown(
    dates: list[date],
    equity: list[float],
    out_dir: str | Path,
    title: str = "Drawdown",
) -> None:
    """Plot a drawdown percentage chart and save as PNG to ``out_dir``.

    Args:
        dates: Ordered list of trade dates.
        equity: Portfolio value at each date (same length as ``dates``).
        out_dir: Directory to save the PNG (created if it does not exist).
        title: Chart title and filename slug.

    Raises:
        ValueError: If ``dates`` and ``equity`` have different lengths or are empty.
    """
    if len(dates) != len(equity):
        raise ValueError(
            f"dates ({len(dates)}) and equity ({len(equity)}) must have the same length"
        )
    if not dates:
        raise ValueError("Cannot plot an empty drawdown chart")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    equity_arr = np.array(equity, dtype=float)
    running_peak = np.maximum.accumulate(equity_arr)
    # Avoid divide-by-zero if starting equity is 0
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdown_pct = np.where(
            running_peak > 0,
            (equity_arr - running_peak) / running_peak * 100.0,
            0.0,
        )

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.fill_between(dates, drawdown_pct, 0, alpha=0.35, color="#dc2626")
    ax.plot(dates, drawdown_pct, linewidth=1, color="#dc2626")
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")

    # Annotate the maximum drawdown point
    idx_min = int(np.argmin(drawdown_pct))
    if drawdown_pct[idx_min] < 0:
        ax.annotate(
            f"{drawdown_pct[idx_min]:.1f}%",
            xy=(dates[idx_min], drawdown_pct[idx_min]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
            color="#dc2626",
            arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 0.8},
        )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    ax.set_title(f"{title} — Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    slug = title.lower().replace(" ", "_").replace("/", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_path / f"{ts}_{slug}_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Drawdown chart saved to {path}")
