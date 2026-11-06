"""Backtest equity curve chart — saves to a caller-specified output directory."""

from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_equity_curve(
    dates: list[date],
    equity: list[float],
    out_dir: str | Path,
    baseline: float = 10_000,
    title: str = "Portfolio Value",
) -> None:
    """Plot an equity curve and save as PNG to ``out_dir``.

    Args:
        dates: Ordered list of trade dates.
        equity: Portfolio value at each date (same length as ``dates``).
        out_dir: Directory to save the PNG (created if it does not exist).
        baseline: Starting capital — plotted as a dashed reference line.
        title: Chart title and filename slug.

    Raises:
        ValueError: If ``dates`` and ``equity`` have different lengths or are empty.
    """
    if len(dates) != len(equity):
        raise ValueError(
            f"dates ({len(dates)}) and equity ({len(equity)}) must have the same length"
        )
    if not dates:
        raise ValueError("Cannot plot an empty equity curve")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, equity, linewidth=1.5, color="#2563eb", label="Portfolio")
    ax.axhline(
        baseline,
        color="#9ca3af",
        linestyle="--",
        linewidth=1,
        label=f"Baseline ${baseline:,.0f}",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()

    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    slug = title.lower().replace(" ", "_").replace("/", "-")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_path / f"{ts}_{slug}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Equity curve saved to {path}")
