"""BacktestResult — generates charts and summary CSVs from a completed strategy run."""

import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.charts.drawdown import plot_drawdown
from backtest.charts.equity_curve import plot_equity_curve


class BacktestResult:
    """Writes all outputs for a completed backtest run.

    Outputs saved to ``out_dir``:
    - ``{name}_equity.png``       — equity curve chart
    - ``{name}_drawdown.png``     — drawdown chart
    - ``{name}_trades.csv``       — one row per trade
    - ``{name}_summary.csv``      — aggregate metrics
    """

    def __init__(self, strategy, out_dir: str | Path, log: logging.Logger) -> None:
        self.strategy = strategy
        self.out_dir = Path(out_dir)
        self.log = log

    def build(self) -> None:
        """Generate all outputs. Skips chart/CSV generation if equity_history is empty."""
        portfolio = self.strategy.portfolio
        name = self.strategy.name

        if not portfolio.equity_history:
            self.log.warning("equity_history is empty for '%s' — skipping outputs", name)
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # equity_history stores (datetime, equity) per bar — aggregate to daily close
        eh = pd.DataFrame(portfolio.equity_history, columns=["ts", "equity"])
        eh["date"] = pd.to_datetime(eh["ts"]).dt.date
        daily = eh.groupby("date")["equity"].last().reset_index()
        dates = list(daily["date"])
        equities = list(daily["equity"])

        plot_equity_curve(
            dates, equities, out_dir=self.out_dir,
            baseline=self.strategy.start_portfolio_amt, title=name,
        )
        plot_drawdown(dates, equities, out_dir=self.out_dir, title=name)

        unclosed = [t for t in portfolio.trade_history if t["closed_at"] is None]
        if unclosed:
            self.log.warning(
                "%d position(s) still open at end of backtest — exit_pnl not finalized",
                len(unclosed),
            )

        trades_path = self.out_dir / f"{name}_trades.csv"
        pd.DataFrame(portfolio.trade_history).to_csv(trades_path, index=False)
        self.log.info("Trades saved to %s", trades_path)

        self._build_summary_csv(dates, equities)

    def _build_summary_csv(self, dates: list[date], equities: list[float]) -> None:
        portfolio = self.strategy.portfolio
        name = self.strategy.name

        closed_trades = [t for t in portfolio.trade_history if t["closed_at"] is not None]
        pnls = [t["exit_pnl"] for t in closed_trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        equity_arr = np.array(equities, dtype=float)
        running_peak = np.maximum.accumulate(equity_arr)
        drawdown_arr = equity_arr - running_peak

        max_drawdown_amt = float(drawdown_arr.min())
        with np.errstate(invalid="ignore", divide="ignore"):
            max_drawdown_pct = float(
                (drawdown_arr / running_peak).min() * 100.0
            ) if running_peak.max() > 0 else 0.0

        in_dd = drawdown_arr < 0
        max_dd_duration = 0
        current_run = 0
        for flag in in_dd:
            if flag:
                current_run += 1
                max_dd_duration = max(max_dd_duration, current_run)
            else:
                current_run = 0

        trade_count = len(closed_trades)
        win_rate = len(winners) / trade_count if trade_count > 0 else 0.0
        avg_winning_trade = float(np.mean(winners)) if winners else 0.0
        avg_losing_trade = float(np.mean(losers)) if losers else 0.0

        daily_returns = np.diff(equity_arr)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * math.sqrt(252))
        else:
            sharpe = 0.0

        edge = avg_winning_trade * win_rate + avg_losing_trade * (1.0 - win_rate)

        days_with_trade = len({t["opened_at"] for t in closed_trades})
        days_not_traded = len(dates) - days_with_trade

        ending_amt = equities[-1]
        ending_profit = ending_amt - self.strategy.start_portfolio_amt

        summary = {
            "strategy": name,
            "start_portfolio_amt": self.strategy.start_portfolio_amt,
            "ending_portfolio_amt": round(ending_amt, 2),
            "ending_portfolio_profit": round(ending_profit, 2),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 4),
            "avg_winning_trade": round(avg_winning_trade, 2),
            "avg_losing_trade": round(avg_losing_trade, 2),
            "edge": round(edge, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_drawdown_pct, 4),
            "max_drawdown_amt": round(max_drawdown_amt, 2),
            "max_drawdown_duration_days": max_dd_duration,
            "days_not_traded": days_not_traded,
        }

        path = self.out_dir / f"{name}_summary.csv"
        pd.DataFrame([summary]).to_csv(path, index=False)
        self.log.info("Summary saved to %s", path)
        self.log.info("=== %s ===", name)
        for k, v in summary.items():
            if k != "strategy":
                self.log.info("  %s: %s", k, v)
