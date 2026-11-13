"""Combined portfolio backtest: iron condor + ORB running against a shared portfolio.

Both strategies share a single $10k portfolio and compete for buying power.
Each strategy's logic is unchanged — this file just wires them together.

Usage:
    uv run python backtests/combined_ic_orb.py
"""

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine, Strategy
from iron_condor_verify import IronCondorVerify
from backtests.orb_backtest import ORBBacktest

sys.path.insert(0, str(Path(__file__).parent))


class CombinedICORB(Strategy):
    """Runs IronCondorVerify and ORBBacktest against a single shared portfolio."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "combined_ic_orb"
        self.start_portfolio_amt = 10_000.0

        self._ic = IronCondorVerify()
        self._orb = ORBBacktest()

        # Merge data sources — both strategies use the same keys so they share slices.
        # spxw_options: IC needs delta + full day; ORB only needs through 12:30 and no delta.
        # We load the IC's config (no before_time, includes delta) so both are served.
        self.data_sources = {
            "spxw_options": {
                "type": "minio",
                "root": "SPXW",
                "provider": "schwab",
                "columns": [
                    "sampled_at", "expiration_date", "contract_type",
                    "strike", "delta", "mark",
                ],
            },
            "spx_1min": {"type": "local", "loader": "spx_1min"},
        }

    def _step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        self._ic._step(data_slice, timestamp)
        self._orb._step(data_slice, timestamp)


if __name__ == "__main__":
    combined = CombinedICORB()
    engine = BacktestEngine(
        combined,
        start_date=date(2025, 12, 15),
        end_date=date(2026, 8, 30),
        out_dir="./tmp/backtests",
        bar_size="30min",
    )

    # Wire shared portfolio and logger into sub-strategies after engine injects them.
    # Engine sets combined.portfolio and combined.log before calling run(), so we
    # hook into run() by overriding it here inline.
    _original_run = engine.run

    def _patched_run():
        combined._ic.portfolio = combined.portfolio
        combined._ic.log = combined.log
        combined._orb.portfolio = combined.portfolio
        combined._orb.log = combined.log
        _original_run()

    engine.run = _patched_run
    engine.run()
