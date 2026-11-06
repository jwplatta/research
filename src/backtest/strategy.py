"""Abstract base class for backtest strategies."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from backtest.portfolio import Portfolio


class Strategy(ABC):
    """Base class for all backtest strategies.

    Subclasses must implement ``_step()``. Do NOT override ``step()`` — the
    base class uses it as a template method to guarantee ``portfolio.update()``
    is called on every bar, even if ``_step()`` does nothing.

    ``self.portfolio`` and ``self.log`` are both injected by ``BacktestEngine``
    after the strategy object is fully constructed.

    Data sources format::

        self.data_sources = {
            'spxw_options': {'type': 'minio', 'root': 'SPXW', 'provider': 'schwab'},
            'spx_1min':     {'type': 'local', 'loader': 'spx_1min'},
        }

    Supported ``'local'`` loader values: ``'spx_1min'``, ``'spx_5min'``,
    ``'vix_5min'``, ``'vix9d_1min'``.
    """

    def __init__(self) -> None:
        self.name: str = "unnamed_strategy"
        self.start_portfolio_amt: float = 10_000.0
        self.portfolio: Portfolio | None = None  # injected by BacktestEngine
        self.log: logging.Logger = logging.getLogger("backtest")  # injected by BacktestEngine
        self.data_sources: dict[str, dict] = {}

    def step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """Called by the engine on every bar. Calls ``_step()`` then ``portfolio.update()``.

        Do not override this method. Implement ``_step()`` instead.
        """
        assert self.portfolio is not None, "portfolio not injected — engine did not call run()"
        self._step(data_slice, timestamp)
        self.portfolio.record_equity(timestamp)

    @abstractmethod
    def _step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """Implement your strategy logic here.

        Args:
            data_slice: Dict mapping each data_source name to a DataFrame
                containing rows with ``sampled_at < bucket_end`` (no lookahead).
                All DataFrames are sorted ascending by their timestamp column.
            timestamp: Start of the current time bucket (America/New_York).
        """
        ...
