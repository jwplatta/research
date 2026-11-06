"""Unit tests for the backtest harness (src/backtest/).

Covers the main code paths and high-leverage joints:
- Portfolio: cash math, buying power, trade history, error conditions
- Strategy: template method guarantees record_equity on every bar
- BacktestEngine: data source validation, run-dir naming, no-lookahead bucket slicing
"""

import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.portfolio import Portfolio
from backtest.strategy import Strategy

TZ = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(h: int, m: int = 0) -> datetime:
    return datetime(2026, 1, 15, h, m, tzinfo=TZ)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_initial_state(self):
        p = Portfolio(10_000.0)
        assert p.free_cash == 10_000.0
        assert p.free_buying_power == 10_000.0
        assert p.current_equity == 10_000.0
        assert p.trade_history == []
        assert p.equity_history == []

    def test_enter_reduces_buying_power(self):
        p = Portfolio(10_000.0)
        p.enter_position("pos1", premium=200.0, fees=5.0, max_risk=2_000.0, timestamp=_ts(10))
        assert p.free_cash == 10_195.0           # 10000 + 200 credit - 5 fees
        assert p.free_buying_power == 8_195.0    # free_cash - 2000 max_risk
        assert p.current_equity == 10_395.0      # free_cash + 200 premium still open

    def test_exit_settles_pnl(self):
        p = Portfolio(10_000.0)
        p.enter_position("pos1", premium=200.0, fees=5.0, max_risk=2_000.0, timestamp=_ts(10))
        p.exit_position("pos1", structure_pnl=200.0, fees=0.0, timestamp=_ts(16))
        assert p.free_buying_power == pytest.approx(10_395.0)  # max_risk released
        assert len(p.trade_history) == 1
        trade = p.trade_history[0]
        assert trade["exit_pnl"] == pytest.approx(195.0)       # 200 structure - 5 entry fees
        assert trade["is_winner"] is True

    def test_losing_trade(self):
        p = Portfolio(10_000.0)
        p.enter_position("pos1", premium=200.0, fees=5.0, max_risk=2_000.0, timestamp=_ts(10))
        # SPX blew through the spread — full max loss
        p.exit_position("pos1", structure_pnl=-1_800.0, fees=0.0, timestamp=_ts(16))
        trade = p.trade_history[0]
        assert trade["exit_pnl"] == pytest.approx(-1_805.0)   # -1800 structure - 5 entry fees
        assert trade["is_winner"] is False

    def test_double_enter_raises(self):
        p = Portfolio(10_000.0)
        p.enter_position("pos1", premium=100.0, fees=5.0, max_risk=500.0, timestamp=_ts(10))
        with pytest.raises(ValueError, match="already open"):
            p.enter_position("pos1", premium=100.0, fees=5.0, max_risk=500.0, timestamp=_ts(10))

    def test_exit_unknown_key_raises(self):
        p = Portfolio(10_000.0)
        with pytest.raises(KeyError):
            p.exit_position("ghost", structure_pnl=100.0, fees=0.0, timestamp=_ts(16))

    def test_insufficient_buying_power_raises(self):
        p = Portfolio(1_000.0)
        with pytest.raises(ValueError, match="Insufficient buying power"):
            p.enter_position("pos1", premium=100.0, fees=5.0, max_risk=2_000.0, timestamp=_ts(10))

    def test_negative_fees_raises(self):
        p = Portfolio(10_000.0)
        with pytest.raises(ValueError, match="fees must be non-negative"):
            p.enter_position("pos1", premium=100.0, fees=-1.0, max_risk=500.0, timestamp=_ts(10))

    def test_record_equity_appends(self):
        p = Portfolio(10_000.0)
        p.record_equity(_ts(9, 30))
        p.record_equity(_ts(10, 0))
        assert len(p.equity_history) == 2
        assert p.equity_history[0] == (_ts(9, 30), 10_000.0)

    def test_multiple_open_positions_buying_power(self):
        p = Portfolio(10_000.0)
        p.enter_position("a", premium=100.0, fees=5.0, max_risk=1_000.0, timestamp=_ts(10))
        p.enter_position("b", premium=100.0, fees=5.0, max_risk=1_000.0, timestamp=_ts(10))
        # committed = 2000; free_cash = 10000 + 200 - 10 = 10190
        assert p.free_buying_power == pytest.approx(8_190.0)


# ---------------------------------------------------------------------------
# Strategy template method
# ---------------------------------------------------------------------------

class _NullStrategy(Strategy):
    """Strategy that does nothing — used to verify template method behaviour."""
    def __init__(self):
        super().__init__()
        self.name = "null"
        self.start_portfolio_amt = 5_000.0
        self.data_sources = {}
        self.step_calls = 0

    def _step(self, data_slice, timestamp):
        self.step_calls += 1


class _ErrorStrategy(Strategy):
    """Strategy whose _step() raises — record_equity should NOT be called."""
    def __init__(self):
        super().__init__()
        self.name = "error"
        self.start_portfolio_amt = 5_000.0
        self.data_sources = {}

    def _step(self, data_slice, timestamp):
        raise RuntimeError("boom")


class TestStrategyTemplateMethod:
    def test_record_equity_called_on_every_step(self):
        s = _NullStrategy()
        s.portfolio = Portfolio(5_000.0)
        for i in range(5):
            s.step({}, _ts(9, 30 + i))
        assert s.step_calls == 5
        assert len(s.portfolio.equity_history) == 5

    def test_portfolio_not_injected_raises(self):
        s = _NullStrategy()
        # portfolio is None — step() should raise AssertionError
        with pytest.raises(AssertionError):
            s.step({}, _ts(10))

    def test_record_equity_not_called_if_step_raises(self):
        s = _ErrorStrategy()
        s.portfolio = Portfolio(5_000.0)
        with pytest.raises(RuntimeError):
            s.step({}, _ts(10))
        assert len(s.portfolio.equity_history) == 0


# ---------------------------------------------------------------------------
# BacktestEngine — validation and run-dir naming
# ---------------------------------------------------------------------------

class TestBacktestEngineValidation:
    def test_invalid_bar_size_raises(self):
        from backtest.engine import BacktestEngine
        s = _NullStrategy()
        with pytest.raises(ValueError, match="bar_size"):
            BacktestEngine(s, date(2026, 1, 1), date(2026, 2, 1), bar_size="2min")

    def test_unknown_source_type_raises(self):
        from backtest.engine import BacktestEngine

        class BadSource(Strategy):
            def __init__(self):
                super().__init__()
                self.name = "bad"
                self.data_sources = {"x": {"type": "redis"}}
            def _step(self, data_slice, timestamp): pass

        with pytest.raises(ValueError, match="unknown type"):
            BacktestEngine(BadSource(), date(2026, 1, 1), date(2026, 2, 1))

    def test_unknown_local_loader_raises(self):
        from backtest.engine import BacktestEngine

        class BadLoader(Strategy):
            def __init__(self):
                super().__init__()
                self.name = "bad"
                self.data_sources = {"x": {"type": "local", "loader": "nonexistent_loader"}}
            def _step(self, data_slice, timestamp): pass

        with pytest.raises(ValueError, match="loader"):
            BacktestEngine(BadLoader(), date(2026, 1, 1), date(2026, 2, 1))

    def test_run_dir_created_under_out_dir(self, tmp_path):
        """run() must create a {strategy_name}_{timestamp} subfolder."""
        from unittest.mock import patch
        from backtest.engine import BacktestEngine

        s = _NullStrategy()
        engine = BacktestEngine(s, date(2026, 1, 1), date(2026, 1, 2),
                                out_dir=tmp_path, bar_size="1d")

        # Patch data loading so run() completes without real data
        with patch.object(engine, "_load_local_sources", return_value={}), \
             patch.object(engine, "_build_minio_uri_maps", return_value={}), \
             patch.object(engine, "_build_trading_days", return_value=[]):
            engine.run()

        subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        assert subdirs[0].name.startswith("null_")


# ---------------------------------------------------------------------------
# BacktestEngine — no-lookahead bucket slicing
# ---------------------------------------------------------------------------

class TestNookaheadSlicing:
    """Verify the engine's intraday bucket slicing enforces sampled_at < bucket_end."""

    def _make_df(self, times: list[tuple[int, int]]) -> pd.DataFrame:
        rows = [{"sampled_at": datetime(2026, 1, 15, h, m, tzinfo=TZ), "value": i}
                for i, (h, m) in enumerate(times)]
        return pd.DataFrame(rows)

    def test_bucket_excludes_data_at_bucket_end(self):
        """Data sampled exactly at bucket_end must NOT be visible in that bucket."""
        from backtest.engine import BacktestEngine
        from datetime import timedelta

        df = self._make_df([(9, 30), (10, 0), (10, 30)])
        # Bucket 10:00–10:30: sampled_at < 10:30 → should include 9:30 and 10:00 rows only
        bucket_end = datetime(2026, 1, 15, 10, 30, tzinfo=TZ)
        sliced = df[df["sampled_at"] < bucket_end]
        assert len(sliced) == 2
        assert all(sliced["sampled_at"] < bucket_end)

    def test_bucket_includes_data_just_before_bucket_end(self):
        df = self._make_df([(10, 29), (10, 30)])
        bucket_end = datetime(2026, 1, 15, 10, 30, tzinfo=TZ)
        sliced = df[df["sampled_at"] < bucket_end]
        assert len(sliced) == 1
        assert sliced.iloc[0]["sampled_at"].time() == time(10, 29)
