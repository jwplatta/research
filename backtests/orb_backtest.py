"""SPXW 0DTE Opening Range Breakout (ORB) backtest using the backtest harness.

Strategy:
  - OR forms over 9:30-10:30 ET; minimum width 10 pts required
  - Bullish breakout (5 consecutive 1-min closes above OR high):
      sell put credit spread, short put at/below OR low
  - Bearish breakout (5 consecutive 1-min closes below OR low):
      sell call credit spread, short call at/above OR high
  - 25-point wide spreads; entry cutoff 12:00 ET; one trade per day max
  - No stop loss — holds to EOD expiration (settlement via 15:59 SPX candle)
  - FOMC decision days skipped

Usage:
    uv run python backtests/orb_backtest.py
"""

import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine, Strategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHWAB_FEE_PER_LEG = 1.25
SPREAD_WIDTH = 25.0
MULTIPLIER = 100
SLIPPAGE_PER_LEG = 0.15          # deducted from each leg at entry
MIN_CREDIT = 0.90                 # minimum net credit after slippage
MIN_OR_WIDTH_PTS = 10.0
OR_FORMATION_CUTOFF = time(10, 30)
ENTRY_CUTOFF = time(12, 0)
BREAKOUT_CONFIRMATION_BARS = 5   # consecutive 1-min closes required beyond OR level
EOD_BUCKET = time(15, 30)        # last 30-min bucket; settle via 15:59 SPX close

TZ = ZoneInfo("America/New_York")

# FOMC decision days — entire day skipped.
# Verify against https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DATES: set[date] = {
    date(2025, 12, 18),
    date(2026, 1, 29),
    date(2026, 3, 19),
    date(2026, 4, 30),
    date(2026, 6, 18),
    date(2026, 7, 30),
}


# ---------------------------------------------------------------------------
# OR + breakout detection
# ---------------------------------------------------------------------------

def _find_or_and_breakout(
    candles: pd.DataFrame,
    trade_date: date,
) -> tuple[float, float, pd.Timestamp, str] | tuple[None, None, None, None]:
    """Detect the opening range and first confirmed breakout in the given candles.

    OR: high/low of all 1-min bars with timestamp <= 10:30 ET.
    Breakout: BREAKOUT_CONFIRMATION_BARS consecutive 1-min closes beyond the OR level,
    all bars scanned up to ENTRY_CUTOFF. Returns (or_low, or_high, breakout_ts, direction)
    where breakout_ts is the last confirmation bar, or (None, None, None, None).
    """
    idx = candles.index
    if idx.empty:
        return None, None, None, None

    # Build tz-aware cutoff timestamps anchored to this day's index timezone
    sample = idx[0]
    formation_cutoff = sample.replace(
        hour=OR_FORMATION_CUTOFF.hour, minute=OR_FORMATION_CUTOFF.minute, second=0, microsecond=0
    )
    entry_cutoff = sample.replace(
        hour=ENTRY_CUTOFF.hour, minute=ENTRY_CUTOFF.minute, second=0, microsecond=0
    )

    or_window = candles[idx <= formation_cutoff]
    if or_window.empty:
        return None, None, None, None

    or_high = float(or_window["high"].max())
    or_low = float(or_window["low"].min())

    if (or_high - or_low) < MIN_OR_WIDTH_PTS:
        return None, None, None, None

    post_or = candles[(idx > formation_cutoff) & (idx <= entry_cutoff)]
    consecutive = 0
    current_dir: str | None = None

    for ts, row in post_or.iterrows():
        close = float(row["close"])
        if close > or_high:
            bar_dir = "bullish"
        elif close < or_low:
            bar_dir = "bearish"
        else:
            consecutive = 0
            current_dir = None
            continue

        if bar_dir == current_dir:
            consecutive += 1
        else:
            consecutive = 1
            current_dir = bar_dir

        if consecutive >= BREAKOUT_CONFIRMATION_BARS:
            return or_low, or_high, ts, current_dir

    return None, None, None, None


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------

def _find_short_strike(
    snapshot: pd.DataFrame,
    contract_type: str,
    boundary: float,
    direction: str,
) -> pd.Series | None:
    """Find the short strike at/just inside the OR boundary.

    Bullish → short put at/below OR low.
    Bearish → short call at/above OR high.
    """
    legs = snapshot[snapshot["contract_type"].str.upper() == contract_type.upper()].copy()
    if legs.empty:
        return None

    if direction == "bullish":
        candidates = legs[legs["strike"] <= boundary]
        if candidates.empty:
            return None
        return candidates.loc[candidates["strike"].idxmax()]
    else:
        candidates = legs[legs["strike"] >= boundary]
        if candidates.empty:
            return None
        return candidates.loc[candidates["strike"].idxmin()]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class ORBBacktest(Strategy):
    """0DTE SPXW ORB: credit spread in the breakout direction, holds to EOD."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "orb_backtest"
        self.start_portfolio_amt = 10_000.0
        self.data_sources = {
            "spxw_options": {
                "type": "minio", "root": "SPXW", "provider": "schwab",
                "before_time": time(12, 30),  # only need options up to entry window
                "columns": ["sampled_at", "expiration_date", "contract_type", "strike", "mark"],
            },
            "spx_1min": {"type": "local", "loader": "spx_1min"},
        }
        self._open_trade: dict | None = None
        self._no_entry_today: bool = False
        self._last_trade_date: date | None = None

    def _step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        trade_date = timestamp.date()
        t = timestamp.strftime("%H:%M")

        # Reset per-day state at first bucket of a new day
        if self._last_trade_date != trade_date:
            self._no_entry_today = False
            self._last_trade_date = trade_date

            if trade_date in FOMC_DATES:
                self.log.info("%s %s SKIP: FOMC decision day", trade_date, t)
                self._no_entry_today = True

        # Stale trade from prior day
        if self._open_trade is not None and self._open_trade["trade_date"] != trade_date:
            self.log.warning(
                "%s %s stale open trade from %s — clearing",
                trade_date, t, self._open_trade["trade_date"],
            )
            self._open_trade = None

        # Exit check
        if self._open_trade is not None:
            self._check_exit(data_slice, timestamp)
            return

        # Entry window: 10:30–12:00 (try each bucket until breakout found or cutoff reached)
        if self._no_entry_today:
            return
        if timestamp.time() < OR_FORMATION_CUTOFF:
            return
        if timestamp.time() >= ENTRY_CUTOFF:
            if not self._no_entry_today:
                self.log.warning("%s %s SKIP: entry cutoff reached, no breakout confirmed", trade_date, t)
                self._no_entry_today = True
            return

        self._try_enter(data_slice, timestamp)

    def _try_enter(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        candles = data_slice["spx_1min"]
        opts = data_slice["spxw_options"]
        trade_date = timestamp.date()
        t = timestamp.strftime("%H:%M")

        or_low, or_high, breakout_ts, direction = _find_or_and_breakout(candles, trade_date)

        if breakout_ts is None:
            return  # no confirmed breakout yet — will retry next bucket

        # Found a breakout — commit to no further entry attempts today
        self._no_entry_today = True

        # Options snapshot at or before breakout_ts (no lookahead)
        opts_today = opts[pd.to_datetime(opts["expiration_date"]).dt.date == trade_date]
        valid = opts_today[opts_today["sampled_at"] <= breakout_ts]
        if valid.empty:
            self.log.warning(
                "%s %s SKIP: no options snapshot at or before breakout %s",
                trade_date, t, breakout_ts.strftime("%H:%M"),
            )
            return

        snapshot = valid[valid["sampled_at"] == valid["sampled_at"].max()]

        if direction == "bullish":
            contract_type = "PUT"
            boundary = or_low
            long_offset = -SPREAD_WIDTH
        else:
            contract_type = "CALL"
            boundary = or_high
            long_offset = +SPREAD_WIDTH

        short_leg = _find_short_strike(snapshot, contract_type, boundary, direction)
        if short_leg is None:
            self.log.warning(
                "%s %s SKIP: no %s strike near OR boundary %.1f",
                trade_date, t, contract_type, boundary,
            )
            return

        short_strike = float(short_leg["strike"])
        long_strike = short_strike + long_offset

        long_rows = snapshot[
            (snapshot["contract_type"].str.upper() == contract_type.upper())
            & (snapshot["strike"] == long_strike)
        ]
        if long_rows.empty:
            self.log.warning(
                "%s %s SKIP: long leg not found at %.0f", trade_date, t, long_strike
            )
            return

        short_price = float(short_leg["mark"])
        long_price = float(long_rows.iloc[0]["mark"])
        credit = short_price - long_price - 2 * SLIPPAGE_PER_LEG

        if credit < MIN_CREDIT:
            self.log.warning(
                "%s %s SKIP: net credit $%.2f below minimum $%.2f",
                trade_date, t, credit, MIN_CREDIT,
            )
            return

        max_risk = SPREAD_WIDTH * MULTIPLIER
        if self.portfolio.free_buying_power < max_risk:
            self.log.warning(
                "%s %s SKIP: insufficient buying power — need $%.0f, have $%.0f",
                trade_date, t, max_risk, self.portfolio.free_buying_power,
            )
            return

        self.log.info(
            "%s %s ENTER: %s %s %s/%s  OR=[%.1f,%.1f]  credit=$%.2f  breakout=%s",
            trade_date, t, direction.upper()[:4], contract_type,
            short_strike, long_strike, or_low, or_high, credit,
            breakout_ts.strftime("%H:%M"),
        )

        entry_fees = SCHWAB_FEE_PER_LEG * 2  # 2 legs
        key = f"orb_{trade_date}"
        self.portfolio.enter_position(
            key,
            premium=credit * MULTIPLIER,
            fees=entry_fees,
            max_risk=max_risk,
            timestamp=timestamp,
        )
        self._open_trade = {
            "key": key,
            "trade_date": trade_date,
            "contract_type": contract_type,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "credit": credit,
        }

    def _check_exit(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """EOD settlement only — no stop loss."""
        if timestamp.time() != EOD_BUCKET:
            return

        trade = self._open_trade
        candles = data_slice["spx_1min"]
        trade_date = trade["trade_date"]
        t = timestamp.strftime("%H:%M")

        settle = float(candles["close"].iloc[-1])
        sc = trade["short_strike"]
        ct = trade["contract_type"]

        if ct == "PUT":
            intrinsic = max(0.0, sc - settle)
        else:
            intrinsic = max(0.0, settle - sc)

        loss = min(intrinsic, SPREAD_WIDTH)
        structure_pnl = (trade["credit"] - loss) * MULTIPLIER

        self.log.info(
            "%s %s EXIT(EOD): settle=%.2f  intrinsic=%.2f  pnl=$%.2f",
            trade_date, t, settle, intrinsic, structure_pnl,
        )
        self.portfolio.exit_position(
            trade["key"],
            structure_pnl=structure_pnl,
            fees=0.0,
            timestamp=timestamp,
        )
        self._open_trade = None


if __name__ == "__main__":
    engine = BacktestEngine(
        ORBBacktest(),
        start_date=date(2025, 12, 15),
        end_date=date(2026, 8, 30),
        out_dir="./tmp/backtests",
        bar_size="30min",
    )
    engine.run()
