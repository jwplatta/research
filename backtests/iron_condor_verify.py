"""Verification backtest: simple 0DTE SPXW iron condor using the backtest harness.

Finds ~7-delta short call and put at 10:00–10:30 ET entry window.
20-point wide spreads.

Exit conditions (checked every 30-min bucket after entry):
  - Stop loss: cost to close >= 4x original credit received
  - EOD:       SPX candle settlement at expiry if stop loss not triggered

Outputs to ./tmp/backtests/.

Usage:
    uv run python backtests/iron_condor_verify.py
"""

import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import BacktestEngine, Strategy

SCHWAB_FEE_PER_LEG = 1.25          # dollars per leg
SPREAD_WIDTH = 20.0                 # points
MULTIPLIER = 100                    # options contract multiplier
MIN_CREDIT = 1.00                   # minimum combined net credit to enter (dollars)
DELTA_TARGET = 0.07
DELTA_RANGE = (0.05, 0.10)
ENTRY_BUCKET = time(10, 0)          # enter on the 10:00–10:30 bucket
EOD_BUCKET = time(15, 30)           # last 30-min bucket; settle using 15:59 SPX close
STOP_LOSS_MULT = 4.0                # exit when cost to close exceeds this × entry credit


def _find_short_strike(
    opts: pd.DataFrame,
    contract_type: str,
    delta_target: float,
    delta_range: tuple[float, float],
) -> pd.Series | None:
    """Find the strike closest to delta_target within delta_range.

    For puts, delta is negative — this function works with absolute values.
    Returns the row as a Series, or None if no match found.
    """
    side = opts[opts["contract_type"].str.upper() == contract_type.upper()].copy()
    if side.empty:
        return None

    side["abs_delta"] = side["delta"].abs()
    in_range = side[(side["abs_delta"] >= delta_range[0]) & (side["abs_delta"] <= delta_range[1])]
    if in_range.empty:
        return None

    idx = (in_range["abs_delta"] - delta_target).abs().idxmin()
    return in_range.loc[idx]


def _current_cost_to_close(
    snap: pd.DataFrame, sc: float, sp: float
) -> float | None:
    """Return the current net debit to close the iron condor, or None if any leg missing."""
    lc, lp = sc + SPREAD_WIDTH, sp - SPREAD_WIDTH

    sc_row = snap[(snap["contract_type"].str.upper() == "CALL") & (snap["strike"] == sc)]
    lc_row = snap[(snap["contract_type"].str.upper() == "CALL") & (snap["strike"] == lc)]
    sp_row = snap[(snap["contract_type"].str.upper() == "PUT") & (snap["strike"] == sp)]
    lp_row = snap[(snap["contract_type"].str.upper() == "PUT") & (snap["strike"] == lp)]

    if any(r.empty for r in [sc_row, lc_row, sp_row, lp_row]):
        return None

    return (
        float(sc_row.iloc[0]["mark"]) + float(sp_row.iloc[0]["mark"])
        - float(lc_row.iloc[0]["mark"]) - float(lp_row.iloc[0]["mark"])
    )


class IronCondorVerify(Strategy):
    """0DTE iron condor with intraday TP/SL exit conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "iron_condor_verify"
        self.start_portfolio_amt = 10_000.0
        self.data_sources = {
            "spxw_options": {
                "type": "minio", "root": "SPXW", "provider": "schwab",
                # no before_time — need options data all day for exit checks
                "columns": [
                    "sampled_at", "expiration_date", "contract_type",
                    "strike", "delta", "mark",
                ],
            },
            "spx_1min": {"type": "local", "loader": "spx_1min"},
        }
        self._open_trade: dict | None = None

    def _step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        trade_date = timestamp.date()
        t = timestamp.strftime("%H:%M")

        # Stale trade from prior day — shouldn't happen for 0DTE but guard against it
        if self._open_trade is not None and self._open_trade["trade_date"] != trade_date:
            self.log.warning(
                "%s %s stale open trade from %s — clearing",
                trade_date, t, self._open_trade["trade_date"],
            )
            self._open_trade = None

        if self._open_trade is not None:
            exited = self._check_exit(data_slice, timestamp)
            if exited:
                return

        if self._open_trade is None and timestamp.time() == ENTRY_BUCKET:
            self._try_enter(data_slice, timestamp)

    def _check_exit(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> bool:
        """Check TP/SL conditions, then EOD settlement. Returns True if position was closed."""
        trade = self._open_trade
        opts = data_slice["spxw_options"]
        trade_date = trade["trade_date"]
        t = timestamp.strftime("%H:%M")
        is_eod = timestamp.time() == EOD_BUCKET

        # Options-based TP/SL check
        opts_today = opts[pd.to_datetime(opts["expiration_date"]).dt.date == trade_date]
        if not opts_today.empty:
            snap = opts_today[opts_today["sampled_at"] == opts_today["sampled_at"].max()]
            cost = _current_cost_to_close(snap, trade["sc"], trade["sp"])

            if cost is not None:
                sl = cost >= STOP_LOSS_MULT * trade["credit"]

                if sl:
                    reason = "SL"
                    structure_pnl = (trade["credit"] - cost) * MULTIPLIER
                    exit_fees = SCHWAB_FEE_PER_LEG * 4
                    self.log.info(
                        "%s %s EXIT(%s): cost_to_close=$%.2f  pnl=$%.2f",
                        trade_date, t, reason, cost, structure_pnl - exit_fees,
                    )
                    self.portfolio.exit_position(
                        trade["key"],
                        structure_pnl=structure_pnl,
                        fees=exit_fees,
                        timestamp=timestamp,
                    )
                    self._open_trade = None
                    return True

        # EOD settlement via SPX candle
        if is_eod:
            candles = data_slice["spx_1min"]
            # Last available candle — tickrake SPX data ends at 15:59
            settle = float(candles["close"].iloc[-1])
            sc, sp = trade["sc"], trade["sp"]
            call_loss = min(max(0.0, settle - sc), SPREAD_WIDTH)
            put_loss = min(max(0.0, sp - settle), SPREAD_WIDTH)
            structure_pnl = (trade["credit"] - call_loss - put_loss) * MULTIPLIER
            self.log.info(
                "%s %s EXIT(EOD): settle=%.2f  call_loss=%.2f  put_loss=%.2f  pnl=$%.2f",
                trade_date, t, settle, call_loss, put_loss, structure_pnl,
            )
            self.portfolio.exit_position(
                trade["key"],
                structure_pnl=structure_pnl,
                fees=0.0,
                timestamp=timestamp,
            )
            self._open_trade = None
            return True

        return False

    def _try_enter(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        """Attempt to enter iron condor at ENTRY_BUCKET."""
        opts = data_slice["spxw_options"]
        trade_date = timestamp.date()
        t = timestamp.strftime("%H:%M")
        key = f"condor_{trade_date}"

        opts = opts[pd.to_datetime(opts["expiration_date"]).dt.date == trade_date].copy()
        if opts.empty:
            self.log.warning("%s %s SKIP: no same-day expirations in options chain", trade_date, t)
            return

        snap = opts[opts["sampled_at"] == opts["sampled_at"].max()]

        short_call = _find_short_strike(snap, "CALL", DELTA_TARGET, DELTA_RANGE)
        short_put = _find_short_strike(snap, "PUT", DELTA_TARGET, DELTA_RANGE)

        if short_call is None or short_put is None:
            missing = []
            if short_call is None:
                missing.append("call")
            if short_put is None:
                missing.append("put")
            self.log.warning(
                "%s %s SKIP: no strike in delta range %s for: %s",
                trade_date, t, DELTA_RANGE, ", ".join(missing),
            )
            return

        long_call_strike = short_call["strike"] + SPREAD_WIDTH
        long_put_strike = short_put["strike"] - SPREAD_WIDTH

        long_call_row = snap[
            (snap["contract_type"].str.upper() == "CALL") & (snap["strike"] == long_call_strike)
        ]
        long_put_row = snap[
            (snap["contract_type"].str.upper() == "PUT") & (snap["strike"] == long_put_strike)
        ]

        if long_call_row.empty or long_put_row.empty:
            missing = []
            if long_call_row.empty:
                missing.append(f"long call @ {long_call_strike}")
            if long_put_row.empty:
                missing.append(f"long put @ {long_put_strike}")
            self.log.warning(
                "%s %s SKIP: long leg(s) not in chain: %s", trade_date, t, ", ".join(missing)
            )
            return

        credit = (
            float(short_call["mark"]) + float(short_put["mark"])
            - float(long_call_row.iloc[0]["mark"]) - float(long_put_row.iloc[0]["mark"])
        )
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

        sc = float(short_call["strike"])
        sp = float(short_put["strike"])
        self.log.info(
            "%s %s ENTER: call spread %s/%s  put spread %s/%s  credit=$%.2f",
            trade_date, t, sc, sc + SPREAD_WIDTH, sp, sp - SPREAD_WIDTH, credit,
        )

        entry_fees = SCHWAB_FEE_PER_LEG * 4
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
            "sc": sc,
            "sp": sp,
            "credit": credit,
        }


if __name__ == "__main__":
    engine = BacktestEngine(
        IronCondorVerify(),
        start_date=date(2025, 12, 15),
        end_date=date(2026, 8, 30),
        out_dir="./tmp/backtests",
        bar_size="30min",
    )
    engine.run()
