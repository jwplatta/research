"""Backtest 1: 0DTE SPXW iron condor with 8-delta short legs.

Entry window: 10:30–11:00 ET (every 5-min bar). Enter on the first bar where a valid
IC exists. Skip the day if no valid trade is found in the window.

Short legs: strike closest to 8-delta (abs_delta target = 0.08).
Wings:      20-point wide.
Min credit: $1.00 combined net.
Exit:       Hold to expiration — EOD settlement at 15:30 bucket via SPX close.
No stop loss.

Logs full trade detail to ./tmp/backtests/gex_wall_ic/ic_delta_trades.csv.

Usage:
    uv run python backtests/gex_wall_ic/ic_delta.py
"""

import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backtest import BacktestEngine, Strategy

SCHWAB_FEE_PER_LEG = 1.25
SPREAD_WIDTH = 20.0
MULTIPLIER = 100
MIN_CREDIT = 1.00
DELTA_TARGET = 0.08
ENTRY_START = time(10, 30)
ENTRY_END = time(10, 55)   # last 5-min bar that starts before 11:00
EOD_BUCKET = time(15, 30)
OUT_DIR = Path("./tmp/backtests/gex_wall_ic")


def _find_short_strike(opts: pd.DataFrame, contract_type: str, delta_target: float) -> pd.Series | None:
    """Find the OTM strike closest to delta_target. Works with absolute delta values."""
    side = opts[opts["contract_type"].str.upper() == contract_type.upper()].copy()
    if side.empty:
        return None
    side["abs_delta"] = side["delta"].abs()
    # Must be OTM (abs_delta < 0.5) and have a mark price
    side = side[(side["abs_delta"] < 0.50) & (side["mark"] > 0)]
    if side.empty:
        return None
    idx = (side["abs_delta"] - delta_target).abs().idxmin()
    return side.loc[idx]


def _settle_pnl(sc: float, sp: float, settle: float, credit: float) -> tuple[float, float]:
    """Return (structure_pnl, net_pnl) at expiration.

    structure_pnl: loss component only (assignment losses × multiplier), passed to portfolio
        which already holds the premium in cash from enter_position.
    net_pnl: full trade P&L including premium, stored in trades CSV.
    """
    call_loss = min(max(0.0, settle - sc), SPREAD_WIDTH)
    put_loss = min(max(0.0, sp - settle), SPREAD_WIDTH)
    structure_pnl = -(call_loss + put_loss) * MULTIPLIER
    net_pnl = (credit - call_loss - put_loss) * MULTIPLIER
    return structure_pnl, net_pnl


class DeltaIC(Strategy):
    """0DTE iron condor — 8-delta short legs, hold to expiration."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "ic_delta"
        self.start_portfolio_amt = 10_000.0
        self.data_sources = {
            "spxw_options": {
                "type": "minio", "root": "SPXW", "provider": "schwab",
                "columns": [
                    "sampled_at", "expiration_date", "contract_type",
                    "strike", "delta", "mark", "underlying_price",
                ],
            },
            "spx_1min": {"type": "local", "loader": "spx_1min"},
        }
        self._open_trade: dict | None = None
        self.trades: list[dict] = []

    def _step(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        trade_date = timestamp.date()
        t = timestamp.time()

        if self._open_trade is not None and self._open_trade["trade_date"] != trade_date:
            self._open_trade = None

        # EOD settlement
        if self._open_trade is not None and t == EOD_BUCKET:
            self._settle(data_slice, timestamp)
            return

        # Entry window: try entry on each 5-min bar in 10:30–11:00
        if self._open_trade is None and ENTRY_START <= t <= ENTRY_END:
            self._try_enter(data_slice, timestamp)

    def _try_enter(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        opts = data_slice["spxw_options"]
        trade_date = timestamp.date()
        t = timestamp.strftime("%H:%M")

        opts_today = opts[pd.to_datetime(opts["expiration_date"]).dt.date == trade_date].copy()
        if opts_today.empty:
            return

        snap = opts_today[opts_today["sampled_at"] == opts_today["sampled_at"].max()]
        if snap.empty:
            return

        short_call = _find_short_strike(snap, "CALL", DELTA_TARGET)
        short_put = _find_short_strike(snap, "PUT", DELTA_TARGET)
        if short_call is None or short_put is None:
            return

        sc = float(short_call["strike"])
        sp = float(short_put["strike"])
        lc_strike = sc + SPREAD_WIDTH
        lp_strike = sp - SPREAD_WIDTH

        lc_row = snap[(snap["contract_type"].str.upper() == "CALL") & (snap["strike"] == lc_strike)]
        lp_row = snap[(snap["contract_type"].str.upper() == "PUT") & (snap["strike"] == lp_strike)]
        if lc_row.empty or lp_row.empty:
            return

        credit = (
            float(short_call["mark"]) + float(short_put["mark"])
            - float(lc_row.iloc[0]["mark"]) - float(lp_row.iloc[0]["mark"])
        )
        if credit < MIN_CREDIT:
            return

        max_risk = SPREAD_WIDTH * MULTIPLIER
        if self.portfolio.free_buying_power < max_risk:
            self.log.warning("%s %s SKIP: insufficient buying power", trade_date, t)
            return

        entry_fees = SCHWAB_FEE_PER_LEG * 4
        self.portfolio.enter_position(
            f"condor_{trade_date}",
            premium=credit * MULTIPLIER,
            fees=entry_fees,
            max_risk=max_risk,
            timestamp=timestamp,
        )
        sc_delta = float(short_call["delta"])
        sp_delta = float(short_put["delta"])
        spot = float(snap["underlying_price"].iloc[0]) if "underlying_price" in snap.columns else float("nan")
        self.log.info(
            "%s %s ENTER: call %s/%s (δ=%.3f) | put %s/%s (δ=%.3f) | credit=$%.2f | spot=%.2f",
            trade_date, t, sc, lc_strike, sc_delta, sp, lp_strike, sp_delta, credit, spot,
        )
        self._open_trade = {
            "key": f"condor_{trade_date}",
            "trade_date": trade_date,
            "sc": sc, "sp": sp, "credit": credit,
        }
        self.trades.append({
            "date": trade_date,
            "entry_time": t,
            "spot": spot,
            "sc": sc, "lc": lc_strike,
            "sp": sp, "lp": lp_strike,
            "sc_delta": sc_delta,
            "sp_delta": sp_delta,
            "credit": credit,
        })

    def _settle(self, data_slice: dict[str, pd.DataFrame], timestamp: datetime) -> None:
        trade = self._open_trade
        candles = data_slice["spx_1min"]
        if candles.empty:
            self._open_trade = None
            return
        settle = float(candles["close"].iloc[-1])
        structure_pnl, net_pnl = _settle_pnl(trade["sc"], trade["sp"], settle, trade["credit"])
        self.log.info(
            "%s EOD: settle=%.2f | pnl=$%.2f",
            trade["trade_date"], settle, net_pnl,
        )
        self.portfolio.exit_position(trade["key"], structure_pnl=structure_pnl, fees=0.0, timestamp=timestamp)
        for rec in self.trades:
            if rec["date"] == trade["trade_date"]:
                rec["settle"] = settle
                rec["pnl"] = net_pnl
                break
        self._open_trade = None


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = DeltaIC()
    engine = BacktestEngine(
        strategy,
        start_date=date(2025, 12, 15),
        end_date=date(2026, 8, 30),
        out_dir=str(OUT_DIR),
        bar_size="5min",
    )
    engine.run()
    if strategy.trades:
        trades_df = pd.DataFrame(strategy.trades)
        out_path = OUT_DIR / "ic_delta_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"Trade log: {out_path}  ({len(trades_df)} trades)")
    else:
        print("No trades recorded.")
