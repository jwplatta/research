"""Backtest 3: 0DTE SPXW iron condor with short legs 5 points outside the GEX walls.

Same as ic_gex_wall.py except:
    short call = call_wall + 5
    short put  = put_wall  - 5

Usage:
    uv run python backtests/gex_wall_ic/ic_gex_wall_5pt.py
"""

import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backtest import BacktestEngine, Strategy
from calc.gex import find_raw_wall_strikes

SCHWAB_FEE_PER_LEG = 1.25
SPREAD_WIDTH = 20.0
MULTIPLIER = 100
MIN_CREDIT = 1.00
WALL_OFFSET = 5.0           # points beyond the wall for short legs
ENTRY_START = time(10, 30)
ENTRY_END = time(10, 55)
EOD_BUCKET = time(15, 30)
OUT_DIR = Path("./tmp/backtests/gex_wall_ic")


def _settle_pnl(sc: float, sp: float, settle: float, credit: float) -> tuple[float, float]:
    call_loss = min(max(0.0, settle - sc), SPREAD_WIDTH)
    put_loss = min(max(0.0, sp - settle), SPREAD_WIDTH)
    return -(call_loss + put_loss) * MULTIPLIER, (credit - call_loss - put_loss) * MULTIPLIER


def _lookup_mark(snap: pd.DataFrame, contract_type: str, strike: float) -> float | None:
    row = snap[(snap["contract_type"].str.upper() == contract_type.upper()) & (snap["strike"] == strike)]
    if row.empty or float(row.iloc[0]["mark"]) <= 0:
        return None
    return float(row.iloc[0]["mark"])


def _lookup_delta(snap: pd.DataFrame, contract_type: str, strike: float) -> float | None:
    row = snap[(snap["contract_type"].str.upper() == contract_type.upper()) & (snap["strike"] == strike)]
    if row.empty:
        return None
    return float(row.iloc[0]["delta"])


class GexWallIC5pt(Strategy):
    """0DTE iron condor — short legs 5pt outside GEX walls, hold to expiration."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "ic_gex_wall_5pt"
        self.start_portfolio_amt = 10_000.0
        self.data_sources = {
            "spxw_options": {
                "type": "minio", "root": "SPXW", "provider": "schwab",
                "columns": [
                    "sampled_at", "expiration_date", "contract_type",
                    "strike", "delta", "mark", "gamma", "open_interest", "underlying_price",
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

        if self._open_trade is not None and t == EOD_BUCKET:
            self._settle(data_slice, timestamp)
            return

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

        if "underlying_price" not in snap.columns or snap["underlying_price"].isna().all():
            return
        spot = float(snap["underlying_price"].iloc[0])

        call_wall, put_wall = find_raw_wall_strikes(snap, spot)
        if call_wall is None or put_wall is None:
            return

        sc = float(call_wall) + WALL_OFFSET
        sp = float(put_wall) - WALL_OFFSET
        lc_strike = sc + SPREAD_WIDTH
        lp_strike = sp - SPREAD_WIDTH

        sc_mark = _lookup_mark(snap, "CALL", sc)
        lc_mark = _lookup_mark(snap, "CALL", lc_strike)
        sp_mark = _lookup_mark(snap, "PUT", sp)
        lp_mark = _lookup_mark(snap, "PUT", lp_strike)

        if any(m is None for m in [sc_mark, lc_mark, sp_mark, lp_mark]):
            return

        credit = sc_mark + sp_mark - lc_mark - lp_mark  # type: ignore[operator]
        if credit < MIN_CREDIT:
            return

        max_risk = SPREAD_WIDTH * MULTIPLIER
        if self.portfolio.free_buying_power < max_risk:
            self.log.warning("%s %s SKIP: insufficient buying power", trade_date, t)
            return

        sc_delta = _lookup_delta(snap, "CALL", sc)
        sp_delta = _lookup_delta(snap, "PUT", sp)

        entry_fees = SCHWAB_FEE_PER_LEG * 4
        self.portfolio.enter_position(
            f"condor_{trade_date}",
            premium=credit * MULTIPLIER,
            fees=entry_fees,
            max_risk=max_risk,
            timestamp=timestamp,
        )
        self.log.info(
            "%s %s ENTER: call_wall=%.0f sc=%s/%s (δ=%.3f) | put_wall=%.0f sp=%s/%s (δ=%.3f) | credit=$%.2f | spot=%.2f",
            trade_date, t,
            call_wall, sc, lc_strike, sc_delta or float("nan"),
            put_wall, sp, lp_strike, sp_delta or float("nan"),
            credit, spot,
        )
        self._open_trade = {"key": f"condor_{trade_date}", "trade_date": trade_date, "sc": sc, "sp": sp, "credit": credit}
        self.trades.append({
            "date": trade_date,
            "entry_time": t,
            "spot": spot,
            "call_wall": float(call_wall),
            "put_wall": float(put_wall),
            "sc": sc, "lc": lc_strike,
            "sp": sp, "lp": lp_strike,
            "sc_dist_from_wall": WALL_OFFSET,
            "sp_dist_from_wall": WALL_OFFSET,
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
        self.log.info("%s EOD: settle=%.2f | pnl=$%.2f", trade["trade_date"], settle, net_pnl)
        self.portfolio.exit_position(trade["key"], structure_pnl=structure_pnl, fees=0.0, timestamp=timestamp)
        for rec in self.trades:
            if rec["date"] == trade["trade_date"]:
                rec["settle"] = settle
                rec["pnl"] = net_pnl
                break
        self._open_trade = None


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = GexWallIC5pt()
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
        out_path = OUT_DIR / "ic_gex_wall_5pt_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"Trade log: {out_path}  ({len(trades_df)} trades)")
    else:
        print("No trades recorded.")
