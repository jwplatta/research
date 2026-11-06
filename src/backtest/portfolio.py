"""Portfolio state tracker for backtest harness."""

from datetime import datetime


class Portfolio:
    """Tracks cash, open positions, trade history, and equity snapshots.

    Fee model:
        - ``premium``: credit received (positive dollars) or debit paid (negative)
        - ``fees``: broker fees (e.g. $1.25 × legs), always a non-negative dollar amount
        - ``structure_pnl``: spread outcome only (credit captured or assignment loss), before fees
        - ``exit_pnl`` stored in trade_history = structure_pnl - entry_fees - exit_fees
    """

    def __init__(self, start_cash: float = 10_000.0) -> None:
        self._cash: float = start_cash
        # {key: {'premium': float, 'entry_fees': float, 'max_risk': float, 'opened_at': datetime}}
        self._positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.equity_history: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def free_cash(self) -> float:
        """Current cash balance (after premiums received and all fees paid)."""
        return self._cash

    @property
    def free_buying_power(self) -> float:
        """Cash minus maximum risk committed by open positions."""
        committed = sum(p["max_risk"] for p in self._positions.values())
        return self._cash - committed

    @property
    def current_equity(self) -> float:
        """Mark-to-open equity: cash plus premium still at risk in open positions.

        Open positions are carried at their entry premium (no mark-to-market).
        For credit strategies this means equity = cash + (credit still on the table).
        """
        open_premium = sum(p["premium"] for p in self._positions.values())
        return self._cash + open_premium

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def enter_position(
        self,
        key: str,
        premium: float,
        fees: float,
        max_risk: float,
        timestamp: datetime,
    ) -> None:
        """Open a position.

        Args:
            key: Unique identifier for this position (e.g. ``'condor_2026-01-15'``).
            premium: Credit received (positive) or debit paid (negative), in dollars.
            fees: Entry-side broker fees in dollars (non-negative).
            max_risk: Maximum possible loss in dollars (e.g. spread_width × 100).
            date: Trade date.

        Raises:
            ValueError: If ``key`` is already open.
            ValueError: If ``free_buying_power < max_risk``.
        """
        if key in self._positions:
            raise ValueError(f"Position '{key}' is already open — cannot enter twice")
        if fees < 0:
            raise ValueError(f"fees must be non-negative, got {fees}")
        if max_risk < 0:
            raise ValueError(f"max_risk must be non-negative, got {max_risk}")
        if self.free_buying_power < max_risk:
            raise ValueError(
                f"Insufficient buying power for '{key}': "
                f"need ${max_risk:,.2f}, have ${self.free_buying_power:,.2f}"
            )

        self._cash += premium
        self._cash -= fees
        self._positions[key] = {
            "premium": premium,
            "entry_fees": fees,
            "max_risk": max_risk,
            "opened_at": timestamp,
        }
        self.trade_history.append({
            "key": key,
            "opened_at": timestamp,
            "closed_at": None,
            "premium": premium,
            "entry_fees": fees,
            "exit_fees": None,
            "total_fees": None,
            "max_risk": max_risk,
            "structure_pnl": None,
            "exit_pnl": None,
            "is_winner": None,
        })

    def exit_position(
        self,
        key: str,
        structure_pnl: float,
        fees: float,
        timestamp: datetime,
    ) -> None:
        """Close an open position.

        Args:
            key: Position identifier matching a prior ``enter_position`` call.
            structure_pnl: Spread P&L before fees (e.g. credit × 100 or -(assignment × 100)).
            fees: Exit-side broker fees in dollars (0 if expiring worthless).
            date: Trade date.

        Raises:
            KeyError: If ``key`` is not currently open.
            ValueError: If ``fees`` is negative.
        """
        if key not in self._positions:
            raise KeyError(f"Cannot exit '{key}' — position not found in open positions")
        if fees < 0:
            raise ValueError(f"fees must be non-negative, got {fees}")

        pos = self._positions.pop(key)
        self._cash += structure_pnl
        self._cash -= fees

        entry_fees = pos["entry_fees"]
        total_fees = entry_fees + fees
        net_pnl = structure_pnl - total_fees

        # Update the matching open record in trade_history
        for record in reversed(self.trade_history):
            if record["key"] == key and record["closed_at"] is None:
                record["closed_at"] = timestamp
                record["exit_fees"] = fees
                record["total_fees"] = total_fees
                record["structure_pnl"] = structure_pnl
                record["exit_pnl"] = net_pnl
                record["is_winner"] = net_pnl > 0
                break
        else:
            raise RuntimeError(
                f"trade_history has no open record for '{key}' — this is a bug"
            )

    def record_equity(self, timestamp: datetime) -> None:
        """Snapshot current equity at this bar's timestamp."""
        self.equity_history.append((timestamp, self.current_equity))
