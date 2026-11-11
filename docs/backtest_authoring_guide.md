# Backtest Authoring Guide

Notes on how to correctly write strategies using the `src/backtest/` harness.
Add to the gotchas section whenever something bites us.

---

## Portfolio Model

The `Portfolio` class tracks cash and open positions. Understanding the flow is critical to
getting correct P&L.

### Entry

```python
portfolio.enter_position(key, premium=credit * MULTIPLIER, fees=entry_fees, max_risk=max_risk, ...)
```

- `premium` (positive = credit received) is **immediately added to cash**
- `max_risk` is reserved from buying power but does NOT reduce cash
- Equity during the trade = `cash + sum(open premiums)` — the open premium is carried at entry value
  (no mark-to-market)

### Exit

```python
portfolio.exit_position(key, structure_pnl=..., fees=exit_fees, ...)
```

- `structure_pnl` is **added to cash on top of the premium already in cash**
- For a credit spread, this means `structure_pnl` should represent **only the assignment/close loss**,
  NOT the full net P&L including premium

---

## Gotchas

### 1. Double-counting premium on credit spreads

**The mistake:** passing `(credit - losses) * MULTIPLIER` as `structure_pnl`.

```python
# WRONG — double counts the credit
pnl = (credit - call_loss - put_loss) * MULTIPLIER
portfolio.exit_position(key, structure_pnl=pnl, ...)
```

Since `portfolio.enter_position` already added `credit * MULTIPLIER` to cash, adding it again
at exit inflates equity by roughly 2× on winning trades and understates losses.

**The fix:** pass only the **loss component** as `structure_pnl`, and separately track real net P&L
for logging/CSV output.

```python
# CORRECT
call_loss = min(max(0.0, settle - sc), SPREAD_WIDTH)
put_loss  = min(max(0.0, sp - settle), SPREAD_WIDTH)

structure_pnl = -(call_loss + put_loss) * MULTIPLIER   # passed to portfolio
net_pnl       = (credit - call_loss - put_loss) * MULTIPLIER  # stored in trades CSV / logged

portfolio.exit_position(key, structure_pnl=structure_pnl, ...)
```

**How to catch it:** after a run, sum the `pnl` column in your trades CSV and compare to
`ending_portfolio_amt - start_portfolio_amt` in the summary CSV. They should be close
(difference = fees). A large discrepancy means double-counting.

**Affected files (found Nov 2026):** `backtests/gex_wall_ic/ic_delta.py`,
`backtests/gex_wall_ic/ic_gex_wall.py`, and the original `backtests/iron_condor_verify.py`.

**Corollary — fixing structure_pnl alone breaks win_rate and Sharpe:**
The portfolio computes `exit_pnl = premium + structure_pnl - total_fees`. If you fix
`structure_pnl` to losses-only without the portfolio including `premium`, every trade shows
`exit_pnl ≈ -fees` → win_rate = 0.0, Sharpe wildly negative. The two must move together.
The fix lives in `src/backtest/portfolio.py:exit_position`.

---

### 2. GEX walls can be nearly ATM on 0DTE chains

`find_raw_wall_strikes()` returns the OTM strike with peak net GEX magnitude. On a 0DTE chain,
the highest OI concentration is often clustered very close to spot (5–30 points on a ~7000 index),
so "the wall" can have a delta of 0.4–0.5 — essentially ATM.

If you place short legs at the wall without a minimum-OTM guard, you end up selling near-ATM
spreads, not the typical low-delta OTM structure. The result is very high premium and a completely
different risk profile than intended.

**Fix:** add a minimum distance filter (e.g. `abs(wall - spot) >= MIN_WALL_DISTANCE`) or a maximum
delta threshold before accepting a wall as a valid short leg placement.

---

### 3. Silent skips in `_try_enter` make no-trade days opaque

If the strategy returns early without logging (credit too low, no strike found, long leg missing),
the log shows nothing — no ENTER and no SKIP. This makes it hard to audit why a specific day
had no trade.

**Fix:** add a debug-level log at each early-exit condition, or at minimum log a single summary
line at the end of `_try_enter` if no trade was placed.

---

## Verification Checklist

Before trusting backtest results:

1. `sum(trades_csv["pnl"]) ≈ ending_portfolio_amt - start_portfolio_amt` (within fees)
2. Spot-check 2–3 trade rows: verify `credit`, `sc_delta`, `sp_delta` are in the expected range
3. Check that "no trade" days have a plausible reason (data gap, credit below min, etc.)
4. For GEX-based strategies: verify average `abs(sc - spot)` and `sc_delta` are in the OTM
   range you intended
