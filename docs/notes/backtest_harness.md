# Backtest Harness

## Status: Implemented and running

Built in `src/backtest/`. Verification backtest at `backtests/iron_condor_verify.py`.
Run with: `uv run python backtests/iron_condor_verify.py`
Outputs to: `./tmp/backtests/` (charts, CSVs, log file)

**Next task**: re-run the verification backtest and confirm it's clean end-to-end.

---

## Module Structure

```
src/backtest/
  __init__.py        — exports BacktestEngine, Strategy, Portfolio, BacktestResult, get_logger
  engine.py          — BacktestEngine: loads data, iterates bars, calls strategy.step()
  strategy.py        — Strategy ABC: template method, _step() is what subclasses implement
  portfolio.py       — Portfolio: tracks cash, positions, trade_history, equity_history
  result.py          — BacktestResult: writes charts + CSVs after a run
  logger.py          — configure(): sets up file + stderr handlers, one log file per run
  charts/
    equity_curve.py  — plot_equity_curve(dates, equity, out_dir, baseline, title)
    drawdown.py      — plot_drawdown(dates, equity, out_dir, title)
backtests/
  iron_condor_verify.py  — verification strategy: 0DTE SPXW iron condor, ~7-delta, 20pt wide
```

---

## How It Works

### BacktestEngine

```python
BacktestEngine(strategy, start_date, end_date, out_dir, bar_size)
```

- `bar_size` options: `'1min'`, `'5min'`, `'30min'`, `'60min'`, `'1d'`
- On `run()`:
  1. Configures logger → `out_dir/{strategy_name}_{timestamp}.log`
  2. Injects `strategy.portfolio = Portfolio(strategy.start_portfolio_amt)`
  3. Injects `strategy.log = log`
  4. Loads all `'local'` sources upfront (full date range)
  5. Builds `{date: uri}` maps for all `'minio'` sources
  6. Builds trading day calendar from local source index dates
  7. For each trading day:
     - Loads minio data for that day; logs WARNING + skips day if any minio source missing
     - If `bar_size='1d'`: calls `strategy.step(full_day_slice, timestamp=09:30)`
     - If intraday: iterates RTH buckets (9:30–16:00 in bar_size steps); for each bucket slices all sources to `sampled_at < bucket_end` (no lookahead); logs DEBUG + skips bucket if any source has no data yet; calls `strategy.step(bucket_slice, timestamp=bucket_start)`
  8. After loop: `BacktestResult(strategy, out_dir, log).build()`

### Strategy ABC

```python
class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "my_strategy"
        self.start_portfolio_amt = 10_000
        self.data_sources = {
            'spxw_options': {'type': 'minio', 'root': 'SPXW', 'provider': 'schwab'},
            'spx_1min':     {'type': 'local', 'loader': 'spx_1min'},
        }

    def _step(self, data_slice, timestamp):
        # implement strategy logic here
        # self.portfolio and self.log are both available
        ...
```

- Subclasses implement `_step()`, **not** `step()`
- `step()` is a non-abstract template method: calls `_step()` then `self.portfolio.record_equity(timestamp)` unconditionally — equity is recorded on every bar even if `_step()` does nothing
- `self.portfolio` and `self.log` are injected by the engine before `run()` starts

### Portfolio

```python
portfolio.enter_position(key, premium, fees, max_risk, timestamp)
portfolio.exit_position(key, structure_pnl, fees, timestamp)
portfolio.record_equity(timestamp)   # called automatically by Strategy.step()
```

- **`premium`**: credit received (positive) or debit paid (negative), in dollars
- **`fees`**: broker fees — use `$1.25 × legs` for Schwab; tracked separately from structure P&L
- **`max_risk`**: `SPREAD_WIDTH * MULTIPLIER` (gross max loss, not net of credit)
- **`structure_pnl`**: spread outcome only (before fees); fees tracked separately on trade record
- **`free_buying_power`**: `cash - sum(max_risk for open positions)`
- **`equity_history`**: `list[tuple[datetime, float]]` — one entry per bar; `result.py` aggregates to daily for charts
- **`trade_history`** fields: `key, opened_at, closed_at, premium, entry_fees, exit_fees, total_fees, max_risk, structure_pnl, exit_pnl, is_winner`

### Data Sources

```python
# MinIO (SPXW options from tractatus-lakehouse bucket)
{'type': 'minio', 'root': 'SPXW', 'provider': 'schwab'}

# Local (SPX/VIX candles from tickrake ibkr-paper parquets)
{'type': 'local', 'loader': 'spx_1min'}   # ~/.tickrake/data/candles/ibkr-paper/1min/
# loader options: 'spx_1min', 'spx_5min', 'vix_5min', 'vix9d_1min'

# Local (arbitrary parquet file)
{'type': 'local', 'path': '/absolute/path/to/file.parquet'}
```

### Logging

- `configure_logger(out_dir, strategy_name)` → creates `out_dir/{name}_{ts}.log`
- `INFO`/`WARNING` → file + stderr; `DEBUG` → file only (intraday bucket skips go to DEBUG)
- Strategies use `self.log.info(...)` / `self.log.debug(...)` for trace output

### BacktestResult outputs

All written to `out_dir/`:
- `{name}_YYYYMMDD_HHMMSS.png` — equity curve
- `{name}_YYYYMMDD_HHMMSS_drawdown.png` — drawdown chart
- `{name}_trades.csv` — one row per trade (open + close)
- `{name}_summary.csv` — sharpe, edge, win_rate, max_drawdown_pct/amt/duration, avg_win/loss, trade_count, days_not_traded, ending_portfolio_amt/profit

---

## Verification Backtest (`backtests/iron_condor_verify.py`)

**Strategy**: 0DTE SPXW iron condor
- Entry: 10:00–10:30 ET bucket (`bar_size='30min'`, `ENTRY_BUCKET = time(10, 0)`)
- Short ~7-delta call + put (`DELTA_TARGET=0.07`, `DELTA_RANGE=(0.05, 0.10)`)
- 20-point wide spreads (`SPREAD_WIDTH=20`)
- Min net credit: $1.00
- 4 legs × $1.25 Schwab fee = $5.00 entry fees
- Settlement: SPX 1-min candle close at or before 16:00 (not options `underlying_price`)
- Exit at EOD expiration (0 exit fees)
- Max risk: `SPREAD_WIDTH * MULTIPLIER = $2,000`

**Data sources**:
- `spxw_options`: minio, SPXW, schwab
- `spx_1min`: local, tickrake ibkr-paper 1-min SPX candles

**Key implementation notes**:
- Strategy tracks `_open_trade` state across buckets — enter at 10:00, check exit at every subsequent bucket, settle at EOD
- Engine iterates all 30-min buckets regardless of whether the strategy acts; the strategy guard (`_open_trade is not None`) prevents double entry
- Exit conditions checked every bucket: TP when cost-to-close ≤ 25% of entry credit, SL when cost-to-close ≥ 4× entry credit
- EOD settlement at 15:30 bucket (bucket_end = 16:00): engine provides candles sliced to `index < 16:00`, so `iloc[-1]` is the 15:59 SPX close — the correct 0DTE settlement proxy
- Expiry filter: `pd.to_datetime(opts["expiration_date"]).dt.date == trade_date` (not string comparison)
- Long legs looked up by exact strike (`short_strike ± SPREAD_WIDTH`); returns if either missing
- **No `before_time`** on minio source — options data needed all day for exit checks; `columns` projection limits I/O to the 6 fields the strategy uses

**Dec 2025 data gap**: minio files missing for Dec 1–17 2025 and late Feb 2026 — these are logged as WARNING and skipped. Data starts flowing from Dec 18/19 2025.

---

## Known Issues / Watch Points

- **100% win rate in Dec 2025–Feb 2026**: the ~7-delta iron condor has very wide strikes (call ~200 pts OTM, put ~325+ pts OTM on SPX ~6900–7000). SPX didn't breach either side in this period. This is a property of the market, not a bug.
- **`sys.path.insert(0, 'src')`**: backtests add `src/` to path at runtime since `src` is not a package (consistent with `scripts/` pattern in this repo).
