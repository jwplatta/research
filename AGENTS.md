# AGENTS

This repository is for researching trading ideas.

## Markdown Front Matter

**Every markdown file written in this repo must include the following front matter at the top**, especially files in `docs/notes/`:

```markdown
---
created_at: YYYY-MM-DD
tags:
description:
---
```

Fill in `created_at` with the current date, `tags` with relevant topic tags (e.g. `backtest`, `spxw`, `options`) with a 3 tag max for any one note, and `description` with a one-line summary of the file's purpose.


## Backtesting

### Harness Overview

Backtests use the custom harness in `src/backtest/`. See [`docs/notes/backtest_harness.md`](docs/notes/backtest_harness.md) for full module documentation, strategy API, portfolio API, and data source configuration.

Key modules:
- `BacktestEngine` — loads data, iterates bars, calls `strategy.step()`
- `Strategy` (ABC) — subclass and implement `_step(data_slice, timestamp)`
- `Portfolio` — tracks cash, positions, P&L; available as `self.portfolio` in `_step`
- `BacktestResult` — writes equity curve, drawdown chart, trades CSV, summary CSV

### File Organization

- **All backtest scripts go in a subfolder of `backtests/`**, not at the root of `backtests/`.
- Each subfolder represents a core research focus (e.g. `gex_wall_ic/`, `rv_iv_entry_ic/`).
- Multiple related scripts (e.g. parameter variants) can share one subfolder.
- Name subfolders to describe the research question, not the instrument alone.

```
backtests/
  gex_wall_ic/          ← iron condor entry near GEX walls
    ic_gex_wall_15pt.py
    ic_gex_wall_20pt.py
  rv_iv_entry_ic/       ← IV/RV ratio as IC entry filter
    rv_iv_entry_ic.py
```

### Output Directory

All backtest output (charts, CSVs, logs) must go to:

```
out/backtests/{subfolder-name}/
```

where `{subfolder-name}` matches the script's parent folder under `backtests/`. Set `out_dir` in the `BacktestEngine` call accordingly:

```python
engine = BacktestEngine(
    MyStrategy(),
    start_date=date(2025, 12, 15),
    end_date=date(2026, 8, 30),
    out_dir="./out/backtests/gex_wall_ic",   # matches backtests/gex_wall_ic/
    bar_size="30min",
)
```

### Writing a Strategy

1. Add `sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))` at the top (adjust depth to reach repo root from the subfolder).
2. Subclass `Strategy`, set `self.name`, `self.start_portfolio_amt`, and `self.data_sources`.
3. Implement `_step(self, data_slice, timestamp)` — do not override `step()`.
4. `self.portfolio` and `self.log` are injected by the engine before `run()` starts.

Minimal skeleton:

```python
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from backtest import BacktestEngine, Strategy


class MyStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "my_strategy"
        self.start_portfolio_amt = 10_000.0
        self.data_sources = {
            "spxw_options": {"type": "minio", "root": "SPXW", "provider": "schwab"},
            "spx_1min":     {"type": "local", "loader": "spx_1min"},
        }

    def _step(self, data_slice: dict, timestamp) -> None:
        opts = data_slice["spxw_options"]
        candles = data_slice["spx_1min"]
        # ... strategy logic using self.portfolio and self.log ...


if __name__ == "__main__":
    BacktestEngine(
        MyStrategy(),
        start_date=date(2025, 12, 15),
        end_date=date(2026, 8, 30),
        out_dir="./out/backtests/my_research_focus",
        bar_size="30min",
    ).run()
```

### Data Sources

```python
# MinIO — SPXW options snapshots from tractatus-lakehouse
{"type": "minio", "root": "SPXW", "provider": "schwab"}

# MinIO with time filter and column projection (reduces I/O)
{"type": "minio", "root": "SPXW", "provider": "schwab",
 "before_time": time(12, 30),
 "columns": ["sampled_at", "expiration_date", "contract_type", "strike", "mark", "delta"]}

# Local — SPX/VIX intraday candles from tickrake ibkr-paper parquets
{"type": "local", "loader": "spx_1min"}   # options: spx_1min, spx_5min, vix_5min, vix9d_1min

# Local — arbitrary parquet
{"type": "local", "path": "/absolute/path/to/file.parquet"}
```

### Portfolio API

```python
self.portfolio.enter_position(key, premium, fees, max_risk, timestamp)
self.portfolio.exit_position(key, structure_pnl, fees, timestamp)
# self.portfolio.record_equity() is called automatically — do not call manually
```

- `premium`: credit received (positive) or debit paid (negative), in dollars
- `fees`: use `$1.25 × legs` for Schwab
- `max_risk`: `SPREAD_WIDTH * MULTIPLIER` (gross max loss, before credit)
- `structure_pnl`: spread outcome only, before fees

### Notes and Examples

- Full harness reference: [`docs/notes/backtest_harness.md`](docs/notes/backtest_harness.md)
- Working examples: `backtests/orb_backtest.py`, `backtests/iron_condor_verify.py`, `backtests/gex_wall_ic/`

If you encounter a technique that isn't obvious from these examples (e.g. multi-leg exit logic, intraday Greeks interpolation), write a short note in `docs/notes/` with a clear description and minimal code example. Keep example notes focused — one concept per file.


## Research Data

### Option Chain Snapshots

Option chain samples are stored in the **tractatus-lakehouse** MinIO bucket. Use
`src/util/minio.py` helpers (`get_duckdb_conn`, `option_parquet_uris`) to read them.

Path pattern inside the bucket:
```
s3://tractatus-lakehouse/options/schwab/YYYY/MM/DD/{root}_samples_YYYY-MM-DD.parquet
```

**Timestamps in these files are UTC.** Convert to Central Time (America/Chicago) after
loading before any time-of-day logic or display.

**Never read from the local tickrake collector directory
(`~/.tickrake/data/options/`) unless the user explicitly asks you to.** The
lakehouse is the authoritative source for option chain data.

### SPX / Equity Candle Data (IBKR)

5-min and 1-min SPX (and other symbol) candles are stored locally at:
```
~/.tickrake/data/candles/ibkr-paper/{bar_size}/{year}/{symbol}.parquet
```

If a date range is not covered by the local parquet files, fetch it from IBKR via
`ib_insync`. See [`docs/notes/ibkr_candle_fetch.md`](docs/notes/ibkr_candle_fetch.md)
for the standard approach. Use this whenever a script or analysis needs intraday bar
data that isn't already present locally.

---

## Skills

Use `skillex` for skill management.

- Find available skills with `skillex list`.
- Install skills with `skillex pull <skill-name> --agent codex`.
- Update skills with `skillex update <skill-name> --agent codex`.

Use `skillex` first when the task is to install, update, or discover shared skills.

This repo also vendors Python workflow skills locally under `skills/`. Prefer these local copies for Python project setup, testing, and code-quality work in this repository:

- `skills/python-project-setup`
- `skills/python-testing`
- `skills/python-code-quality`

Use the `quant-studies` skill for designing, running, and iterating on cross-sectional equity backtests and related study workflows in this repository.
