"""Load SPX/VIX intraday candles from data/candles/ in the repo.

Candle parquets live at: data/candles/{symbol}_{barsize}_{year}.parquet
e.g. data/candles/spx_5min_2026.parquet

Use scripts/fetch_spx_candles.py to fetch missing dates from IBKR.
Do NOT read from or write to ~/.tickrake.
"""

from datetime import date
from pathlib import Path

import pandas as pd

CANDLES_DIR = Path(__file__).parent.parent.parent / "data" / "candles"
TZ = "America/New_York"


def _load_candles(symbol: str, barsize: str, start: date, end: date) -> pd.DataFrame:
    """Load yearly parquets for a symbol/barsize from data/candles/."""
    frames = []
    for year in range(start.year, end.year + 1):
        path = CANDLES_DIR / f"{symbol.lower()}_{barsize}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        df.index = df.index.tz_convert(TZ)
        mask = (df.index.date >= start) & (df.index.date < end)
        frames.append(df[mask])

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    return pd.concat(frames).sort_index()


def load_spx_1min(start: date, end: date) -> pd.DataFrame:
    """Load SPX 1-minute OHLCV bars for [start, end)."""
    return _load_candles("spx", "1min", start, end)


def load_vix9d_1min(start: date, end: date) -> pd.DataFrame:
    """Load VIX9D 1-minute OHLCV bars for [start, end)."""
    return _load_candles("vix9d", "1min", start, end)


def load_spx_5min(start: date, end: date) -> pd.DataFrame:
    """Load SPX 5-minute OHLCV bars for [start, end)."""
    return _load_candles("spx", "5min", start, end)


def load_vix_5min(start: date, end: date) -> pd.DataFrame:
    """Load VIX 5-minute OHLCV bars for [start, end)."""
    return _load_candles("vix", "5min", start, end)
