"""Load SPX/VIX intraday candles from local tickrake ibkr-paper parquets."""

from datetime import date
from pathlib import Path

import pandas as pd

CANDLES_1MIN_DIR = Path.home() / ".tickrake" / "data" / "candles" / "ibkr-paper" / "1min"
CANDLES_5MIN_DIR = Path.home() / ".tickrake" / "data" / "candles" / "ibkr-paper" / "5min"
TZ = "America/New_York"


def _load_candles(base_dir: Path, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Load yearly parquets for a symbol from a candle directory."""
    frames = []
    for year in range(start.year, end.year + 1):
        path = base_dir / str(year) / f"{symbol}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TZ)
        df = df.drop(columns=["datetime"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(frames).sort_index()
    mask = (combined.index.date >= start) & (combined.index.date < end)
    return combined[mask]


def load_spx_1min(start: date, end: date) -> pd.DataFrame:
    """Load SPX 1-minute OHLCV bars for [start, end)."""
    return _load_candles(CANDLES_1MIN_DIR, "SPX", start, end)


def load_vix9d_1min(start: date, end: date) -> pd.DataFrame:
    """Load VIX9D 1-minute OHLCV bars for [start, end)."""
    return _load_candles(CANDLES_1MIN_DIR, "VIX9D", start, end)


def load_vix_5min(start: date, end: date) -> pd.DataFrame:
    """Load VIX 5-minute OHLCV bars for [start, end)."""
    return _load_candles(CANDLES_5MIN_DIR, "VIX", start, end)
