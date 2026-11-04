"""Intraday inflection point detection from 1-min OHLCV bars.

Uses a zigzag algorithm: local extrema with a minimum move filter to
suppress noise. Produces clean swing highs/lows — the actual intraday
pivot points the market respected.
"""

from datetime import time

import pandas as pd

RTH_START = time(9, 30)
RTH_END   = time(16, 0)


def find_inflection_points(
    prices: pd.DataFrame,
    lookback: int = 10,
    min_move_pts: float = 8.0,
) -> pd.DataFrame:
    """Detect intraday swing highs and lows from 1-min bars.

    A bar is a local high candidate if its close is the highest within
    [-lookback, +lookback] bars. Similarly for lows. Candidates are then
    filtered so that consecutive inflections alternate high/low and are
    at least min_move_pts apart — suppressing minor wiggles.

    Args:
        prices: 1-min OHLCV DataFrame with tz-aware DatetimeIndex.
        lookback: Number of bars on each side for local extremum test.
        min_move_pts: Minimum price move from the prior inflection to keep.

    Returns:
        DataFrame with columns: timestamp, price, type ("high" | "low").
        Sorted by timestamp ascending.
    """
    # RTH only
    df = prices[
        (prices.index.time >= RTH_START) & (prices.index.time <= RTH_END)
    ].copy()

    if len(df) < 2 * lookback + 1:
        return pd.DataFrame(columns=["timestamp", "price", "type"])

    closes = df["close"].to_numpy()
    idx    = df.index

    # --- Phase 1: find local extremum candidates ---
    candidates: list[dict] = []
    for i in range(lookback, len(closes) - lookback):
        window = closes[i - lookback : i + lookback + 1]
        c = closes[i]
        if c == window.max():
            candidates.append({"timestamp": idx[i], "price": float(c), "type": "high"})
        elif c == window.min():
            candidates.append({"timestamp": idx[i], "price": float(c), "type": "low"})

    if not candidates:
        return pd.DataFrame(columns=["timestamp", "price", "type"])

    # --- Phase 2: enforce alternation and minimum move ---
    filtered: list[dict] = []
    last: dict | None = None

    for cand in candidates:
        if last is None:
            filtered.append(cand)
            last = cand
            continue

        same_type = cand["type"] == last["type"]
        move = abs(cand["price"] - last["price"])

        if same_type:
            # Replace last if this one is more extreme
            if (cand["type"] == "high" and cand["price"] > last["price"]) or \
               (cand["type"] == "low"  and cand["price"] < last["price"]):
                filtered[-1] = cand
                last = cand
        else:
            # Different type — only keep if move is large enough
            if move >= min_move_pts:
                filtered.append(cand)
                last = cand

    return pd.DataFrame(filtered)
