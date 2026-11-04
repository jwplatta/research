"""Tests for classifier data quality: feature completeness by month.

Catches silent NaN gaps that cause months to be silently dropped from training.

Prerequisite: master_features.csv built by scripts/gex_wall_build_master.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

OUT = Path("out/gex_walls")
MASTER = OUT / "master_features.csv"

# Features used in each classifier
CALL_FEATURES = [
    "vix_open", "atm_iv_zscore_5d", "vix_zscore_20d",
    "iv_25d_call", "gex_skew_0dte",
]
PUT_FEATURES = [
    "net_gex_spot", "iv_25d_put", "atm_iv_zscore_5d",
    "gex_at_wall_strike", "gex_skew_0dte", "vix_open",
]
ALL_FEATURES = list(set(CALL_FEATURES + PUT_FEATURES))

# VIX data starts Jan 2026. vix_zscore_20d needs 20-day lookback, so Jan is
# the warm-up month and will have ~96% NaN — expected. Training starts Feb.
TRAINING_START = date(2026, 2, 1)
TRAINING_END   = date(2026, 8, 31)


@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    if not MASTER.exists():
        pytest.skip("master_features.csv not built yet")
    df = pd.read_csv(MASTER)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    bool_map = {"True": True, "False": False, True: True, False: False}
    for col in ["bounce_60m", "bounce_30m", "eod_held"]:
        if col in df.columns:
            df[col] = df[col].map(bool_map)
    return df


def test_vix_open_coverage_jan_aug(master):
    """vix_open should be present for ≥95% of events Jan–Aug 2026 (no IBKR VIX before Jan 2026)."""
    sub = master[(master["date"] >= TRAINING_START) & (master["date"] <= TRAINING_END)]
    pct = sub["vix_open"].notna().mean()
    assert pct >= 0.95, f"vix_open coverage {pct:.1%} < 95% in Jan–Aug 2026"


def test_no_full_month_dropout_call_features(master):
    """No calendar month in Jan–Aug 2026 should lose >20% of call wall events to NaN features.

    A month dropping >20% signals a data pipeline gap (e.g. z-scores not extended).
    """
    calls = master[(master["wall_type"] == "call") &
                   (master["date"] >= TRAINING_START) &
                   (master["date"] <= TRAINING_END)].copy()
    calls["month"] = pd.to_datetime(calls["date"]).dt.to_period("M")

    for feat in CALL_FEATURES:
        if feat not in calls.columns:
            continue
        monthly = calls.groupby("month")[feat].apply(lambda s: s.isna().mean())
        bad = monthly[monthly > 0.20]
        assert len(bad) == 0, (
            f"Feature '{feat}' missing >20% of call wall events in months: "
            f"{dict(bad.round(2))}"
        )


def test_no_full_month_dropout_put_features(master):
    """No calendar month in Jan–Aug 2026 should lose >20% of put wall events to NaN features."""
    puts = master[(master["wall_type"] == "put") &
                  (master["date"] >= TRAINING_START) &
                  (master["date"] <= TRAINING_END)].copy()
    puts["month"] = pd.to_datetime(puts["date"]).dt.to_period("M")

    for feat in PUT_FEATURES:
        if feat not in puts.columns:
            continue
        monthly = puts.groupby("month")[feat].apply(lambda s: s.isna().mean())
        bad = monthly[monthly > 0.20]
        assert len(bad) == 0, (
            f"Feature '{feat}' missing >20% of put wall events in months: "
            f"{dict(bad.round(2))}"
        )


def test_vix_zscore_20d_coverage_in_training(master):
    """vix_zscore_20d should be present for ≥85% of events in Jan–Aug 2026.

    The first ~20 days (lookback window) will be NaN, so 100% is not expected,
    but 90% dropout in a mid-training month is a pipeline bug.
    """
    sub = master[(master["date"] >= TRAINING_START) & (master["date"] <= TRAINING_END)]
    pct = sub["vix_zscore_20d"].notna().mean()
    assert pct >= 0.85, (
        f"vix_zscore_20d coverage {pct:.1%} in training window — "
        "likely not recomputed after extending date range"
    )


def test_atm_iv_zscore_5d_coverage_in_training(master):
    """atm_iv_zscore_5d should be present for ≥90% of events in Jan–Aug 2026."""
    sub = master[(master["date"] >= TRAINING_START) & (master["date"] <= TRAINING_END)]
    pct = sub["atm_iv_zscore_5d"].notna().mean()
    assert pct >= 0.90, (
        f"atm_iv_zscore_5d coverage {pct:.1%} — "
        "re-run gex_wall_iv_zscore.py and rebuild master"
    )


def test_training_days_count(master):
    """Should have ≥120 trading days of data in the Jan–Aug 2026 training window."""
    sub = master[(master["date"] >= TRAINING_START) & (master["date"] <= TRAINING_END)]
    n_days = sub["date"].nunique()
    assert n_days >= 110, f"Only {n_days} training days in {TRAINING_START}–{TRAINING_END} window"


def test_bounce_60m_coverage_in_training(master):
    """Target variable bounce_60m should be present for ≥95% of training events."""
    sub = master[(master["date"] >= TRAINING_START) & (master["date"] <= TRAINING_END)]
    pct = sub["bounce_60m"].notna().mean()
    assert pct >= 0.95, f"bounce_60m coverage {pct:.1%} < 95% in training window"
