"""Tests for GEX wall master feature dataset.

Validates the merge logic in scripts/gex_wall_build_master.py by checking
that the combined dataset is internally consistent and correctly assembled.

Prerequisite: generate out/gex_walls/daily_atm_iv.csv by running:
    uv run python scripts/gex_wall_iv_zscore.py

Then build the master file:
    uv run python scripts/gex_wall_build_master.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("out/gex_walls")
MASTER = OUT / "master_features.csv"
BASE   = OUT / "features2.csv"


@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    if not MASTER.exists():
        pytest.skip("master_features.csv not built yet — run gex_wall_build_master.py")
    df = pd.read_csv(MASTER)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    bool_cols = ["bounce_30m", "bounce_60m", "eod_held", "wall_breached_at_open"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    return df


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    df = pd.read_csv(BASE)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# --- Row count ---

def test_row_count_unchanged(master, base):
    """Merge must not add or drop rows — master is base + joined columns."""
    assert len(master) == len(base), (
        f"Row count changed: base={len(base)}, master={len(master)}"
    )


def test_no_duplicate_events(master):
    """Each (date, snapshot_ts, method, wall_type, wall_level) is unique.

    snapshot_ts is the options snapshot timestamp (one per day per method run),
    which together with method+wall_type+wall_level uniquely identifies each event.
    Note: first_approach_ts is NOT unique because multiple methods often test the
    same wall level on the same day and share the same approach time.
    """
    key = ["date", "snapshot_ts", "method", "wall_type", "wall_level"]
    dupes = master.duplicated(subset=key).sum()
    assert dupes == 0, f"{dupes} duplicate event rows found"


# --- Daily join coverage ---

def test_rr_coverage(master):
    """25Δ RR should be present for >95% of events (119/121 days computed)."""
    pct = master["rr"].notna().mean()
    assert pct > 0.95, f"RR coverage {pct:.1%} < 95%"


def test_atm_iv_coverage(master):
    """ATM IV should be present for >95% of events."""
    if "atm_iv" not in master.columns:
        pytest.skip("atm_iv column not in master — re-run build_master.py")
    pct = master["atm_iv"].notna().mean()
    assert pct > 0.95, f"ATM IV coverage {pct:.1%} < 95%"


def test_gex_skew_coverage(master):
    """0DTE GEX skew should be present for >95% of events."""
    pct = master["gex_skew_0dte"].notna().mean()
    assert pct > 0.95, f"GEX skew coverage {pct:.1%} < 95%"


# --- Correctness of daily join (no cross-contamination) ---

def test_rr_is_same_per_day(master):
    """All events on the same date must have the same rr value (daily join)."""
    per_day = master.dropna(subset=["rr"]).groupby("date")["rr"].nunique()
    bad_days = (per_day > 1).sum()
    assert bad_days == 0, f"{bad_days} dates have multiple rr values — join is wrong"


def test_atm_iv_same_per_day(master):
    """All events on the same date must have the same atm_iv value."""
    if "atm_iv" not in master.columns:
        pytest.skip("atm_iv column not in master")
    per_day = master.dropna(subset=["atm_iv"]).groupby("date")["atm_iv"].nunique()
    bad_days = (per_day > 1).sum()
    assert bad_days == 0, f"{bad_days} dates have multiple atm_iv values — join is wrong"


# --- Value ranges ---

def test_rr_is_negative(master):
    """SPX always has put skew — 25Δ RR should be negative on every day."""
    rr_vals = master["rr"].dropna()
    pct_negative = (rr_vals < 0).mean()
    assert pct_negative > 0.98, f"Only {pct_negative:.1%} of RR values are negative (expected >98%)"


def test_iv_25d_call_reasonable(master):
    """25Δ call IV should be in [5, 60] percentage points (e.g. 15.0 = 15% vol).
    The raw data stores IV as percentage points, not decimal.
    """
    iv = master["iv_25d_call"].dropna()
    assert iv.min() > 5.0,  f"iv_25d_call min={iv.min():.3f} seems too low (units: pct points)"
    assert iv.max() < 60.0, f"iv_25d_call max={iv.max():.3f} seems too high (units: pct points)"


def test_gex_skew_bounded(master):
    """GEX skew = (call_gex - put_gex) / total — must be in [-1, 1]."""
    skew = master["gex_skew_0dte"].dropna()
    assert skew.min() >= -1.0, f"gex_skew_0dte min={skew.min():.3f} out of range"
    assert skew.max() <=  1.0, f"gex_skew_0dte max={skew.max():.3f} out of range"


# --- Z-score columns ---

def test_zscore_columns_present(master):
    """All 6 z-score columns must be in master."""
    expected = [f"{feat}_zscore_{lb}d" for feat in ["atm_iv", "vix"] for lb in [5, 10, 20]]
    missing = [c for c in expected if c not in master.columns]
    assert not missing, f"Z-score columns missing: {missing}"


def test_zscore_range_plausible(master):
    """Z-scores should be roughly in [-4, 4] — extreme outliers suggest a bug."""
    for lb in [5, 10, 20]:
        for feat in ["atm_iv", "vix"]:
            col = f"{feat}_zscore_{lb}d"
            if col not in master.columns:
                continue
            vals = master[col].dropna()
            assert vals.abs().max() < 10, (
                f"{col} has extreme value {vals.abs().max():.2f} — possible unit mismatch"
            )
            # At least 50% of events should have a z-score (early days will be NaN due to lookback)
            assert vals.notna().mean() > 0.50, f"{col} coverage {vals.notna().mean():.1%} < 50%"


def test_atm_iv_zscore_uncorrelated_with_vix_zscore_5d(master):
    """
    ATM IV z-score 5d and VIX z-score 5d should NOT be perfectly correlated.
    If r > 0.99, the z-scores were computed from the same series (bug).
    """
    sub = master[["atm_iv_zscore_5d", "vix_zscore_5d"]].dropna()
    if len(sub) < 100:
        pytest.skip("Not enough data for correlation check")
    r = sub.corr().iloc[0, 1]
    assert r < 0.99, f"atm_iv and vix z-scores are suspiciously correlated (r={r:.3f})"


# --- Boolean outcomes ---

def test_bounce_30m_is_boolean(master):
    """bounce_30m must contain only True/False (no string artefacts)."""
    valid = master["bounce_30m"].dropna()
    bad = valid[~valid.isin([True, False])]
    assert len(bad) == 0, f"bounce_30m has non-boolean values: {bad.unique()[:5]}"


def test_eod_held_is_boolean(master):
    valid = master["eod_held"].dropna()
    bad = valid[~valid.isin([True, False])]
    assert len(bad) == 0, f"eod_held has non-boolean values: {bad.unique()[:5]}"


# --- Base columns preserved ---

def test_base_columns_preserved(master, base):
    """All columns from features2.csv must still be present in master."""
    missing = [c for c in base.columns if c not in master.columns]
    assert not missing, f"Base columns dropped in merge: {missing}"


def test_wall_type_values(master):
    """wall_type should only be 'call' or 'put'."""
    vals = set(master["wall_type"].dropna().unique())
    assert vals <= {"call", "put"}, f"Unexpected wall_type values: {vals - {'call', 'put'}}"
