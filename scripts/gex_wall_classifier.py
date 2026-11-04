"""GEX wall solidity classifiers — call wall and put wall, separate models.

Target: bounce_60m (wall holds for ~1 hour after first approach)

Train: Dec 2025 – May 2026
Test:  Jun 2026 – Aug 2026

Features:
  Call wall model: vix_open, atm_iv_zscore_5d, vix_zscore_20d, iv_25d_call,
                   gex_at_wall_strike, gex_skew_0dte
  Put wall model:  net_gex_spot, iv_25d_put, atm_iv_zscore_5d,
                   gex_at_wall_strike, gex_skew_0dte, vix_open

Pipeline: StandardScaler → LogisticRegression → CalibratedClassifierCV
Evaluation: ROC-AUC, precision/recall at thresholds, calibration curve
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats

OUT_DIR = Path("out/gex_walls")

# Full training window — Sept/Oct 2026 reserved for live test once data is available
TRAIN_END = date(2026, 8, 31)

TARGET = "bounce_60m"

CALL_FEATURES = [
    "vix_open",
    "atm_iv_zscore_5d",
    "vix_zscore_20d",
    "iv_25d_call",
    "gex_at_wall_strike_z20d",   # z-scored version (regime-stable)
    "gex_skew_0dte",
]

PUT_FEATURES = [
    "net_gex_spot_z20d",         # z-scored version (regime-stable)
    "iv_25d_put",
    "atm_iv_zscore_5d",
    "gex_at_wall_strike_z20d",   # z-scored version (regime-stable)
    "gex_skew_0dte",
    "vix_open",
]

BOOL_COLS = ["bounce_30m", "bounce_60m", "eod_held", "wall_breached_at_open"]


def rolling_zscore_daily(df: pd.DataFrame, col: str, lb: int, out_col: str) -> pd.DataFrame:
    """Compute rolling z-score of a column using daily means (leak-free)."""
    daily = (
        df[["date", col]]
        .dropna(subset=[col])
        .groupby("date")[col]
        .mean()
        .sort_index()
    )
    mu = daily.rolling(lb, min_periods=lb // 2).mean()
    sd = daily.rolling(lb, min_periods=lb // 2).std()
    z  = ((daily - mu) / sd).rename(out_col)
    return df.join(z, on="date")


def load_master() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "master_features.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    # Regime-stable GEX features: z-score over 20-day rolling window
    df = rolling_zscore_daily(df, "gex_at_wall_strike", 20, "gex_at_wall_strike_z20d")
    df = rolling_zscore_daily(df, "net_gex_spot",       20, "net_gex_spot_z20d")

    return df


def prepare_data(df: pd.DataFrame, features: list[str], wall_type: str):
    sub = df[(df["wall_type"] == wall_type) & (df["date"] <= TRAIN_END)].copy()
    sub = sub.dropna(subset=features + [TARGET])
    sub = sub[sub[TARGET].isin([True, False])]
    sub = sub.sort_values("date")

    X = sub[features].values
    y = sub[TARGET].astype(int).values

    return X, y, sub


def build_pipeline() -> Pipeline:
    base = LogisticRegression(max_iter=1000, random_state=42)
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])


def evaluate_cv(X, y, features: list[str], label: str, sub: pd.DataFrame,
                val_days: int = 20) -> Pipeline:
    """Walk-forward CV split on trading days (not rows).

    Each fold: train on all days up to cutoff, validate on next val_days trading days.
    This avoids the problem of daily features (vix_open, etc.) being repeated across
    many event rows — we split on days so the validation window is a clean future period.
    """
    dates = sub["date"].values
    unique_days = sorted(set(dates))
    n_days = len(unique_days)

    # Build folds: at least 60 train days, then slide val_days forward
    min_train_days = 60
    fold_starts = range(min_train_days, n_days - val_days, val_days)
    if len(fold_starts) == 0:
        raise ValueError(f"Not enough days ({n_days}) for CV with min_train={min_train_days}, val={val_days}")

    base_rate = y.mean()
    fold_aucs = []
    all_proba = np.full(len(y), np.nan)

    for cutoff_i in fold_starts:
        train_days = set(unique_days[:cutoff_i])
        val_days_  = set(unique_days[cutoff_i: cutoff_i + val_days])

        tr_idx  = np.where([d in train_days for d in dates])[0]
        val_idx = np.where([d in val_days_  for d in dates])[0]

        if len(tr_idx) < 50 or len(val_idx) < 10:
            continue

        pipe = build_pipeline()
        pipe.fit(X[tr_idx], y[tr_idx])
        proba = pipe.predict_proba(X[val_idx])[:, 1]
        auc   = roc_auc_score(y[val_idx], proba)
        fold_aucs.append(auc)
        all_proba[val_idx] = proba

        td0 = unique_days[0 if cutoff_i == min_train_days else cutoff_i - val_days]
        td1 = unique_days[cutoff_i - 1]
        vd0 = unique_days[cutoff_i]
        vd1 = unique_days[min(cutoff_i + val_days - 1, n_days - 1)]
        print(f"    Fold train {td0}→{td1} ({len(tr_idx):,} rows) | val {vd0}→{vd1} ({len(val_idx):,} rows) | AUC={auc:.3f}")

    print(f"\n{'='*60}")
    print(f"  {label}  (walk-forward CV on trading days, {val_days}d val windows)")
    print(f"{'='*60}")
    print(f"  Training events: {len(y):,}  ({len(unique_days)} trading days)")
    print(f"  Base hold rate:  {base_rate:.1%}")
    print(f"  Mean ROC-AUC:    {np.mean(fold_aucs):.4f}  (±{np.std(fold_aucs):.4f})")
    print()

    # Aggregate OOF predictions (rows that were in any validation window)
    oof_mask  = ~np.isnan(all_proba)
    proba_oof = all_proba[oof_mask]
    y_oof     = y[oof_mask]

    print(f"\n  OOF threshold analysis (target: {TARGET}):")
    print(f"  {'Threshold':>10}  {'N signals':>10}  {'Precision':>10}  {'Lift':>8}  {'Coverage':>10}")
    print(f"  {'-'*58}")
    for thresh in [0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = proba_oof >= thresh
        n    = mask.sum()
        if n == 0:
            print(f"  {thresh:>10.0%}  {'—':>10}")
            continue
        prec = y_oof[mask].mean()
        lift = prec / base_rate
        cov  = n / len(y_oof)
        print(f"  {thresh:>10.0%}  {n:>10,}  {prec:>10.1%}  {lift:>8.2f}x  {cov:>10.1%}")

    # Calibration on OOF
    print(f"\n  OOF calibration (predicted vs actual hold rate):")
    print(f"  {'Pred bucket':>12}  {'N':>8}  {'Actual':>10}  {'Delta':>8}")
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        mask = (proba_oof >= lo) & (proba_oof < hi)
        n    = mask.sum()
        if n == 0:
            continue
        actual = y_oof[mask].mean()
        pred_mid = (lo + hi) / 2
        delta = actual - pred_mid
        print(f"  {lo:.0%}–{hi:.0%}        {n:>8,}  {actual:>10.1%}  {delta:>+8.1%}")

    # Binomial test at 70% threshold on OOF
    for thresh in [0.65, 0.70]:
        mask = proba_oof >= thresh
        n, k = mask.sum(), y_oof[mask].sum()
        if n > 10:
            p_val = stats.binomtest(k, n, base_rate, alternative="greater").pvalue
            prec  = k / n
            print(f"\n  Binomial test (OOF) at {thresh:.0%}: n={n}, precision={prec:.1%}, null={base_rate:.1%}, p={p_val:.4f}")

    # Fit final model on all training data
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    # Coefficients from final model
    try:
        lr = final_pipe.named_steps["clf"].calibrated_classifiers_[0].estimator
        coefs = dict(zip(features, lr.coef_[0]))
        print(f"\n  Final model coefficients (trained on all data):")
        for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "↑ hold" if coef > 0 else "↓ hold"
            print(f"    {feat:<35} {coef:>+.4f}  {direction}")
    except Exception:
        pass

    return final_pipe


def main() -> None:
    print("Loading master features...")
    df = load_master()
    print(f"  {len(df):,} events, {df['date'].nunique()} days")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Training through: {TRAIN_END} (Sept-Oct 2026 reserved for live test)")

    results = {}

    for wall_type, features, label in [
        ("call", CALL_FEATURES, "CALL WALL MODEL"),
        ("put",  PUT_FEATURES,  "PUT WALL MODEL"),
    ]:
        print(f"\n--- {label} ---")
        X, y, sub = prepare_data(df, features, wall_type)
        print(f"  Training events: {len(X):,} ({sub['date'].nunique()} days)")

        if len(X) < 200:
            print("  Skipping — insufficient data")
            continue

        pipe = evaluate_cv(X, y, features, label, sub)
        results[wall_type] = (pipe, features)

    # Save final models' in-sample predictions for inspection
    print("\n--- Saving in-sample predictions ---")
    for wall_type, (pipe, features) in results.items():
        sub = df[(df["wall_type"] == wall_type) & (df["date"] <= TRAIN_END)]
        sub = sub.dropna(subset=features + [TARGET])
        sub = sub[sub[TARGET].isin([True, False])]
        proba = pipe.predict_proba(sub[features].values)[:, 1]
        out = sub[["date", "method", "wall_type", "wall_level", "spot",
                   "first_approach_ts", TARGET]].copy()
        out["pred_prob"] = proba
        out_path = OUT_DIR / f"predictions_{wall_type}_insample.csv"
        out.to_csv(out_path, index=False)
        print(f"  Saved → {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
