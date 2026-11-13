"""Vanna exposure calculations.

Do not use Unicode characters in this file (or anywhere in this codebase).
Use plain ASCII only: e.g. spot^2 not spot2, sqrt(T) not sqrt(T).

Vanna: sensitivity of delta to implied vol (dDelta/dVol = d^2V/dS d sig).
  Positive vanna on a call: as IV rises, delta increases (call becomes more sensitive to spot).
  Aggregate vanna exposure (VEX): sum of vanna x OI x spot^2 across all contracts.
  When spot moves toward a high-vanna strike, dealers must hedge by buying/selling vol,
  which creates directional pressure. Rising vol in a negative VEX environment = selling.

Requires Black-Scholes d1, d2 per contract, computed from:
    - underlying_price (spot S)
    - strike (K)
    - volatility (sig, Schwab implied vol as a percentage, e.g. 15.0 = 15%)
    - expiration_date and snapshot timestamp (time to expiry T in years)
    - risk_free_rate (constant approximation, default 5%)

Input DataFrame must have columns:
    strike, gamma, open_interest, contract_type, expiration_date,
    underlying_price, volatility, sampled_at

See src/calc/charm.py for charm (delta decay) exposure calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252


def _prepare(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Parse, filter to strike_range, and compute d1/d2/T for each row."""
    df = opts.copy()
    for col in ["gamma", "open_interest", "underlying_price", "volatility", "strike"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["contract_type"] = df["contract_type"].astype(str).str.upper()
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    df["sampled_at"] = pd.to_datetime(df["sampled_at"], utc=True)

    df = df.dropna(subset=["gamma", "open_interest", "volatility", "strike", "expiration_date", "sampled_at"])
    df = df[df["open_interest"] > 0]
    df = df[df["volatility"] > 0]
    df = df[df["strike"].between(spot - strike_range, spot + strike_range)]
    if df.empty:
        return df

    # Time to expiry in years.
    # For same-day expiry (0DTE), use intraday minutes remaining to 4pm ET so
    # that charm doesn't blow up from a near-zero T denominator.
    # For multi-day expiry, use calendar days / 365.
    TRADING_MINS_PER_YEAR = 252 * 6.5 * 60
    snap_dt = df["sampled_at"].dt.tz_convert("America/New_York")

    def _t_years(exp_date: object, snap: object) -> float:
        import pandas as _pd
        from zoneinfo import ZoneInfo as _ZI
        _ET = _ZI("America/New_York")
        days = (exp_date - snap.date()).days  # type: ignore[union-attr]
        if days > 0:
            return float(days) / 365.0
        close = snap.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=_ET)  # type: ignore[union-attr]
        mins = max((close - snap).total_seconds() / 60.0, 1.0)
        return mins / TRADING_MINS_PER_YEAR

    df["T"] = [_t_years(exp, snap) for exp, snap in zip(df["expiration_date"], snap_dt)]
    df = df[df["T"] > 0]
    if df.empty:
        return df

    S = spot
    K = df["strike"].to_numpy(dtype=float)
    # volatility is stored as a percentage (e.g. 15.0); convert to decimal for BS
    sigma = df["volatility"].to_numpy(dtype=float) / 100.0
    T = df["T"].to_numpy(dtype=float)
    r = risk_free_rate

    sqrt_T = np.sqrt(T)
    df["d1"] = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    df["d2"] = df["d1"] - sigma * sqrt_T
    df["sqrt_T"] = sqrt_T
    df["sigma"] = sigma

    return df


def vanna_by_strike(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Aggregate vanna exposure (VEX) by strike.

    vanna_per_contract = -phi(d1) * d2 / sigma   (Black-Scholes)
    sign: calls = +1, puts = -1 (same as GEX convention)
    VEX at strike = sum(vanna * OI * spot^2) across contracts at that strike

    Returns DataFrame with columns: strike, net_vanna
    """
    df = _prepare(opts, spot=spot, strike_range=strike_range, risk_free_rate=risk_free_rate)
    if df.empty:
        return pd.DataFrame(columns=["strike", "net_vanna"])

    phi_d1 = norm.pdf(df["d1"].to_numpy(dtype=float))
    vanna = -phi_d1 * df["d2"].to_numpy(dtype=float) / df["sigma"].to_numpy(dtype=float)
    sign = df["contract_type"].map({"CALL": 1.0, "PUT": -1.0}).to_numpy(dtype=float)

    df["vanna_exposure"] = vanna * df["open_interest"].to_numpy(dtype=float) * (spot ** 2) * sign

    result = df.groupby("strike", as_index=False)["vanna_exposure"].sum()
    return result.rename(columns={"vanna_exposure": "net_vanna"})


def aggregate_vanna(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Compute aggregate vanna scalar for the full snapshot.

    Returns dict with:
        total_vanna:  signed sum of VEX across all strikes
        call_vanna, put_vanna: side-split totals
    """
    df = _prepare(opts, spot=spot, strike_range=strike_range, risk_free_rate=risk_free_rate)
    if df.empty:
        return {k: 0.0 for k in ["total_vanna", "call_vanna", "put_vanna"]}

    phi_d1 = norm.pdf(df["d1"].to_numpy(dtype=float))
    d2 = df["d2"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    oi = df["open_interest"].to_numpy(dtype=float)
    sign = df["contract_type"].map({"CALL": 1.0, "PUT": -1.0}).to_numpy(dtype=float)

    vanna = -phi_d1 * d2 / sigma
    vex = vanna * oi * (spot ** 2) * sign

    is_call = df["contract_type"].to_numpy() == "CALL"
    return {
        "total_vanna": float(vex.sum()),
        "call_vanna":  float(vex[is_call].sum()),
        "put_vanna":   float(vex[~is_call].sum()),
    }
