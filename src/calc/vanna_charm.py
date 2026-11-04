"""Vanna and Charm exposure calculations.

Vanna: sensitivity of delta to implied vol (dDelta/dVol = d²V/dS dσ).
  Positive vanna on a call: as IV rises, delta increases (call becomes more sensitive to spot).
  Aggregate vanna exposure (VEX): sum of vanna × OI × spot across all contracts.
  When spot moves toward a high-vanna strike, dealers must hedge by buying/selling vol,
  which creates directional pressure. Rising vol in a negative VEX environment = selling.

Charm: rate of change of delta with respect to time (dDelta/dt = d²V/dS dt).
  Also called delta decay. Positive charm on a call: delta drifts toward 0 as expiry nears.
  Aggregate charm exposure (CEX): sum of charm × OI × spot.
  Charm flows are strongest into close and Friday expiry — they drive systematic directional
  hedging as dealers rebalance delta as time passes.

Both require Black-Scholes d1, d2 per contract. We compute these from:
    - underlying_price (spot S)
    - strike (K)
    - volatility (σ, Schwab implied vol in decimal)
    - expiration_date and snapshot timestamp (time to expiry T in years)
    - risk_free_rate (constant approximation, default 5%)

Input DataFrame must have columns:
    strike, gamma, open_interest, contract_type, expiration_date,
    underlying_price, volatility, sampled_at
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

    # Time to expiry in years
    snap_date = df["sampled_at"].dt.tz_convert("America/New_York").dt.date
    df["T"] = df["expiration_date"].combine(snap_date, lambda exp, snap: max((exp - snap).days, 0) / 365.0)
    df = df[df["T"] > 0]
    if df.empty:
        return df

    S = spot
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["volatility"].to_numpy(dtype=float)
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

    vanna_per_contract = -φ(d1) × d2 / σ   (Black-Scholes)
    sign: calls = +1, puts = -1 (same as GEX convention)
    VEX at strike = sum(vanna × OI × spot²) across contracts at that strike

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


def charm_by_strike(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> pd.DataFrame:
    """Aggregate charm exposure (CEX) by strike.

    charm_per_contract = -φ(d1) × (2r·T - d2·σ·√T) / (2T·σ·√T)   (Black-Scholes)
    sign: calls = +1, puts = -1
    CEX at strike = sum(charm × OI × spot²) across contracts at that strike

    Returns DataFrame with columns: strike, net_charm
    """
    df = _prepare(opts, spot=spot, strike_range=strike_range, risk_free_rate=risk_free_rate)
    if df.empty:
        return pd.DataFrame(columns=["strike", "net_charm"])

    r = risk_free_rate
    phi_d1 = norm.pdf(df["d1"].to_numpy(dtype=float))
    d2 = df["d2"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    sqrt_T = df["sqrt_T"].to_numpy(dtype=float)

    charm = -phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    sign = df["contract_type"].map({"CALL": 1.0, "PUT": -1.0}).to_numpy(dtype=float)

    df["charm_exposure"] = charm * df["open_interest"].to_numpy(dtype=float) * (spot ** 2) * sign

    result = df.groupby("strike", as_index=False)["charm_exposure"].sum()
    return result.rename(columns={"charm_exposure": "net_charm"})


def aggregate_vanna_charm(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Compute aggregate vanna and charm scalars for the full snapshot.

    Returns dict with:
        total_vanna:  signed sum of VEX across all strikes
        total_charm:  signed sum of CEX across all strikes
        call_vanna, put_vanna, call_charm, put_charm: side-split totals
    """
    df = _prepare(opts, spot=spot, strike_range=strike_range, risk_free_rate=risk_free_rate)
    if df.empty:
        return {k: 0.0 for k in ["total_vanna", "total_charm", "call_vanna", "put_vanna", "call_charm", "put_charm"]}

    phi_d1 = norm.pdf(df["d1"].to_numpy(dtype=float))
    d2 = df["d2"].to_numpy(dtype=float)
    sigma = df["sigma"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    sqrt_T = df["sqrt_T"].to_numpy(dtype=float)
    r = risk_free_rate
    oi = df["open_interest"].to_numpy(dtype=float)
    sign = df["contract_type"].map({"CALL": 1.0, "PUT": -1.0}).to_numpy(dtype=float)

    vanna = -phi_d1 * d2 / sigma
    charm = -phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)

    vex = vanna * oi * (spot ** 2) * sign
    cex = charm * oi * (spot ** 2) * sign

    is_call = df["contract_type"].to_numpy() == "CALL"
    return {
        "total_vanna": float(vex.sum()),
        "total_charm": float(cex.sum()),
        "call_vanna":  float(vex[is_call].sum()),
        "put_vanna":   float(vex[~is_call].sum()),
        "call_charm":  float(cex[is_call].sum()),
        "put_charm":   float(cex[~is_call].sum()),
    }
