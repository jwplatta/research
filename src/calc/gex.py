"""GEX (Gamma Exposure) calculation functions.

Copied from tractatus/calc/gex.py. No tractatus dependency — pure numpy/pandas.

Key functions:
    net_gex_by_strike()         — aggregate GEX by strike across all expiries
    find_raw_wall_strikes()     — SpotGamma-style: OTM peak magnitude, no DTE weight
    find_aggregate_wall_strikes() — DTE-weighted aggregate or per-expiry clustering
    find_decision_zones()       — zone clustering with magnitude + persistence score

Input DataFrame must have columns:
    strike, gamma, open_interest, contract_type ("CALL"/"PUT"), expiration_date
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd


def _side_gex_rows(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float,
) -> pd.DataFrame:
    """Return side-specific GEX rows within the visible strike range."""
    df = opts.copy()
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["K"] = pd.to_numeric(df["strike"], errors="coerce")
    df["contract_type"] = df["contract_type"].astype(str).str.upper()
    df = df.dropna(subset=["gamma", "open_interest", "K", "contract_type"])
    df = df[df["open_interest"] > 0]

    mask = (df["K"] >= spot - strike_range) & (df["K"] <= spot + strike_range)
    df = df[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["K", "gex", "expiration_date"])

    sign = df["contract_type"].map({"CALL": 1.0, "PUT": -1.0})
    df["gex"] = df["gamma"] * df["open_interest"] * (spot**2) * sign
    df = df.dropna(subset=["gex"])
    if "expiration_date" in df.columns:
        df["expiration_date"] = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date
    else:
        df["expiration_date"] = pd.NaT
    return df


def _aggregate_side_gex_by_strike(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float,
    anchor_date: pd.Timestamp | None = None,
    weight_by_distance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregated call and put GEX-by-strike frames within the visible range."""
    df = _side_gex_rows(opts, spot=spot, strike_range=strike_range)
    if df.empty:
        empty = pd.DataFrame(columns=["K", "gex"])
        return empty, empty

    if weight_by_distance:
        anchor = anchor_date.date() if isinstance(anchor_date, pd.Timestamp) else anchor_date
        if anchor is not None and df["expiration_date"].notna().any():
            dte = pd.to_datetime(df["expiration_date"], errors="coerce").dt.date.map(
                lambda exp: max((exp - anchor).days, 1) if pd.notna(exp) else 1
            )
            df["weighted_gex"] = df["gex"] / dte.astype(float)
        else:
            df["weighted_gex"] = df["gex"]
        value_col = "weighted_gex"
    else:
        value_col = "gex"

    calls = df[df["contract_type"] == "CALL"].groupby("K", as_index=False)[value_col].sum()
    puts = df[df["contract_type"] == "PUT"].groupby("K", as_index=False)[value_col].sum()
    return calls.rename(columns={value_col: "gex"}), puts.rename(columns={value_col: "gex"})  # type: ignore[call-overload]


def _clustered_wall_candidates(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clustered per-expiry OTM wall candidates for calls and puts."""
    df = _side_gex_rows(opts, spot=spot, strike_range=strike_range)
    if df.empty or df["expiration_date"].isna().all():
        empty = pd.DataFrame(columns=["K", "count", "abs_gex"])
        return empty, empty

    calls = df[(df["contract_type"] == "CALL") & (df["K"] >= spot)].copy()
    puts = df[(df["contract_type"] == "PUT") & (df["K"] <= spot)].copy()

    call_walls = pd.DataFrame(columns=["K", "gex"])
    put_walls = pd.DataFrame(columns=["K", "gex"])
    if not calls.empty:
        per_exp_call = calls.groupby(["expiration_date", "K"], as_index=False)["gex"].sum()
        call_walls = (
            per_exp_call.sort_values(["expiration_date", "gex"], ascending=[True, False])  # type: ignore[call-overload]
            .groupby("expiration_date", as_index=False)
            .first()[["K", "gex"]]
        )
    if not puts.empty:
        per_exp_put = puts.groupby(["expiration_date", "K"], as_index=False)["gex"].sum()
        put_walls = (
            per_exp_put.sort_values(["expiration_date", "gex"], ascending=[True, True])  # type: ignore[call-overload]
            .groupby("expiration_date", as_index=False)
            .first()[["K", "gex"]]
        )

    call_clusters = (
        call_walls.assign(abs_gex=lambda d: d["gex"].abs())
        .groupby("K", as_index=False)
        .agg(count=("K", "size"), abs_gex=("abs_gex", "sum"))
    )
    put_clusters = (
        put_walls.assign(abs_gex=lambda d: d["gex"].abs())
        .groupby("K", as_index=False)
        .agg(count=("K", "size"), abs_gex=("abs_gex", "sum"))
    )
    return call_clusters, put_clusters


def _bs_gamma(
    s: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
    r: float = 0.0,
    q: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Black-Scholes gamma."""
    sqrt_t = np.sqrt(np.maximum(t, 1e-10))
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
    pdf_d1 = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    result: npt.NDArray[np.float64] = pdf_d1 / (s * sigma * sqrt_t)
    return result


def net_gex_by_strike(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
) -> pd.DataFrame:
    """Compute net GEX by strike. Returns DataFrame[strike, net_gex].

    Formula: gamma * open_interest * spot² per contract.
    Calls positive, puts negative. Filtered to ±strike_range around spot.
    """
    df = opts.copy()

    is_call = (df["contract_type"] == "CALL").to_numpy(dtype=bool)
    k = pd.to_numeric(df["strike"], errors="coerce").to_numpy(dtype=float)
    oi = pd.to_numeric(df["open_interest"], errors="coerce").to_numpy(dtype=float)
    gam = pd.to_numeric(df["gamma"], errors="coerce").to_numpy(dtype=float)

    gex_each = gam * oi * (spot**2)
    sign = np.where(is_call, 1.0, -1.0)
    net_gex_each = gex_each * sign

    gex_df = pd.DataFrame({"strike": k, "net_gex": net_gex_each})
    net: pd.DataFrame = gex_df.groupby("strike")["net_gex"].sum().reset_index()

    mask = (net["strike"] >= spot - strike_range) & (net["strike"] <= spot + strike_range)
    return net[mask].reset_index(drop=True)


def find_raw_wall_strikes(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
) -> tuple[float | None, float | None]:
    """Return SpotGamma-style call and put wall strikes.

    Call wall: OTM strike with highest raw net call GEX above spot.
    Put wall: OTM strike with most negative raw net put GEX below spot.
    No DTE weighting — pure peak open_interest × gamma.
    """
    calls, puts = _aggregate_side_gex_by_strike(
        opts, spot=spot, strike_range=strike_range, anchor_date=None, weight_by_distance=False,
    )
    otm_calls = calls[calls["K"] >= spot]
    otm_puts = puts[puts["K"] <= spot]

    call_source = otm_calls if not otm_calls.empty else calls
    put_source = otm_puts if not otm_puts.empty else puts

    call_wall = (
        None if call_source.empty
        else float(call_source.loc[call_source["gex"].idxmax(), "K"])  # type: ignore[arg-type, index]
    )
    put_wall = (
        None if put_source.empty
        else float(put_source.loc[put_source["gex"].idxmin(), "K"])  # type: ignore[arg-type, index]
    )
    return call_wall, put_wall


def find_aggregate_wall_strikes(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    method: str = "distance_weighted_aggregate",
    anchor_date: pd.Timestamp | None = None,
) -> tuple[float | None, float | None]:
    """Return dominant call and put wall strikes.

    Methods:
    - distance_weighted_aggregate: OTM GEX by strike, inverse-DTE weighted (near-term matters more)
    - per_expiry_clustering: wall per expiry first, then clustered by repeated occurrence
    """
    if method == "per_expiry_clustering":
        calls, puts = _clustered_wall_candidates(opts, spot=spot, strike_range=strike_range)
        if calls.empty and puts.empty:
            return None, None
        call_wall = (
            None if calls.empty
            else float(
                calls.sort_values(["count", "abs_gex", "K"], ascending=[False, False, True]).iloc[0]["K"]
            )
        )
        put_wall = (
            None if puts.empty
            else float(
                puts.sort_values(["count", "abs_gex", "K"], ascending=[False, False, False]).iloc[0]["K"]
            )
        )
        return call_wall, put_wall

    calls, puts = _aggregate_side_gex_by_strike(
        opts, spot=spot, strike_range=strike_range, anchor_date=anchor_date, weight_by_distance=True,
    )
    if calls.empty and puts.empty:
        return None, None

    otm_calls = calls[calls["K"] >= spot]
    otm_puts = puts[puts["K"] <= spot]
    call_source = otm_calls if not otm_calls.empty else calls
    put_source = otm_puts if not otm_puts.empty else puts

    call_wall = (
        None if call_source.empty
        else float(call_source.loc[call_source["gex"].idxmax(), "K"])  # type: ignore[arg-type, index]
    )
    put_wall = (
        None if put_source.empty
        else float(put_source.loc[put_source["gex"].idxmin(), "K"])  # type: ignore[arg-type, index]
    )
    return call_wall, put_wall


def _distance_weighted_zone_candidates(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float,
    anchor_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _side_gex_rows(opts, spot=spot, strike_range=strike_range)
    calls, puts = _aggregate_side_gex_by_strike(
        opts, spot=spot, strike_range=strike_range, anchor_date=anchor_date, weight_by_distance=True,
    )
    if rows.empty:
        empty = pd.DataFrame(columns=["K", "score"])
        return empty, empty

    total_expiries = max(int(rows["expiration_date"].dropna().nunique()), 1)

    def _build_candidates(side_rows: pd.DataFrame, side_agg: pd.DataFrame) -> pd.DataFrame:
        if side_agg.empty:
            return pd.DataFrame(columns=["K", "score"])
        persistence = (
            side_rows.groupby("K")["expiration_date"].nunique().rename("expiry_count").reset_index()
        )
        candidates = side_agg.merge(persistence, on="K", how="left").fillna({"expiry_count": 0})
        mag = candidates["gex"].abs()
        max_mag = float(mag.max()) or 1.0
        candidates["mag_score"] = mag / max_mag
        candidates["persistence_score"] = candidates["expiry_count"] / total_expiries
        candidates["score"] = (
            0.70 * candidates["mag_score"] + 0.30 * candidates["persistence_score"]
        )
        return candidates[["K", "score"]]

    call_rows = rows[(rows["contract_type"] == "CALL") & (rows["K"] >= spot)]
    put_rows = rows[(rows["contract_type"] == "PUT") & (rows["K"] <= spot)]
    call_candidates = _build_candidates(call_rows, calls[calls["K"] >= spot])
    put_candidates = _build_candidates(put_rows, puts[puts["K"] <= spot])
    return call_candidates, put_candidates


def _cluster_candidates_into_zones(
    candidates: pd.DataFrame,
    top_n: int,
    merge_gap: float = 25.0,
    zone_pad: float = 5.0,
    max_zone_width: float = 20.0,
) -> list[dict[str, float]]:
    if candidates.empty or top_n <= 0:
        return []

    candidates = candidates.sort_values("K").reset_index(drop=True)

    max_score = float(candidates["score"].max()) or 1.0
    score_floor = max(0.6 * max_score, float(candidates["score"].quantile(0.75)))
    strong = candidates[candidates["score"] >= score_floor].copy()
    min_required = min(len(candidates), max(top_n * 4, top_n))
    if len(strong) < min_required:
        strong = candidates.nlargest(min_required, "score").copy()
    candidates = strong.sort_values("K").reset_index(drop=True)

    rows = candidates.to_dict("records")
    peaks: list[dict[str, float]] = []
    for idx, row in enumerate(rows):
        score = float(row["score"])
        prev_score = float(rows[idx - 1]["score"]) if idx > 0 else float("-inf")
        next_score = float(rows[idx + 1]["score"]) if idx + 1 < len(rows) else float("-inf")
        if score >= prev_score and score >= next_score:
            peaks.append({"K": float(row["K"]), "score": score})

    if not peaks:
        best_k = float(candidates.loc[candidates["score"].idxmax(), "K"])  # type: ignore[arg-type, index]
        peaks = [{"K": best_k, "score": max_score}]

    peak_threshold = 0.8
    used_ranges: list[tuple[float, float]] = []
    zones: list[dict[str, float]] = []
    for peak in sorted(peaks, key=lambda item: item["score"], reverse=True):
        peak_k = float(peak["K"])
        peak_score = float(peak["score"])
        group = [
            {"K": float(row["K"]), "score": float(row["score"])}
            for row in rows
            if abs(float(row["K"]) - peak_k) <= merge_gap
            and float(row["score"]) >= peak_score * peak_threshold
        ]
        if not group:
            group = [peak]

        strikes = np.array([item["K"] for item in group], dtype=float)
        scores = np.array([item["score"] for item in group], dtype=float)
        total_score = float(scores.sum())
        center = (
            float(np.average(strikes, weights=scores)) if total_score > 0 else float(strikes.mean())
        )
        low = max(float(strikes.min() - zone_pad), center - max_zone_width / 2.0)
        high = min(float(strikes.max() + zone_pad), center + max_zone_width / 2.0)

        overlaps_existing = any(
            not (high < existing_low or low > existing_high)
            for existing_low, existing_high in used_ranges
        )
        if overlaps_existing:
            continue
        zones.append({"low": low, "high": high, "center": center, "score": total_score})
        used_ranges.append((low, high))

    zones.sort(key=lambda zone: zone["score"], reverse=True)
    return zones[:top_n]


def find_decision_zones(
    opts: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
    anchor_date: pd.Timestamp | None = None,
    top_n: int = 2,
    merge_gap: float = 25.0,
    zone_pad: float = 5.0,
    max_zone_width: float = 20.0,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return resistance (calls) and support (puts) zones.

    Uses distance-weighted aggregate scoring: 70% magnitude + 30% persistence
    (how many expiries agree on the same strike).

    Returns:
        (resistance_zones, support_zones) — each a list of dicts with keys:
        low, high, center, score. Sorted by score descending, up to top_n zones.
    """
    call_candidates, put_candidates = _distance_weighted_zone_candidates(
        opts, spot=spot, strike_range=strike_range, anchor_date=anchor_date,
    )
    resistance_zones = _cluster_candidates_into_zones(
        call_candidates, top_n=top_n, merge_gap=merge_gap, zone_pad=zone_pad,
        max_zone_width=max_zone_width,
    )
    support_zones = _cluster_candidates_into_zones(
        put_candidates, top_n=top_n, merge_gap=merge_gap, zone_pad=zone_pad,
        max_zone_width=max_zone_width,
    )
    return resistance_zones, support_zones


def find_zero_gamma_level(
    snap: pd.DataFrame,
    spot: float,
    strike_range: float = 300.0,
) -> float | None:
    """Find the price where cumulative net GEX (by price) crosses zero.

    Uses the net GEX curve sorted by strike and interpolates the first sign
    change. Returns None if no crossing found within strike_range of spot.

    Ported from tractatus/calc/gex.py::find_zero_gamma_level.
    """
    gex_df = net_gex_by_strike(snap, spot=spot, strike_range=strike_range)
    gex_df = gex_df.dropna(subset=["net_gex"]).sort_values("strike")
    if gex_df.empty:
        return None

    prices = gex_df["strike"].to_numpy(dtype=float)
    gex    = gex_df["net_gex"].to_numpy(dtype=float)

    sign = np.sign(gex)
    sign_filled = sign.copy()
    last_nonzero: float | None = None
    for i in range(len(sign)):
        if sign[i] != 0:
            last_nonzero = float(sign[i])
        elif last_nonzero is not None:
            sign_filled[i] = last_nonzero

    idx = np.where(np.diff(sign_filled) != 0)[0]
    if len(idx) == 0:
        return None

    i = int(idx[0])
    x1, x2 = float(prices[i]), float(prices[i + 1])
    y1, y2 = float(gex[i]), float(gex[i + 1])
    zgl = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else (x1 + x2) / 2.0
    return float(zgl)


def net_gex_at_spot(
    snap: pd.DataFrame,
    spot: float,
    window: float = 25.0,
) -> float:
    """Sum of net GEX across all strikes within ±window pts of spot.

    Positive = dealer long gamma (dampening regime).
    Negative = dealer short gamma (amplifying regime).
    """
    gex_df = net_gex_by_strike(snap, spot=spot, strike_range=window)
    gex_df = gex_df.dropna(subset=["net_gex"])
    return float(gex_df["net_gex"].sum()) if not gex_df.empty else 0.0


def gex_at_strike(
    snap: pd.DataFrame,
    spot: float,
    strike: float,
    tolerance: float = 2.5,
) -> float:
    """Net GEX at a specific strike (sum within ±tolerance pts).

    Returns 0.0 if no data near that strike.
    """
    gex_df = net_gex_by_strike(snap, spot=spot, strike_range=abs(strike - spot) + 50.0)
    gex_df = gex_df.dropna(subset=["net_gex"])
    near = gex_df[gex_df["strike"].between(strike - tolerance, strike + tolerance)]
    return float(near["net_gex"].sum()) if not near.empty else 0.0
