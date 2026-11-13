# Black-Scholes Greeks on SPXW Option Chains

You are going to build a complete options pricing pipeline from raw trade data.
By the end you will have computed implied volatility, delta, gamma, vega, and theta
for every contract in a real SPXW option chain, and extended the chain into the
wings using vol smile interpolation.

Follow each step in order. Every step tells you exactly what to write and what to
run to confirm it worked before moving on.

---

## What You Are Working With

The data comes from the provider dataset -- purchased trade history for SPXW options.
Each row is a contract that printed a trade in a given minute:

| Column | What it is |
|--------|-----------|
| `sampled_at` | Minute the trade happened (America/Chicago) |
| `contract_type` | "CALL" or "PUT" |
| `strike` | Strike price |
| `expiration_date` | Expiry date |
| `close` | Last trade price in this minute |
| `total_volume` | Contracts traded in this minute |

**What is NOT in the data:** spot price, implied volatility, delta, gamma, vega, theta.
You will compute all of those yourself. That is the whole point.

---

## Part 0 -- Load and Prepare the Data (Run This, Don't Change It)

This part is already written for you. Run it to get a clean 5-minute option chain
with spot prices attached. You will use this data for all exercises that follow.

### Step 0.1 -- Load one day of SPXW trades

```python
from datetime import date
from src.util.minio import get_duckdb_conn, option_parquet_uris

con = get_duckdb_conn()
uri = option_parquet_uris("SPXW", date(2024, 3, 1), date(2024, 3, 2), provider="massive")[0]

raw = con.execute(f"""
    SELECT sampled_at, expiration_date, contract_type, strike,
           open, high, low, close, total_volume
    FROM read_parquet('{uri}', union_by_name=true)
    WHERE expiration_date = '2024-03-01'
      AND close IS NOT NULL
    ORDER BY sampled_at, strike
""").fetchdf()

print(raw.shape)
print(raw.head())
```

You should see roughly 20,000 rows. The `close` column has real trade prices.
The `volatility`, `delta`, `gamma` columns do not exist in this file at all.

### Step 0.2 -- Resample to 5-minute buckets

The raw data is at 1-minute resolution but only for contracts that actually traded
that minute. This resamples to a 5-minute grid using the last trade price in each
bucket. Contracts that go quiet for more than 3 buckets (15 minutes) are dropped
rather than forward-filled with stale prices.

```python
import pandas as pd
from datetime import date
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STALE_THRESHOLD = 3


def build_5min_index(trade_date: date) -> pd.DatetimeIndex:
    start = pd.Timestamp(trade_date).replace(hour=9, minute=30, tzinfo=ET)
    end   = pd.Timestamp(trade_date).replace(hour=16, minute=0,  tzinfo=ET)
    return pd.date_range(start, end, freq="5min")


def resample_chain(raw_df: pd.DataFrame, stale_threshold: int = STALE_THRESHOLD) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["sampled_at"] = pd.to_datetime(raw_df["sampled_at"], utc=True).dt.tz_convert(ET)

    dates = raw_df["sampled_at"].dt.date.unique()
    target_idx = pd.DatetimeIndex(
        [ts for d in sorted(dates) for ts in build_5min_index(d)]
    )

    id_cols = ["expiration_date", "contract_type", "strike"]
    raw_df["bucket"] = raw_df["sampled_at"].dt.floor("5min")
    agg = raw_df.groupby(["bucket"] + id_cols).agg(
        close=("close", "last"),
        total_volume=("total_volume", "sum"),
    ).reset_index().rename(columns={"bucket": "sampled_at"})

    close_wide = agg.pivot_table(
        index="sampled_at", columns=id_cols, values="close", aggfunc="last"
    ).reindex(target_idx)

    traded = close_wide.notna()
    stale_count = traded.apply(
        lambda col: col.groupby((col != col.shift()).cumsum()).cumcount().where(~col, 0)
    )
    close_filled = close_wide.ffill()
    close_filled[stale_count > stale_threshold] = float("nan")

    result = (
        close_filled
        .stack(id_cols, future_stack=True)
        .rename("close")
        .reset_index()
        .dropna(subset=["close"])
    )
    stale_long = (
        stale_count
        .stack(id_cols, future_stack=True)
        .rename("buckets_since_trade")
        .reset_index()
    )
    result = result.merge(stale_long, on=["sampled_at"] + id_cols, how="left")
    return result.sort_values(["sampled_at", "strike"]).reset_index(drop=True)


chain = resample_chain(raw)
print(chain["sampled_at"].nunique(), "5-min buckets")
print(chain.head())
```

You should see 79 buckets covering 09:30 to 16:00 ET.

### Step 0.3 -- Join SPX spot price

```python
from src.util.spx_candles import load_spx_1min

spx = load_spx_1min(date(2024, 3, 1), date(2024, 3, 2))
spx_5min = spx["close"].resample("5min").last().ffill().rename("spot")
spx_5min.index.name = "sampled_at"

chain = chain.merge(spx_5min.reset_index(), on="sampled_at", how="left")
print(chain[["sampled_at", "strike", "contract_type", "close", "spot"]].head())
```

Each row now has a `close` (trade price) and a `spot` (SPX price at that moment).

### Step 0.4 -- Pick one snapshot to work with

For all exercises below you will work on a single 5-minute snapshot.
Grab the 11:00 ET bucket:

```python
snap = chain[chain["sampled_at"] == pd.Timestamp("2024-03-01 11:00", tz=ET)].copy()
print(f"{len(snap)} contracts in snapshot")
print(snap[["strike", "contract_type", "close", "spot"]].head(10))
```

You should see roughly 100-150 rows, all with real trade prices and spot ~5100.

---

## Part 1 -- The Black-Scholes Formula

Black-Scholes prices a European option using five inputs:

```
C = S * N(d1) - K * exp(-r*T) * N(d2)    (call)
P = K * exp(-r*T) * N(-d2) - S * N(-d1)  (put)
```

| Symbol | Meaning |
|--------|---------|
| S | Spot price |
| K | Strike price |
| r | Risk-free rate (decimal, use 0.05) |
| T | Time to expiry in years |
| sigma | Implied volatility (decimal) |
| N(x) | Cumulative normal CDF |

The two intermediate values everything else depends on:

```
d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
```

The foundation functions `bs_d1`, `bs_d2`, and `bs_call_price` are already
implemented in `src/calc/greeks.py`. Read through them now. The Greeks you will
write use the same d1 and d2.

### Step 1.1 -- Verify the foundation functions

Run this to confirm they work:

```python
from src.calc.greeks import bs_d1, bs_d2, bs_call_price

S, K, r, sigma, T = 5100.0, 5100.0, 0.05, 0.15, 1/252
d1 = bs_d1(S, K, r, sigma, T)
d2 = bs_d2(d1, sigma, T)
price = bs_call_price(S, K, r, sigma, T)
print(f"d1={d1:.4f}  d2={d2:.4f}  call price=${price:.2f}")
```

You should see: `d1=0.0501  d2=-0.0444  call price=$21.47` (approximately).

---

## Part 2 -- Time to Expiry

Before computing anything you need T, the time to expiry in years, for each row.

For a 0DTE contract with hours left until 4pm, using calendar days would give T
close to zero and blow up the formulas. Instead, use intraday minutes remaining.

### Step 2.1 -- Write the T helper

Open `src/calc/greeks.py`. Find `greeks_df`. Before you fill in that stub, write
this standalone helper in your notebook or script -- you will use it inside
`greeks_df` later:

```python
from zoneinfo import ZoneInfo
import pandas as pd

ET = ZoneInfo("America/New_York")
TRADING_MINS_PER_YEAR = 252 * 6.5 * 60  # 98280

def t_years(expiration_date, sampled_at) -> float:
    """Time to expiry in years. Uses intraday minutes for 0DTE."""
    snap = pd.Timestamp(sampled_at).tz_convert(ET)
    exp  = pd.Timestamp(expiration_date).date()
    days = (exp - snap.date()).days
    if days > 0:
        return float(days) / 365.0
    close_ts = snap.replace(hour=16, minute=0, second=0, microsecond=0, tzinfo=ET)
    mins = max((close_ts - snap).total_seconds() / 60.0, 1.0)
    return mins / TRADING_MINS_PER_YEAR
```

### Step 2.2 -- Verify T on your snapshot

```python
row = snap.iloc[0]
T = t_years(row["expiration_date"], row["sampled_at"])
print(f"T = {T:.6f} years = {T * 252 * 390:.1f} trading minutes")
```

For the 11:00 ET snapshot with 0DTE, you should see T around 0.003 years
(roughly 300 trading minutes remaining).

---

## Part 3 -- Implied Volatility

You have a trade price (`close`) but no sigma. You need to find the sigma that
makes the BS price equal to the observed trade price. That sigma is called
**implied volatility (IV)**.

This is a root-finding problem: find sigma such that `BS_price(sigma) - market_price = 0`.
You will use `scipy.optimize.brentq` which searches a bracketed interval.

### Step 3.1 -- Write implied_vol()

Add this function to your script (it is also shown in `src/calc/greeks.py` for reference):

```python
from scipy.optimize import brentq
import numpy as np

def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, flag: str) -> float | None:
    """Return IV as a decimal, or None if solver fails."""
    def put_price(S, K, r, sigma, T):
        return bs_call_price(S, K, r, sigma, T) - S + K * np.exp(-r * T)

    pricer = bs_call_price if flag.upper() == "CALL" else put_price
    intrinsic = max(S - K, 0.0) if flag.upper() == "CALL" else max(K - S, 0.0)
    if market_price <= intrinsic + 1e-6:
        return None  # price is at intrinsic; no vol solution exists

    try:
        return brentq(lambda sigma: pricer(S, K, r, sigma, T) - market_price,
                      1e-6, 10.0, xtol=1e-6)
    except ValueError:
        return None
```

### Step 3.2 -- Verify on a single row

```python
row = snap[snap["contract_type"] == "CALL"].sort_values("strike").iloc[10]
S   = float(row["spot"])
K   = float(row["strike"])
T   = t_years(row["expiration_date"], row["sampled_at"])
iv  = implied_vol(float(row["close"]), S, K, T, 0.05, "CALL")
print(f"Strike {K:.0f} call  close=${row['close']:.2f}  IV={iv:.1%}")
```

You should get an IV somewhere between 10% and 30% depending on the strike.

### Step 3.3 -- Compute IV for every row in the snapshot

```python
ivs = []
for _, row in snap.iterrows():
    T   = t_years(row["expiration_date"], row["sampled_at"])
    iv  = implied_vol(float(row["close"]), float(row["spot"]),
                      float(row["strike"]), T, 0.05, str(row["contract_type"]))
    ivs.append(iv)

snap = snap.copy()
snap["T"]  = [t_years(r["expiration_date"], r["sampled_at"]) for _, r in snap.iterrows()]
snap["iv"] = ivs
snap = snap.dropna(subset=["iv"]).reset_index(drop=True)
print(f"{len(snap)} contracts with valid IV")
print(snap[["strike", "contract_type", "close", "iv"]].head(10))
```

You should see IVs in the 10-30% range for near-ATM contracts. Deep OTM contracts
where the solver fails (minimum tick price $0.05 with no vol solution) are dropped.

---

## Part 4 -- The Greeks

Each Greek tells you how the option price moves when one input changes.
You will implement four of them. Open `src/calc/greeks.py` now -- you will
write your code into the stubs there.

All four use d1 (and sometimes d2) which you already know how to compute.
`norm.cdf(x)` is the cumulative normal CDF. `norm.pdf(x)` is the normal PDF (bell curve).

### Step 4.1 -- Delta

**What it means:** How much the option price changes per $1 move in spot.

**Formula:**
```
Call delta = N(d1)        ranges 0 to 1
Put  delta = N(d1) - 1   ranges -1 to 0
```

**Write this in the `delta()` stub in `src/calc/greeks.py`:**

```python
d1 = bs_d1(S, K, r, sigma, T)
if flag.upper() == "CALL":
    return norm.cdf(d1)
else:
    return norm.cdf(d1) - 1
```

**Verify:**

```python
from src.calc.greeks import delta

S, K, r, sigma, T = 5100.0, 5100.0, 0.05, 0.15, 1/252
print(delta(S, K, r, sigma, T, "CALL"))  # expect ~0.50
print(delta(S, K, r, sigma, T, "PUT"))   # expect ~-0.50
print(delta(S, 5200.0, r, sigma, T, "CALL"))  # expect near 0 (far OTM)
```

### Step 4.2 -- Gamma

**What it means:** How fast delta changes as spot moves. Same for calls and puts.

**Formula:**
```
Gamma = phi(d1) / (S * sigma * sqrt(T))
```

**Write this in the `gamma()` stub:**

```python
d1      = bs_d1(S, K, r, sigma, T)
phi_d1  = norm.pdf(d1)
return phi_d1 / (S * sigma * np.sqrt(T))
```

**Verify:**

```python
from src.calc.greeks import gamma

S, K, r, sigma, T = 5100.0, 5100.0, 0.05, 0.15, 1/252
print(gamma(S, K, r, sigma, T))   # expect a small positive number ~0.008
# Gamma peaks ATM -- check it falls off away from spot:
print(gamma(S, 5200.0, r, sigma, T))  # expect much smaller
```

### Step 4.3 -- Vega

**What it means:** How much the option price changes per 1-point move in IV.
Same for calls and puts.

**Formula:**
```
Vega = S * phi(d1) * sqrt(T)
```

Divide by 100 to get the conventional "per 1% change in vol" units.

**Write this in the `vega()` stub:**

```python
d1     = bs_d1(S, K, r, sigma, T)
phi_d1 = norm.pdf(d1)
return S * phi_d1 * np.sqrt(T) / 100.0
```

**Verify:**

```python
from src.calc.greeks import vega

S, K, r, sigma, T = 5100.0, 5100.0, 0.05, 0.15, 1/252
print(vega(S, K, r, sigma, T))  # expect ~0.21 (price changes $0.21 per 1% vol move)
```

### Step 4.4 -- Theta

**What it means:** How much the option price decays per calendar day.
Almost always negative -- options lose value as time passes.

**Formula:**
```
Call theta = -(S * phi(d1) * sigma) / (2 * sqrt(T))  -  r*K*exp(-r*T)*N(d2)
Put  theta = -(S * phi(d1) * sigma) / (2 * sqrt(T))  +  r*K*exp(-r*T)*N(-d2)
```

Divide by 365 at the end to convert from per-year to per-calendar-day.

**Write this in the `theta()` stub:**

```python
d1     = bs_d1(S, K, r, sigma, T)
d2     = bs_d2(d1, sigma, T)
phi_d1 = norm.pdf(d1)
decay  = -(S * phi_d1 * sigma) / (2 * np.sqrt(T))
if flag.upper() == "CALL":
    carry = -r * K * np.exp(-r * T) * norm.cdf(d2)
else:
    carry =  r * K * np.exp(-r * T) * norm.cdf(-d2)
return (decay + carry) / 365.0
```

**Verify:**

```python
from src.calc.greeks import theta

S, K, r, sigma, T = 5100.0, 5100.0, 0.05, 0.15, 1/252
print(theta(S, K, r, sigma, T, "CALL"))  # expect a negative number, around -15 to -25
print(theta(S, K, r, sigma, T, "PUT"))   # similar, slightly different due to carry term
```

---

## Part 5 -- Apply Greeks to the Full Snapshot

Now wire everything together. Open `src/calc/greeks.py` and implement `greeks_df()`.
Your snapshot already has `iv` and `T` columns from Part 3.

### Step 5.1 -- Implement greeks_df()

**Write this in the `greeks_df()` stub:**

```python
df = opts_snapshot.copy()

# Vectorize inputs
S     = df["spot"].to_numpy(dtype=float)
K     = df["strike"].to_numpy(dtype=float)
T_arr = df["T"].to_numpy(dtype=float)
sigma = df["iv"].to_numpy(dtype=float)
flags = df["contract_type"].str.upper().tolist()

# Compute Greeks row by row (loop is fine; chain is small)
deltas, gammas, vegas, thetas = [], [], [], []
for i in range(len(df)):
    deltas.append(delta(S[i], K[i], r, sigma[i], T_arr[i], flags[i]))
    gammas.append(gamma(S[i], K[i], r, sigma[i], T_arr[i]))
    vegas.append(vega(S[i],  K[i], r, sigma[i], T_arr[i]))
    thetas.append(theta(S[i], K[i], r, sigma[i], T_arr[i], flags[i]))

df["delta"] = deltas
df["gamma"] = gammas
df["vega"]  = vegas
df["theta"] = thetas
return df
```

### Step 5.2 -- Run it on your snapshot

```python
from src.calc.greeks import greeks_df

result = greeks_df(snap)
spot_val = float(snap["spot"].iloc[0])
atm = result[(result["strike"] - spot_val).abs() <= 10]
print(atm[["strike", "contract_type", "iv", "delta", "gamma", "vega", "theta"]])
```

You should see something like:

```
 strike contract_type    iv  delta  gamma  vega   theta
   5095          CALL  0.14   0.56   0.08  0.21  -18.4
   5095           PUT  0.17  -0.44   0.08  0.21  -17.9
   5100          CALL  0.14   0.50   0.09  0.22  -19.1
   5100           PUT  0.18  -0.50   0.09  0.22  -18.6
```

Notice the IV differs between call and put at the same strike. That is the vol skew.

---

## Part 6 -- The Vol Smile and Chain Extension

Around spot the chain is dense. But outside the actively traded zone, many strikes
are missing. Rather than leaving them out, you will fit a vol smile to the traded
strikes and use it to synthesize prices across a fixed strike range (4000-8000).

You will fit two models and compare them:
- **PchipInterpolator** -- passes through every traded IV exactly, monotone between
  knots, no oscillation. Does not extrapolate outside the fitted range.
- **Polynomial regression (degree 3)** -- least-squares fit across all traded strikes.
  Smooths over noisy ticks. Extrapolates, for better or worse.

Open `src/calc/vol_surface.py`. You will implement `fit_vol_smile()` and `extend_chain()`.

### Step 6.1 -- Plot the smile to see what you are fitting

Run `compute_iv` first to add the IV column, then plot:

```python
from src.calc.vol_surface import compute_iv

snap_iv = compute_iv(snap)

import matplotlib.pyplot as plt

calls = snap_iv[snap_iv["contract_type"] == "CALL"].sort_values("strike")
puts  = snap_iv[snap_iv["contract_type"] == "PUT"].sort_values("strike")
spot  = float(snap_iv["spot"].iloc[0])

plt.figure(figsize=(10, 4))
plt.plot(calls["strike"], calls["iv"], "b.-", label="calls")
plt.plot(puts["strike"],  puts["iv"],  "r.-", label="puts")
plt.axvline(spot, color="k", linestyle="--", label="spot")
plt.xlabel("strike")
plt.ylabel("IV")
plt.title("Vol smile -- 2024-03-01 11:00 ET")
plt.legend()
plt.show()
```

You should see a curve that slopes upward as you move away from spot in both directions.
Puts carry higher IV than calls at equidistant strikes -- that is the skew.

### Step 6.2 -- Implement fit_vol_smile()

Open `src/calc/vol_surface.py` and find `fit_vol_smile()`. Write this:

```python
from scipy.interpolate import PchipInterpolator
import numpy as np

result = {
    "CALL": {"pchip": None, "poly": None},
    "PUT":  {"pchip": None, "poly": None},
}
df = snap_with_iv[snap_with_iv["iv"] >= min_iv].copy()
spot = float(df["spot"].iloc[0])
df["log_m"] = np.log(df["strike"] / spot)

for side in ("CALL", "PUT"):
    sub = df[df["contract_type"].str.upper() == side].sort_values("log_m")
    sub = sub.drop_duplicates(subset=["log_m"])
    if len(sub) < min_strikes:
        continue
    x = sub["log_m"].to_numpy()
    y = sub["iv"].to_numpy()
    result[side]["pchip"] = PchipInterpolator(x, y, extrapolate=False)
    coeffs = np.polyfit(x, y, poly_degree)
    result[side]["poly"]  = np.poly1d(coeffs)

return result
```

Why log-moneyness instead of raw strike: it centers the x-axis at 0 (ATM) so the
smile shape is comparable across days regardless of where spot is.

Why two models: Pchip passes through every point exactly -- great for interpolation
between traded strikes, but returns NaN outside them. Polynomial smooths over noise
and extrapolates -- useful to see what it predicts in the wings, but can curve wrong
at the edges.

**Verify:**

```python
from src.calc.vol_surface import fit_vol_smile

smiles = fit_vol_smile(snap_iv)

import numpy as np
x = np.linspace(-0.04, 0.04, 80)
pchip_curve = smiles["CALL"]["pchip"](x)
poly_curve  = smiles["CALL"]["poly"](x)

plt.figure(figsize=(10, 4))
plt.plot(x, pchip_curve, "b-",  label="pchip (interpolating)")
plt.plot(x, poly_curve,  "g--", label="poly degree-3 (smoothing)")
plt.scatter(
    np.log(calls["strike"] / spot),
    calls["iv"],
    color="b", s=15, zorder=5, label="traded IVs"
)
plt.xlabel("log-moneyness log(K/S)")
plt.ylabel("IV")
plt.title("Call smile -- two models")
plt.legend()
plt.show()
```

Look at where the two curves agree and where they diverge. In the wings (beyond the
last traded strike) pchip returns NaN; poly keeps going. Whether the poly extrapolation
is sensible depends on how well a cubic fits the smile shape.

### Step 6.3 -- Implement extend_chain()

Find `extend_chain()` in `src/calc/vol_surface.py`. Implement it step by step.

**Step A -- build the variable-density strike grid:**

The chain needs two densities. Near spot you want 5-pt resolution; 500 points
away from spot a 5-pt grid gives you nothing useful and thousands of rows to price.

`_build_strike_grid()` is already implemented in `vol_surface.py`. Call it:

```python
from src.calc.vol_surface import _build_strike_grid

spot = float(snap_with_iv["spot"].iloc[0])
T    = float(snap_with_iv["T"].iloc[0])

candidates = _build_strike_grid(
    core_low=4800,  core_high=5400,  core_step=5,    # fine grid around spot
    tail_low=4000,  tail_high=8000,  tail_step=25,   # coarse grid in the wings
)
print(f"{len(candidates)} candidate strikes")
print("First 5:", candidates[:5])    # 4000, 4025, 4050...
print("Around core low:", candidates[candidates >= 4790][:5])  # ...4775, 4800, 4805...
```

You should see the grid transition from 25-pt spacing to 5-pt spacing at 4800.

**Step B -- for each side and each missing candidate strike, interpolate both IVs:**

```python
for side in ("CALL", "PUT"):
    pchip = smiles[side]["pchip"]
    poly  = smiles[side]["poly"]
    if pchip is None and poly is None:
        continue

    traded_strikes = set(
        snap_with_iv[snap_with_iv["contract_type"].str.upper() == side]["strike"]
    )

    for K in candidates:
        if K in traded_strikes:
            continue

        log_m    = np.log(K / spot)
        iv_pchip = float(pchip(log_m)) if pchip is not None else np.nan
        iv_poly  = float(poly(log_m))  if poly  is not None else np.nan

        # Only include this strike if pchip can interpolate it (not NaN).
        # We don't extrapolate pchip into the wings -- if it returns NaN,
        # the strike is outside the range of traded strikes.
        if np.isnan(iv_pchip) or iv_pchip <= 0:
            continue
```

**Step C -- price using pchip IV and append the row:**

```python
        if side == "CALL":
            price = bs_call_price(spot, K, r, iv_pchip, T)
        else:
            price = bs_call_price(spot, K, r, iv_pchip, T) - spot + K * np.exp(-r * T)

        row = meta.to_dict()
        row.update({
            "contract_type":      side,
            "strike":             K,
            "close":              price,
            "iv":                 iv_pchip,
            "iv_pchip":           iv_pchip,
            "iv_poly":            iv_poly,
            "T":                  T,
            "buckets_since_trade": np.nan,
        })
        synthetic_rows.append(row)
```

After the loops, back-fill `iv_pchip` and `iv_poly` onto the real traded rows
(set both equal to the existing `iv` column), concatenate, sort, and return.

**Verify:**

```python
from src.calc.vol_surface import extend_chain

snap_ext = extend_chain(
    snap_iv, smiles,
    core_low=4800,  core_high=5400,  core_step=5,
    tail_low=4000,  tail_high=8000,  tail_step=25,
)
real      = snap_ext["buckets_since_trade"].notna().sum()
synthetic = snap_ext["buckets_since_trade"].isna().sum()
print(f"Real trades: {real},  Synthetic: {synthetic}")

# Compare the two IVs on synthetic rows
syn = snap_ext[snap_ext["buckets_since_trade"].isna()].copy()
syn["iv_diff"] = (syn["iv_poly"] - syn["iv_pchip"]).abs()
print(syn[["contract_type", "strike", "iv_pchip", "iv_poly", "iv_diff"]]
      .sort_values(["contract_type", "strike"]))
```

Large `iv_diff` values mean the two models disagree -- those strikes are in a region
where pchip is near the edge of its fitted range and poly may be capturing different
curvature. Treat Greeks computed there with more skepticism.

Also check the grid density transition worked:

```python
calls_ext = snap_ext[snap_ext["contract_type"] == "CALL"].sort_values("strike")
print(calls_ext[["strike", "iv_pchip", "buckets_since_trade"]].to_string(index=False))
```

You should see 25-pt spacing below 4800, switching to 5-pt above it.

**Verify:**

```python
from src.calc.vol_surface import extend_chain

snap_full = extend_chain(snap_iv, smiles)
real      = snap_full["buckets_since_trade"].notna().sum()
synthetic = snap_full["buckets_since_trade"].isna().sum()
print(f"Real trades: {real}")
print(f"Synthetic (smile-interpolated): {synthetic}")

# Inspect the synthetic rows
syn = snap_full[snap_full["buckets_since_trade"].isna()]
print(syn[["contract_type", "strike", "iv", "close"]].sort_values(["contract_type", "strike"]))
```

You should see a handful of additional strikes filling in the wings, each with a
BS-consistent price derived from the interpolated IV.

---

## Part 7 -- Compute Greeks on the Full Extended Chain

Now apply everything to the extended snapshot.

### Step 7.1 -- Run the pipeline end to end

```python
from datetime import date
import pandas as pd
from src.util.minio import get_duckdb_conn, option_parquet_uris
from src.util.spx_candles import load_spx_1min
from src.calc.vol_surface import compute_iv, fit_vol_smile, extend_chain
from src.calc.greeks import greeks_df

trade_date = date(2024, 3, 1)
con = get_duckdb_conn()
uri = option_parquet_uris("SPXW", trade_date, date(2024, 3, 2), provider="massive")[0]

raw = con.execute(f"""
    SELECT sampled_at, expiration_date, contract_type, strike,
           open, high, low, close, total_volume
    FROM read_parquet('{uri}', union_by_name=true)
    WHERE expiration_date = '{trade_date}' AND close IS NOT NULL
""").fetchdf()

chain = resample_chain(raw)
spx   = load_spx_1min(trade_date, date(2024, 3, 2))
spx_5 = spx["close"].resample("5min").last().ffill().rename("spot")
spx_5.index.name = "sampled_at"
chain = chain.merge(spx_5.reset_index(), on="sampled_at", how="left")

snap     = chain[chain["sampled_at"] == pd.Timestamp("2024-03-01 11:00", tz=ET)].copy()
snap_iv  = compute_iv(snap)
smiles   = fit_vol_smile(snap_iv)
snap_ext = extend_chain(
    snap_iv, smiles,
    core_low=4800, core_high=5400, core_step=5,
    tail_low=4000, tail_high=8000, tail_step=25,
)
result   = greeks_df(snap_ext)

print(result[["strike", "contract_type", "iv", "delta", "gamma", "vega", "theta"]]
      .sort_values(["contract_type", "strike"])
      .to_string(index=False))
```

---

## Why BS Works Here (and Where It Breaks Down)

**Why it works for SPXW:** SPXW options are European cash-settled. They can only be
exercised at expiry, so the BS closed-form is exact under its assumptions.

**Why it fails for American options:** American options (single-name equity options,
SPY) can be exercised early. Deep ITM puts especially have an early exercise premium
that BS cannot price. You need a binomial tree or Monte Carlo for those.

**The vol smile is a symptom, not a problem you solved:** When you back out different
IVs for different strikes, you are using BS as a quoting convention, not a model.
The BS model itself assumes constant vol and cannot generate a smile. The smile exists
because the market prices in tail risk (crash risk in OTM puts) that the GBM
assumption ignores. Using per-contract IV is correct for pricing and Greeks at a
single strike. It does not capture how IV moves as spot moves -- for that you need
vanna and volga (see `src/calc/vanna_charm.py`).

---

## Reference

- `src/calc/greeks.py` -- stubs you implemented: `delta`, `gamma`, `vega`, `theta`, `greeks_df`
- `src/calc/vol_surface.py` -- stubs you implemented: `fit_vol_smile`, `extend_chain`
- `src/util/minio.py` -- data loading
- `src/util/spx_candles.py` -- SPX spot price
- `src/calc/vanna_charm.py` -- reference for 0DTE T calculation and vanna/charm
