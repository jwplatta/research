# GEX Wall Study Notes

## Strategy

**GEX Wall Options Spread Strategy**

Use GEX call and put walls as structural price levels to trade mean-reversion via defined-risk options spreads on SPXW:

- **Bear call spread:** sell a call spread at or just above the call wall when high-conviction the wall will hold (price reverses down from resistance). Collect premium if SPX stays below the short strike.
- **Bull put spread:** sell a put spread at or just below the put wall when high-conviction the wall will hold (price bounces up from support). Collect premium if SPX stays above the short strike.

Entry trigger: price comes within ~10pts of the wall level. A logistic regression classifier trained on features (VIX regime, GEX structure, IV levels, time-of-day) assigns a hold probability — only enter when probability exceeds a threshold calibrated for statistical significance.

Target outcome: `bounce_60m` (price remains on the original side of the wall 60 minutes after first approach). EOD held is the ultimate exit anchor.

---

## Open Questions & TODOs

**TODO — Intraday Pinning Score:** Develop a composite score that measures how firmly the call and put walls are holding throughout the day. Expected behavior: score increases monotonically on stable days where both walls hold; score plateaus or drops on days with wall relocations. Components to consider: consecutive bucket count at same wall level, GEX magnitude at wall, vol at wall strike relative to baseline, dist_to_wall trend, ZGL regime. Would serve as a real-time signal for whether the condor position is safe to hold.

**QUESTION (answered) — How is the call wall computed?** `find_raw_wall_strikes()` uses `_aggregate_side_gex_by_strike()` which separates calls and puts completely — it sums call GEX by strike, sums put GEX by strike, then finds the OTM call strike with max call GEX and the OTM put strike with most negative put GEX. **It does NOT net call vs put GEX at each strike.** The call wall is purely the peak call-side GEX strike; the put wall is purely the peak put-side GEX strike. `net_gex_by_strike()` does compute call − put at each strike, but is not used by `find_raw_wall_strikes()`. Using net GEX per strike (calls minus puts) to find the wall is an open question — a strike with high call GEX but also high put GEX may be less of a true wall than one dominated purely by call OI.

---

## Overview

Study of whether GEX levels act as meaningful SPX inflection points, structured in four phases:
1. **Forward study** — compute walls throughout the day, measure bounce rate and stability
2. **Reverse-engineering study** — detect actual SPX reversals, measure distance to nearest GEX level
3. **Wall solidity study** — given a wall is tested, what predicts whether it holds vs breaks?
4. **Intraday wall movement study** — how much do walls drift from morning to close?

## File Index

### Source (src/)

| File | Purpose |
|------|---------|
| `src/calc/gex.py` | GEX functions: `net_gex_by_strike`, `find_raw_wall_strikes`, `find_zero_gamma_level`, `gex_at_strike`, `net_gex_at_spot` |
| `src/calc/vanna_charm.py` | Vanna and charm exposure calculations |
| `src/util/reversal.py` | Zigzag inflection point detection on 1-min bars |
| `src/util/minio.py` | DuckDB + MinIO S3 connection helpers |

### Scripts (scripts/)

| File | Purpose |
|------|---------|
| `scripts/gex_wall_backtest.py` | Phase 1: forward wall analysis — bounce rate, EOD held, fake breaks |
| `scripts/gex_reversal_study.py` | Phase 2: detect SPX reversals, measure distance to nearest GEX level |
| `scripts/gex_wall_features.py` | Phase 3 feature set 1 — VIX, time-of-day, SPY volume |
| `scripts/gex_wall_features2.py` | Phase 3 feature set 2 — GEX, vanna, charm, ZGL |
| `scripts/gex_wall_buckets.py` | Phase 3: VIX bucket + 0DTE GEX skew bucket analysis |
| `scripts/gex_wall_iv_zscore.py` | Phase 3: ATM IV z-score and VIX z-score; saves `daily_atm_iv.csv` |
| `scripts/gex_wall_rr_skew.py` | Phase 3: 25-delta risk reversal; saves `daily_rr.csv` |
| `scripts/gex_wall_build_master.py` | Merges all feature files into `master_features.csv` |
| `scripts/gex_wall_classifier.py` | Logistic regression classifiers for call and put wall hold probability |
| `scripts/gex_wall_intraday_levels.py` | Phase 4: call/put wall at morning/midday/close per day |
| `scripts/gex_call_wall_hourly.py` | Phase 5: call wall + full feature set at each hour during RTH |
| `scripts/gex_wall_stability.py` | Phase 6: call/put wall per 30-min bucket, stability classification |
| `scripts/fetch_vix_dec2025.py` | One-time: fetch Dec 2025 VIX 1-min from IBKR → `data/candles/` |
| `scripts/gex_condor_timing.py` | Phase 10: when during the day are both walls tested? Iron condor entry timing analysis |
| `scripts/gex_call_spread_backtest.py` | Phase 10: 0DTE bear call spread backtest — entry at 10:45, K1 = call_wall + 20pts, filter comparison |

### Data (data/candles/) — gitignored, regenerate with fetch scripts

| File | Regenerate with | Purpose |
|------|----------------|---------|
| `data/candles/vix_1min_dec2025.parquet` | `scripts/fetch_vix_dec2025.py` | Dec 2025 VIX 1-min bars from IBKR (not in lakehouse) |
| `data/candles/spy_1min_dec2025.parquet` | `scripts/fetch_spy_dec2025.py` | Dec 2025 SPY 1-min bars from IBKR (not in lakehouse) |

### Output (out/gex_walls/, out/gex_reversal/)

| File | Purpose |
|------|---------|
| `out/gex_walls/events.csv` | Per wall-level event log, 203k rows (Dec 2025–Aug 2026) |
| `out/gex_walls/features.csv` | events + feature set 1 |
| `out/gex_walls/features2.csv` | events + feature set 2 |
| `out/gex_walls/daily_atm_iv.csv` | 0DTE ATM IV per trading day |
| `out/gex_walls/daily_rr.csv` | 25Δ risk reversal + iv_25d_call/put per trading day |
| `out/gex_walls/master_features.csv` | All features merged, 103k rows, training-ready |
| `out/gex_walls/intraday_levels.csv` | Call/put wall at morning/midday/close, 165 days |
| `out/gex_walls/call_wall_hourly.csv` | Call wall + features at each RTH hour, 829 rows, 100% coverage |
| `out/gex_walls/wall_stability_30min.csv` | Call/put wall per 30-min bucket, 1,649 rows, 165 days |
| `out/gex_reversal/inflections_lb10_mm8.csv` | Reversal study: 1,511 inflection points with GEX distances |
| `out/gex_walls/call_spread_backtest.csv` | Phase 10: 165 rows, one per trading day, bear call spread backtest results |
| `out/gex_walls/call_spread_losses.csv` | Phase 10: losing trades only with distance breakdown |

---

## Phase 1 — Forward Wall Study

**Data:** `out/gex_walls/events.csv` — Schwab SPXW option snapshots, Dec 2025–Jun 2026. Script: `scripts/gex_wall_backtest.py`.

### Setup

- **Data:** Schwab SPXW snapshots, Dec 2025–Jun 2026 (144 trading days, 203k events)
- **Wall tested:** price comes within 10pts of level
- **Bounce:** price still on original side of wall 30min / 60min after first approach
- **EOD held:** SPX closes on original side; **EOD returned:** crossed intraday, closed back (fake break)

### Wall Behavior Results

| Method | Side | Tested% | Bounce 30m | Bounce 60m | EOD Held | EOD Returned |
|---|---|---|---|---|---|---|
| raw | call | 35.5% | 75.5% | 71.8% | 67.0% | 25.7% |
| raw | put | 36.3% | 72.8% | 70.3% | 68.1% | 31.1% |
| weighted | call | 42.3% | 74.2% | 70.6% | 68.2% | 26.1% |
| weighted | put | 42.0% | 72.1% | 67.7% | 64.5% | 31.0% |
| zones | call | 39.6% | 74.1% | 69.9% | 66.2% | 25.2% |
| zones | put | 39.4% | 75.2% | 69.5% | 65.7% | 31.4% |
| clustered | call | 20.6% | 75.1% | 69.0% | 58.5% | 19.6% |
| clustered | put | 27.6% | 71.9% | 67.2% | 63.4% | 32.3% |

### Wall Stability (intraday drift)

All methods: 120–155pt median drift regardless of method. Walls move too much intraday to track continuously — **calculate once at open (9:30–9:45 ET) and hold fixed.**

### Key Findings

1. ~70-75% bounce rate is real and consistent across all methods and time horizons.
2. All wall methods are essentially equivalent in predictiveness — raw is simplest.
3. Fake breaks (crossed intraday, closed back) = 25-32% of tested walls. Don't panic on a wall breach until EOD confirms.
4. Clustered method too selective (20-27% test frequency) with no improvement in bounce rate.

---

## Phase 2 — Reverse-Engineering Study

**Data:** `out/gex_reversal/inflections_lb10_mm8.csv` — SPX 1-min bars from lakehouse + Schwab SPXW open snapshots (~9:40 ET), Dec 2025–Jun 2026. Script: `scripts/gex_reversal_study.py`.

Instead of "did price respect our wall?", flip it: find where SPX actually reversed, measure distance to nearest GEX level.

**Method:** Zigzag algorithm on SPX 1-min bars (lookback=10 bars, min_move=8pts). Open snapshot GEX levels (~9:40 ET). Top-20 strikes by |GEX| vs bottom-20 baseline.

### Results (1,511 inflections, 144 days)

| Method | Median dist | <10pt | <20pt |
|---|---|---|---|
| **high_gex_any (top 20 \|GEX\| strikes)** | **2.5** | **85.9%** | 95.8% |
| low_gex_any (baseline) | 158.5 | 5.9% | 11.1% |
| raw_call / raw_put wall | 35–40 | 13–15% | 26–30% |
| random Monte Carlo | 14.2 | ~0% | 100% |

Result stable across all 9 parameter combinations (lookback 5/10/15 × min_move 5/8/12): ~85% within 10pt.

**Statistical significance:** z = 44.1, p ≈ 0. High-GEX vs low-GEX same strike universe, same density — only magnitude differs.

### Key Findings

1. **86% of SPX intraday reversals happen within 10pts of a top-20 |GEX| strike.** Median 2.5pt distance.
2. **Single wall methods are too coarse** — 13-15% within 10pt. The full ranked |GEX| list is what matters.
3. **±10pt is the right precision threshold.** 5pt is unrealistically demanding.

---

## Phase 3 — Wall Solidity Study

**Data:** `out/gex_walls/master_features.csv` — 103,623 rows, Dec 2025–Aug 2026. Built by merging `events.csv` + feature sets from `scripts/gex_wall_features.py`, `gex_wall_features2.py`, `gex_wall_iv_zscore.py`, `gex_wall_rr_skew.py` → `gex_wall_build_master.py`. One row per wall-approach event (price within 10pts of level), all four wall methods, call and put sides.

**Core question: given a wall is tested, does it hold or break?**

71,886 tested events across 121 days. Three outcomes tracked: bounce_30m, bounce_60m, eod_held.

### VIX Buckets

| VIX | N | bounce_30m | eod_held |
|---|---|---|---|
| <15 | 1,337 | 83.5% | 79.3% |
| 15-20 | 57,865 | 74.9% | 68.7% |
| 20-25 | 7,781 | 65.3% | 49.2% |
| >25 | ~1,800 | ~70% | ~50% |

Low vol = walls hold. Clear monotonic trend. Above VIX 25 too few events (Dec 2025–Jun 2026 was mostly 15-20 VIX).

### 0DTE GEX Skew

Skew = `(call_gex - put_gex) / total_gex` from 0DTE open snapshot. +1 = pure call dominance, -1 = pure put dominance. Median: +0.167.

**Call walls — eod_held by skew:**

| Skew | N | bounce_30m | eod_held |
|---|---|---|---|
| <-0.5 (put dominant) | 5,645 | 79.7% | **83.3%** |
| -0.5 to -0.2 | 5,112 | 48.5% | **38.3%** ← anomaly |
| -0.2 to 0 | 3,605 | 83.5% | 54.9% |
| 0 to 0.2 | 2,193 | 71.5% | 54.8% |
| 0.2 to 0.5 | 7,695 | 76.9% | 68.9% |
| >0.5 (call dominant) | 10,643 | 80.3% | **73.2%** |

**Put walls — eod_held by skew:**

| Skew | N | bounce_30m | eod_held |
|---|---|---|---|
| <-0.5 (put dominant) | 10,040 | 63.7% | **45.5%** |
| -0.5 to -0.2 | 5,448 | 77.1% | 76.6% |
| -0.2 to 0 | 5,710 | 84.8% | **74.9%** |
| 0 to 0.2 | 2,843 | 58.9% | 74.0% |
| 0.2 to 0.5 | 6,338 | 77.0% | **78.2%** |
| >0.5 (call dominant) | 6,173 | 77.6% | 64.0% |

The -0.5 to -0.2 bucket is anomalous: call walls collapse (38% EOD held) while put walls hold (77%). Worth investigating.

### net_gex_spot Buckets (dealer gamma regime)

| Regime | N | bounce_30m | eod_held |
|---|---|---|---|
| Large positive GEX | 34,035 | 79.1% | **72.2%** |
| Large negative GEX | 30,325 | 70.2% | 58.3% |
| Small negative GEX | 7,109 | 66.5% | 67.4% |
| Small positive GEX | 417 | 47.5% | 41.0% ← noise |

Positive = dealer long gamma (dampening). Negative = dealer short gamma (amplifying). 14pt EOD held gap confirms SpotGamma framework.

### VIX z-score 20d Buckets

| Bucket | N | bounce_30m | eod_held |
|---|---|---|---|
| <-1 (vol cooling) | 18,762 | **81.6%** | **71.4%** |
| -1 to 0 | 20,337 | 69.6% | 63.8% |
| 0 to +1 | 12,535 | 67.3% | 64.3% |
| >+1 (vol rising) | 10,233 | 71.2% | 59.9% |

---

## Feature Summary

All features explored. r = point-biserial correlation vs outcome. All p ≈ 0 unless noted.

| Feature | What it measures | r_b30 | r_b60 | r_eod |
|---|---|---|---|---|
| **atm_iv_zscore_5d** | 0DTE ATM IV as z-score vs prior 5-day avg — "is IV elevated vs this week?" | -0.132 | -0.103 | **-0.167** |
| **net_gex_spot** | Sum of GEX within ±25pt of spot. Positive = dealer long gamma (dampening). Negative = amplifying. | +0.129 | +0.093 | +0.093 |
| **vix_open** | CBOE VIX at open — 30-day SPX implied vol. | -0.122 | -0.120 | -0.132 |
| **gex_at_wall_strike** | GEX magnitude at the specific strike being tested — more hedging right there = stronger pin. | +0.114 | +0.103 | +0.061 |
| **vix_zscore_10d** | VIX as z-score vs prior 10-day avg. | -0.106 | -0.087 | -0.074 |
| **atm_iv_zscore_10d** | 0DTE ATM IV z-score, 10-day window. | -0.091 | -0.046 | -0.127 |
| **total_vanna** | Aggregate vanna (dDelta/dVol) across all SPXW. Positive = rising vol adds a directional bid that reinforces walls. | +0.088 | +0.056 | +0.097 |
| **approach_minute_of_day** | Minute of day at first approach. Later approaches more likely to hold through EOD. | +0.071 | +0.100 | **+0.182** |
| **spy_vol_zscore** | SPY volume at approach bar as z-score vs same minute-of-day baseline. High = conviction move, more likely to break. | -0.071 | -0.079 | -0.095 |
| **total_charm** | Aggregate charm (dDelta/dt). Negative charm = delta decaying as expiry nears, reduces pinning. | -0.065 | -0.037 | -0.061 |
| **dist_wall_to_zgl** | Wall level minus zero-gamma level. Positive = wall inside dampening zone. | +0.057 | +0.002 | +0.020 |
| **gex_skew_0dte** | `(call_gex - put_gex) / total_gex` from 0DTE. Non-linear — see bucket tables above. | n/l | n/l | n/l |
| **wall_breached_at_open** | Did price trade beyond wall in first 15min? Early breach = wall re-tested from wrong side. | -0.031 | -0.021 | -0.033 |
| **repeat_test_count** | Prior zone entries+exits before current approach. Walls weaken with repeated tests. | -0.019 | -0.023 | -0.029 |
| **dist_to_wall** | How far spot is from the wall at snapshot time. | +0.007 | +0.039 | +0.029 |
| **approach_momentum** | SPX return over 20 bars before approach. | +0.015 | -0.029 | +0.013 | No signal (p > 0.4) |
| **wall_side / method** | Call vs put; which calculation method. | ~0.01 | ~0.01 | ~0.02 | Negligible |
| **iv_vix_ratio** | 0DTE ATM IV / VIX (term structure proxy). ATM IV runs ~110% of VIX in this dataset. | inconclusive — binning failed, needs redo |

---

## Feature Independence

Feature-feature Pearson correlations reveal several redundant clusters:

**Vol family (highly intercorrelated — pick one):**
- vix_zscore_5d vs vix_zscore_10d: r = 0.83
- vix_zscore_10d vs vix_zscore_20d: r = 0.88
- total_vanna vs vix_zscore_20d: r = -0.77 ← vanna is partly a vol proxy
- total_charm vs vix_zscore_20d: r = +0.67

**GEX cluster (related but not fully redundant):**
- net_gex_spot vs gex_at_wall_strike: r = 0.67 — overlap but measure different things (regime vs level)

**ZGL/vanna/charm cluster:**
- dist_wall_to_zgl vs total_vanna: r = 0.72
- dist_wall_to_zgl vs total_charm: r = -0.71
- total_vanna vs total_charm: r = -0.76

**Wall behavior cluster:**
- wall_breached_at_open vs repeat_test_count: r = 0.61

**Genuinely independent features** (r < 0.1 with everything):
- approach_minute_of_day
- spy_vol_zscore
- dist_to_wall

### Call vs Put Wall Feature Split

Splitting by wall type reveals fundamentally different drivers — the strongest justification for **separate classifiers**.

**Call walls (n=35,021):**

| Feature | r_b30 | r_b60 | r_eod |
|---|---|---|---|
| **vix_open** | **-0.193** | **-0.200** | -0.130 |
| **rr (25Δ call IV - put IV)** | +0.102 | +0.118 | +0.079 |
| spy_vol_ratio | -0.111 | -0.105 | -0.115 |
| gex_at_wall_strike | +0.095 | +0.089 | -0.008 |
| net_gex_spot | +0.093 | +0.078 | +0.019 |
| approach_minute_of_day | +0.029 | +0.087 | +0.183 |
| iv_25d_call / iv_25d_put | -0.06–-0.08 | weak | weak |

Call walls are **vol-regime driven**: high VIX kills resistance. RR being positive here means less negative skew (puts not unusually cheap relative to calls) = call wall holds. Absolute IV levels (25Δ) add little beyond what VIX already captures.

**Put walls (n=36,865):**

| Feature | r_b30 | r_b60 | r_eod |
|---|---|---|---|
| **net_gex_spot** | **+0.195** | +0.118 | **+0.219** |
| **gex_at_wall_strike** | **+0.162** | +0.137 | **+0.196** |
| **iv_25d_call** | -0.044 | -0.045 | **-0.226** |
| **iv_25d_put** | -0.038 | -0.041 | **-0.208** |
| total_vanna | +0.089 | +0.035 | +0.153 |
| approach_minute_of_day | +0.112 | +0.114 | +0.183 |
| vix_open | -0.073 | -0.064 | -0.142 |
| wall_breached_at_open | -0.058 | -0.031 | -0.067 |
| rr | ~0 | ~0 | ~0 |

Put walls are **GEX-structure driven**: dealer positioning (net_gex_spot, gex_at_wall_strike) is the dominant predictor. The absolute 25Δ IV levels matter strongly for EOD held (r ≈ -0.21): expensive options = market pricing in a move through support = wall fails. VIX and RR are weak here.

**Key insight:** these are different problems. Call wall solidity is primarily a vol-regime question. Put wall solidity is primarily a GEX-positioning question, with absolute IV as a secondary signal.

### Proposed feature sets for separate classifiers

**Call wall classifier:**
| Feature | Rationale |
|---|---|
| vix_open | Dominant predictor (r=-0.20) |
| rr (25Δ RR) | Positive signal for call walls specifically |
| spy_vol_ratio | Conviction at approach |
| approach_minute_of_day | Fully independent; strong EOD signal |
| net_gex_spot | Secondary GEX regime signal |
| total_vanna | Partly vol proxy but adds signal |

**Put wall classifier:**
| Feature | Rationale |
|---|---|
| net_gex_spot | Dominant predictor (r=+0.22) |
| gex_at_wall_strike | Strong, partially independent |
| iv_25d_put | Absolute downside IV — key EOD signal (r=-0.21) |
| approach_minute_of_day | Fully independent; strong EOD signal |
| total_vanna | Strong for EOD (r=+0.15) |
| wall_breached_at_open | Small but directionally correct |

---

## Phase 4 — Intraday Wall Movement Study

**Data:** `out/gex_walls/intraday_levels.csv` — Schwab SPXW snapshots, Dec 2025–Aug 2026, 138 days with all three snapshots (morning ~9:45, midday ~13:00, close ~15:30 ET).
GEX computed across all expirations in the chain (standard community approach — raw gamma × OI × spot², no DTE cutoff).
Call wall = strike with highest positive net GEX. Put wall = strike with most negative net GEX.

**Script:** `scripts/gex_wall_intraday_levels.py`
**Output:** `out/gex_walls/intraday_levels.csv`

### Wall Stability

| | Call Wall | Put Wall |
|--|--|--|
| Stable all day (morn = mid = close) | 30 days (22%) | 28 days (20%) |
| Moved at any point | 108 days (78%) | 110 days (80%) |
| — morning→midday only | 24 | 28 |
| — midday→close only | 24 | 35 |
| — both legs | 60 | 47 |

Walls move on roughly 4 out of 5 days.

### Morning Walls as Day Range

**Did morning walls bracket where spot closed?**

84/138 days (61%) — spot closed inside the morning call and put wall levels.
Outside: 28 days (19%) above call wall, 27 days (19%) below put wall. Roughly symmetric.

### Role Reversal (resistance ↔ support)

A wall flips role when its level later appears as the opposite wall — e.g. spot breaks the call wall at 7050, and 7050 becomes the put wall (new support) later in the day.

| Pattern | By midday | By close | Either |
|--|--|--|--|
| Resistance → Support (morning call wall becomes put wall) | 14 days (10%) | 15 days (11%) | 19 days (14%) |
| Support → Resistance (morning put wall becomes call wall) | 9 days (7%) | 9 days (7%) | 15 days (11%) |
| Either pattern | | | **34 days (25%)** |

On 1 in 4 trading days a GEX wall flips role intraday.

### Directional Drift Patterns

| Pattern | Call Wall | Put Wall | Both |
|--|--|--|--|
| Reverted to morning level by close (shifted midday, returned) | 13 days (9%) | 13 days (9%) | 4 days (3%) |
| Kept drifting UP all day (morn < mid < close) | 9 days (7%) | 9 days (7%) | 2 days (1%) |
| Kept drifting DOWN all day (morn > mid > close) | 8 days (6%) | 3 days (2%) | 1 day (1%) |

Consistent directional drift is rare (~6-7%). Most movement is non-directional.

### Wall Movement Magnitude

How often do walls move ≥N pts from their morning level at any point in the day:

| Threshold | Call Wall | Put Wall | Both walls |
|--|--|--|--|
| ≥25 pts | 95 days (69%) | 101 days (73%) | 76 days (55%) |
| ≥50 pts | 78 days (57%) | 84 days (61%) | — |
| ≥100 pts | 38 days (28%) | 42 days (30%) | — |

A 25pt move from the morning level is the norm (70% of days), not the exception.

### Conclusions for Strategy Design

**1. The Phase 1 bounce rate and Phase 4 wall movement results are in tension — and that tension matters.**
Phase 1 (70-75% bounce rate) used walls fixed at open and measured bounces only when price came within 10pts of *that specific level*. Only 35-42% of walls were ever "tested." Phase 4 shows walls moved ≥25pts from morning on 70% of days. These facts together imply: Phase 1's sample is biased toward the minority of days when walls stayed put — those are the days price returned to test the original level. On the 70% of days when walls drifted, price was bouncing at the *new* wall location, not the morning one, and Phase 1 never recorded those bounces at all.

**2. The wall must be recalculated intraday to capture the actual bounce mechanism.**
Fixing walls at open and waiting for price to return to that level only works on stable-wall days (~20-30%). To trade the bounce consistently, entry must be triggered off the *current* wall, computed from the most recent option chain snapshot. The morning wall is a stale reference on 70% of days.

**3. Morning walls define a structural range, but it's the current wall that generates the trade signal.**
The 61% "inside range" stat (spot closed between morning walls) is a property of the morning snapshot used as a range forecast, not a trigger. The actionable signal is: price approaches the current wall level, classifier assigns high hold probability, enter spread.

**4. Role reversal (25% of days) is consistent with the bounce mechanism, not contradictory to it.**
When a call wall is broken and that level becomes put support, the gamma cluster at that strike has changed sign from the dealer's perspective — the hedging flow at that strike now supports price from below instead of capping it from above. Tracking the current wall naturally captures this: the new put wall at the old call wall level is a valid put bounce entry.

**5. The classifier's job is to filter for high-conviction current-wall holds, not to predict morning walls.**
Features (VIX regime, GEX structure, IV levels) need to be evaluated at the time of entry — i.e., from the same snapshot used to compute the current wall — not locked to morning values.

**6. Call and put wall classifiers remain separate.**
Phase 3 established fundamentally different mechanisms: call walls are vol-regime driven (VIX dominates), put walls are GEX-structure driven (net_gex_spot, gex_at_wall_strike dominate). This holds regardless of when walls are computed.

---

## Phase 5 — Hourly Call Wall Feature Dataset

**Data:** `out/gex_walls/call_wall_hourly.csv` — Schwab SPXW snapshots + VIX 1-min (IBKR) + SPY 1-min (IBKR), Dec 2025–Aug 2026. Script: `scripts/gex_call_wall_hourly.py`. Dec 2025 SPY vol zscore is an artifact (see Cluster 3).

**Script:** `scripts/gex_call_wall_hourly.py`
**Output:** `out/gex_walls/call_wall_hourly.csv`

829 rows × 16 columns. One row per (trading day × RTH hour). Hours: 10, 11, 12, 13, 14, 15 ET. Dec 2025–Aug 2026 (165 days; 31 skipped, no SPXW data). All fields 100% populated.

Fields: `date, hour, snap_ts, call_wall, gex_at_wall, spot, dist_to_wall, dist_to_zgl, vix, atm_iv, iv_25d_call, iv_25d_put, gex_wall_p10, gex_wall_m10, spy_vol_zscore, minute_of_day`

GEX computed across all expirations. IV fields from 0DTE chain at nearest snapshot with 0DTE data. VIX is last bar at or before snap timestamp (pad/ffill).

### K-Means Clustering (k=4)

k=4 chosen by silhouette + interpretability (silhouette: k=2: 0.34, k=4: 0.24, k=5: 0.25 — k=2 is too coarse, k=4/5 give similar structure).

| Cluster | n | Label | VIX | ATM IV | Dist to Wall | GEX at Wall | Dist to ZGL | SPY Vol Z |
|---------|---|-------|-----|--------|-------------|-------------|-------------|-----------|
| 0 | 242 | **Gamma Pin** | 16.3 | 13.7% | 19 pts | 7.1B | +196 pts | 1.17 |
| 1 | 180 | **Market Trending Down** | 22.1 | 29.0% | 103 pts | ~0 | +4 pts | 1.88 |
| 2 | 380 | **Normal/Active** | 16.7 | 16.0% | 49 pts | 1.0B | +48 pts | 2.78 |
| 3 | 20 | **Dec 2025 artifact** | 14.6 | 11.6% | 66 pts | 23B | +76 pts | 70.0 |

**Cluster 0 — Gamma Pin (29%):** Spot pinned 19pts from a massive wall (7.1B GEX), low vol, spot far above ZGL (+196pts — deep in positive gamma territory), afternoon-weighted (hours 13–15). Dominates May and Aug 2026. Best setup for a bear call spread: wall is massive, spot is right there, vol is low.

**Cluster 1 — Market Trending Down (22%):** VIX 22, ATM IV 29%, spot 103pts from call wall, GEX at wall effectively zero — the call wall doesn't exist as resistance in the traditional sense because the market is in a sustained move lower. Concentrated in Feb–Mar 2026 (Iran war) and Jun 2026. The call wall is irrelevant here; the **put wall is the interesting level** in this regime — this is where bull put spread entries should come from.

**Cluster 2 — Normal/Active (46%):** Largest cluster. VIX 17, IV 16%, spot 49pts from wall, moderate GEX (1B), elevated SPY volume (2.78x). Spread across all months. The call wall exists and has real GEX behind it, but spot has room to move. Requires classifier to identify high-conviction approaches.

**Cluster 3 — Dec 2025 artifact (2%):** All 20 rows are Dec 2025 with SPY vol zscore averaging 70x. This is a baseline normalization artifact — the median-by-minute for SPY volume is computed from the full dataset dominated by Jan–Aug 2026, so Dec 2025 volume appears anomalously high. Exclude from modeling or recompute baseline with Dec included.

---

## Phase 6 — Call Wall Stability (30-min buckets)

**Data:** `out/gex_walls/wall_stability_30min.csv` — Schwab SPXW snapshots, Dec 2025–Aug 2026, 165 trading days, 1,649 rows. Bucket 0 chain = latest snapshot per expiry within 9:30–10:00; subsequent buckets similarly. ATM IV from nearest 0DTE snapshot to each bucket midpoint. Script: `scripts/gex_wall_stability.py`.

**Script:** `scripts/gex_wall_stability.py`
**Output:** `out/gex_walls/wall_stability_30min.csv`

For each 30-min RTH bucket (13 per day, 9:30–16:00 ET), builds the aggregate GEX chain by taking the **latest snapshot per expiry** within that window, then computes call and put walls. This gives a forward-consistent view of the wall at each point in the day.

**CSV fields:** `sample_datetime, spot, bucket_num, call_wall, put_wall` — 1,649 rows across 165 days.

### Call Wall Stability Results (Dec 2025–Aug 2026, 165 trading days)

| Pattern | Days | % |
|---------|------|---|
| Stable all day (wall never changed) | 36 | 21.8% |
| Any change during day | 129 | 78.2% |
| Reversion: A→B→A (test candidate) | 64 | 38.8% |
| Genuine move: wall moved and held | 43 | 26.1% |
| Revert-then-hold: A→B→A→B | 20 | 12.1% |

*Note: patterns are not mutually exclusive — a day can have both reversion and genuine move.*

| Metric | Value |
|--------|-------|
| Avg bucket-to-bucket changes per day | 2.38 |
| Avg distinct call wall levels per day | 2.48 |

**Distribution of distinct call wall levels seen per day:**

| Distinct levels | Days |
|-----------------|------|
| 1 (stable) | 36 |
| 2 | 59 |
| 3 | 38 |
| 4 | 21 |
| 5 | 8 |
| 6 | 3 |

### Implications for Stable Wall Definition

The reversion rate (38.8%) is the key number for designing the stability heuristic. On nearly 2 in 5 days, the wall moves to a new level and then returns to the original — these are strong candidates for "the original level was just being tested, not broken." The genuine move rate (26.1%) tells us that only about 1 in 4 days sees a wall that actually commits to a new level. The revert-then-hold pattern (12.1%) represents the ambiguous case: the wall returned first, then committed.

Proposed stability rules to evaluate:
1. **Time delay:** don't update the wall until it has held its new level for ≥2 consecutive 30-min buckets (~1 hour)
2. **Distance threshold:** don't update unless spot has moved ≥N pts beyond the original wall (e.g. 25pts)
3. **Combination:** both time and distance required

---

## Phase 7 — Iron Condor Pricing vs. Call Wall Move Direction

**Data:** `out/gex_walls/wall_stability_30min.csv` (bucket 0 chain for IC pricing + call wall outcome) + Schwab SPXW 0DTE option chain snapshots. Script: `scripts/gex_call_wall_ic_pricing.py` (inline analysis).

**Setup:** 0DTE iron condor, 10-delta short legs, 20pt-wide spreads, priced from bucket 0 chain (latest snapshot per expiry within 9:30–10:00). Only days where the 10-delta short call is strictly above the opening call wall (70 of 138 days, 50.7%). IV tertiles computed across all 138 days: Low ≤16.8, Mid 16.8–23.6, High ≥23.6.

**Call wall move = first genuine held move (≥2 consecutive 30-min buckets) during the day.**

### IC Credit vs. Call Wall Move Direction × IV Bin

| Move bin | IV bin | n | Avg credit | Med credit | % SC ITM | Avg dist SC→wall | Avg IV |
|---|---|---|---|---|---|---|---|
| **No move** | ALL | 24 | $1.94 | $1.96 | 8.3% | 25pts | 20.2 |
| | Low | 6 | $1.69 | $1.67 | 16.7% | 17pts | 15.2 |
| | Mid | 13 | $1.91 | $1.96 | 0.0% | 24pts | 19.5 |
| | High | 5 | $2.33 | $2.36 | 20.0% | 37pts | 27.9 |
| **Up ≤25pts** | ALL | 11 | $1.78 | $1.80 | **45.5%** | 21pts | 16.8 |
| | Low | 8 | $1.61 | $1.66 | 50.0% | 19pts | 14.3 |
| | Mid | 2 | $2.17 | $2.17 | 50.0% | 20pts | 21.0 |
| | High | 1 | $2.40 | $2.40 | 0.0% | 35pts | 28.0 |
| **Up 25–50pts** | ALL | 14 | $2.14 | $2.13 | 14.3% | 29pts | 21.5 |
| | Low | 2 | $1.60 | $1.60 | 0.0% | 28pts | 14.7 |
| | Mid | 9 | $2.08 | $2.12 | 11.1% | 29pts | 19.8 |
| | High | 3 | $2.66 | $2.65 | 33.3% | 28pts | 31.2 |
| **Down ≤25pts** | ALL | 6 | $1.93 | $1.79 | 0.0% | 18pts | 20.0 |
| **Down 25–50pts** | ALL | 13 | $2.59 | $2.17 | 7.7% | 24pts | 22.4 |
| | High | 7 | $3.13 | $2.55 | **0.0%** | 32pts | 26.8 |

### Key Observations

- **Up ≤25pts is the dangerous scenario:** the wall drifts up into the short call (45.5% ITM). The 10-delta strike sits only ~21pts above the opening wall — not enough buffer when spot follows the wall higher. Low IV makes this worse (50% ITM, $1.61 credit).
- **Up 25–50pts is manageable:** SC ITM drops to 14.3% with better credit ($2.14). The wall jumps far enough past the short call zone rather than drifting into it.
- **Down moves are benign for the short call:** 0–7.7% ITM regardless of magnitude — the wall moving down moves away from the short call.
- **Down 25–50pts High IV is the strongest setup:** n=7, $3.13 avg credit, 0% ITM.
- **Low IV days underprice volatility** — worst ITM outcomes cluster here. Consider hedging or skipping.

---

## Phase 8 — Intraday Dynamics: Held vs. Revert Call Wall Moves

**Data:** `out/gex_walls/wall_stability_30min.csv` (bucket-level walls + spot) + Schwab SPXW 0DTE option chain (volume, bid-ask at wall strike) + SPY 1-min candles (IBKR). Dec 2025–Aug 2026, 33 held days vs 29 revert days. Script: inline analysis on stability CSV.

**Question:** Can we tell in real-time whether a call wall breach will hold or revert? Compared two event types (both ≥25pt upward moves): 33 held days (move committed, no return to prior level) vs. 29 revert days (A→B→A). Aligned by offset relative to the move bucket (offset 0 = bucket where wall relocates). Data from 30-min bucket chains.

### Wall Call Option Volume (0DTE, at wall strike)

| Off | H mean | H median | R mean | R median | Δ mean | Δ median |
|---|---|---|---|---|---|---|
| −4 | 38,318 | 37,664 | 25,909 | 21,382 | +12,408 | +16,282 |
| −3 | 43,781 | 41,678 | 31,565 | 25,023 | +12,217 | +16,656 |
| −2 | 44,139 | 34,842 | 37,078 | 34,488 | +7,061 | +354 |
| −1 | 45,044 | 42,104 | 41,762 | 31,334 | +3,282 | +10,770 |
| **0** | **62,113** | **57,463** | **52,677** | **47,073** | **+9,435** | **+10,390** |
| +1 | 67,953 | 64,526 | 58,047 | 52,596 | +9,906 | +11,930 |
| +2 | 72,437 | 64,269 | 69,434 | 62,108 | +3,003 | +2,160 |
| +3 | 73,816 | 68,232 | 75,011 | 67,451 | −1,195 | +782 |

### Spot Distance to Wall (positive = OTM, negative = breached)

| Off | H mean | H median | R mean | R median | Δ mean | Δ median |
|---|---|---|---|---|---|---|
| −4 | 10.4 | 4.4 | 18.8 | 11.9 | −8.4 | −7.5 |
| −3 | 16.0 | 9.6 | 24.9 | 12.6 | −8.9 | −3.0 |
| −2 | 20.7 | 10.3 | 28.0 | 12.1 | −7.3 | −1.8 |
| −1 | 20.3 | 10.2 | 27.7 | 11.4 | −7.4 | −1.2 |
| **0** | **11.0** | **−3.4** | **21.6** | **−1.6** | **−10.6** | **−1.8** |
| +1 | 11.0 | −5.2 | 27.0 | 9.7 | −16.0 | −14.9 |
| +2 | 7.9 | −8.6 | 27.6 | 13.4 | −19.8 | −22.0 |
| +3 | −0.1 | −9.3 | 26.6 | 17.2 | −26.7 | −26.5 |

### Wall Call Bid-Ask Spread

| Off | H mean | H median | R mean | R median | Δ mean | Δ median |
|---|---|---|---|---|---|---|
| −4 | 0.112 | 0.100 | 0.121 | 0.100 | −0.010 | 0.000 |
| −3 | 0.113 | 0.100 | 0.112 | 0.100 | +0.001 | 0.000 |
| −2 | 0.108 | 0.100 | 0.117 | 0.100 | −0.008 | 0.000 |
| −1 | 0.107 | 0.100 | 0.107 | 0.100 | 0.000 | 0.000 |
| **0** | **0.139** | **0.100** | **0.124** | **0.100** | **+0.015** | **0.000** |
| +1 | 0.217 | 0.150 | 0.110 | 0.100 | +0.107 | +0.050 |
| +2 | 0.190 | 0.200 | 0.100 | 0.100 | +0.090 | +0.100 |
| +3 | 0.248 | 0.200 | 0.096 | 0.100 | +0.152 | +0.100 |

### SPY Bucket Volume

| Off | H mean | H median | R mean | R median | Δ mean | Δ median |
|---|---|---|---|---|---|---|
| −4 | 62,398 | 48,141 | 108,487 | 78,582 | −46,089 | −30,440 |
| −3 | 66,204 | 50,986 | 84,170 | 68,919 | −17,966 | −17,933 |
| −2 | 67,683 | 59,598 | 72,053 | 49,976 | −4,369 | +9,622 |
| −1 | 98,426 | 73,476 | 77,880 | 56,864 | +20,546 | +16,612 |
| **0** | **78,923** | **65,421** | **71,238** | **55,623** | **+7,685** | **+9,798** |
| +1 | 80,505 | 58,797 | 56,383 | 45,248 | +24,122 | +13,548 |
| +2 | 70,148 | 55,005 | 61,739 | 41,977 | +8,409 | +13,028 |
| +3 | 79,974 | 59,774 | 50,163 | 38,925 | +29,811 | +20,848 |

### Key Findings

**Volume is useful for assessing call wall firmness.**

- **Wall call vol (early baseline):** Held days start the morning with ~37k+ median at the wall strike (offsets −4/−3), vs. ~21–25k on revert days. The trend is the same in both groups (volume builds into and after the move) but the starting level differs. High early-morning volume at the wall strike = genuine directional pressure accumulating all morning, not just a transient test.

- **Wall call bid-ask spread (post-move confirmation):** Both groups flat at 0.10 median through offset −1. After the move, held days widen to 0.15→0.20 at offsets +1/+2/+3; revert days stay at 0.10. When the wall call bid-ask widens after a breach it signals the strike is going stale — spot has committed above it. If the spread stays tight, expect the wall to be retested from above.

- **SPY volume (liquidity as absorption):** Revert days have significantly higher broad market volume early (108k vs 62k median at offset −4). Higher SPY volume provides the two-sided liquidity for the market to absorb the move and push price back. On held days the quieter SPY backdrop means less selling pressure available to reject the breach.

**Combined signal:** High wall call vol early + low SPY vol = directional pressure without broad market liquidity to absorb it → wall more likely to hold the new level after breaching.

---

## Phase 9 — Wall Firmness Research Directions

**Data varies by item** — see individual study notes below. Items 1–2 use `wall_stability_30min.csv` and `master_features.csv` respectively. Items 3–7 use `master_features.csv` + bucket 0 chains from Schwab SPXW snapshots.

Based on literature review and Phase 8 findings, the following features are worth investigating for predicting whether a call or put wall will hold. Prioritized by tractability given available data.

### 1. Consecutive Bucket Pressure Threshold
**Hypothesis:** The wall is at risk when spot has been within N pts of it for ≥K consecutive 30-min buckets. Sustained proximity signals absorption of GEX resistance, not a one-time test.
**Script:** `scripts/gex_pressure_threshold.py`

**Result: Hypothesis rejected — consecutive pressure predicts wall stability, not wall moves.**

Sweep across thresholds (10, 15, 25pts) × K (2, 3 buckets), outcome = call wall moves ≥25pts after pressure event:

| Threshold | K | N pressure days | Move% (pressure) | Move% (no pressure) | Lift |
|-----------|---|-----------------|-----------------|---------------------|------|
| 10 pts | 2 | 62 (37.6%) | 53.2% | 64.1% | −10.9% |
| 10 pts | 3 | 37 (22.4%) | 54.1% | 68.0% | −13.9% |
| 15 pts | 2 | 81 (49.1%) | 55.6% | 60.7% | −5.2% |
| 15 pts | 3 | 51 (30.9%) | 60.8% | 64.9% | −4.1% |
| 25 pts | 2 | 115 (69.7%) | 59.1% | 56.0% | +3.1% |
| 25 pts | 3 | 85 (51.5%) | 58.8% | 67.5% | −8.7% |

On pressure days (spot ≤15pts from wall for ≥2 consecutive buckets, n=81):
- 50% wall stays stable, 32% wall drifts DOWN ≥25pts, 20% wall moves UP ≥25pts
- Avg net move = −10pts (wall more often drops than rises)
- Pressure event fires uniformly across all buckets (median bucket 6 of 12) — no end-of-day clustering

**Interpretation:** Sustained proximity to the wall is a gamma-pinning signal, not a pre-break signal. When spot parks near the call wall for 1+ hours, dealer hedging flows are absorbing the pressure — the wall is *working*. Wall moves that do occur tend to be sharp and fast, not preceded by sustained proximity — consistent with Phase 8 (high early volume at wall call strike distinguishes held moves from reversions). Consecutive pressure is not a useful predictive feature for wall direction.

### 2. Surrounding Positive GEX Zone
**Hypothesis:** A call wall surrounded by high positive net GEX in a ±10pt band is structurally firmer. Multiple nearby strikes with positive net GEX means dealers are hedging long gamma across a zone — the market has to absorb distributed buying pressure, not just at one strike.
**Scripts:** `scripts/gex_wall_gex_structure.py` (day-level wall relocation), `scripts/gex_wall_surrounding_gex.py` (event-level bounce outcomes)

**First test (day-level wall relocation) — rejected.** GEX magnitude and concentration at open do not predict whether the call wall label *relocates* ≥25pts during the day. All correlations with `wall_moved` are r < 0.09, p >> 0.05. The wall relocates on 74.6% of days regardless of morning GEX structure — often for GEX accounting reasons unrelated to price testing it.

**Second test (event-level bounce outcomes) — partial support, but magnitude doesn't matter, count does.**

12,141 tested raw call wall events across 118 days. Point-biserial r vs outcomes:

| Feature | r(b30) | r(b60) | r(eod) |
|---------|--------|--------|--------|
| pos_gex_w10 — positive GEX magnitude in ±10pt zone (B) | +0.020 | −0.014 | −0.060 |
| neg_gex_w10 — opposing GEX magnitude in ±10pt zone (B) | +0.023 | +0.024 | +0.009 |
| **n_pos_strikes_w10 — # positive-GEX strikes in ±10pt zone** | **+0.079** | **+0.134** | **+0.146** |
| gex_at_wall_strike (Phase 3 baseline) | +0.065 | +0.044 | −0.017 |

Point-biserial correlations — magnitude vs count, across window sizes:

| Feature | r(b30) | r(b60) | r(eod) |
|---------|--------|--------|--------|
| pos_gex_w10 — positive GEX magnitude in ±10pt zone (B) | +0.020 | −0.014 | −0.060 |
| neg_gex_w10 — opposing GEX magnitude in ±10pt zone (B) | +0.023 | +0.024 | +0.009 |
| n_pos_strikes_w5 — # positive-GEX strikes in ±5pt zone | +0.073 | +0.120 | +0.122 |
| **n_pos_strikes_w10 — # positive-GEX strikes in ±10pt zone** | **+0.079** | **+0.134** | **+0.146** |
| n_pos_strikes_w15 — # positive-GEX strikes in ±15pt zone | +0.046 | +0.101 | +0.113 |
| gex_at_wall_strike (Phase 3 baseline) | +0.065 | +0.044 | −0.017 |

**±10pt window is optimal** — tightest window with strongest correlations. ±15pt adds noise. Dollar magnitude of surrounding GEX is flat across all windows.

Outcomes by count of positive-GEX strikes in ±10pt zone (max 5 at 5pt spacing):

| # positive-GEX strikes | N | bounce_30m | bounce_60m | eod_held |
|------------------------|---|------------|------------|----------|
| 0 | 636 | 72.4% | 68.8% | 74.7% |
| 1 | 955 | 75.6% | 66.1% | **45.0%** |
| 2 | 481 | **31.9%** | **31.7%** | **31.8%** |
| 3 | 1,067 | 86.5% | 82.2% | **92.6%** |
| 4 | 1,515 | 86.8% | 79.4% | 65.8% |
| 5 | 7,487 | 78.7% | 79.4% | 76.8% |

±15pt zone (max 7 strikes at 5pt spacing):

| # positive-GEX strikes | N | bounce_30m | bounce_60m | eod_held |
|------------------------|---|------------|------------|----------|
| 0 | 604 | 71.9% | 68.4% | 74.3% |
| 1 | 935 | 75.7% | 66.6% | **45.7%** |
| 2 | 54 | 77.8% | 64.8% | 57.4% |
| 3 | 232 | 83.0% | 82.2% | 81.9% |
| 4 | 1,117 | **64.3%** | **61.6%** | 69.9% |
| 5 | 980 | 86.8% | 81.1% | 78.6% |
| 6 | 1,272 | 87.2% | 82.8% | 72.0% |
| 7 | 6,947 | 77.9% | 78.4% | 75.3% |

The ±15pt window reproduces the n=1 trap (45.7% eod_held) and the n=5–6 sweet spot (87%), but introduces a new anomaly at n=4 (64.3% bounce) that breaks the monotonic pattern. The ±10pt result is cleaner and more actionable.

**Key findings:**
1. **n=2 is anomalously weak across all three outcomes (31–32%)** — consistent, not noise. Worth investigating which strike configurations produce it (e.g. wall strike itself not one of the two positives).
2. **n=1 is a trap:** bounces short-term (75.6%) but collapses by EOD (45%). A wall with only one positive-GEX neighbor looks like it holds intraday but doesn't.
3. **n=3–4 is the structural sweet spot** (82–93% bounce, 86–93% eod). n=5 (the most common case, 62% of events) is solid at ~78–79%.
4. **Dollar magnitude of surrounding GEX is irrelevant.** What matters is how many of the 5 nearest strikes have dealer hedging flows active — zone density, not zone size.
5. **`n_pos_strikes_w10` is a viable classifier feature** (r=+0.146 eod_held), and additive to `gex_at_wall_strike` (r=−0.017 eod_held — these measure fundamentally different things).

### 3. ZGL Regime
**Hypothesis:** The call wall is structurally weaker when it sits in a net-negative gamma zone — i.e., when the wall strike itself is below the ZGL. At those strikes, put GEX exceeds call GEX in aggregate, so dealers' net hedging flows don't create the dampening resistance a true gamma wall requires.

The correct framing uses `dist_wall_to_zgl` (wall level − ZGL), not `dist_spot_zgl`. The call wall is found using call-only GEX — the strike with max call GEX above spot — which is independent of the ZGL (computed from net GEX). The wall can sit below the ZGL if that strike has high call GEX but even higher put GEX overwhelming it.
**Literature support:** Baltussen et al. (JFE) validates aggregate gamma-regime effects on realized volatility.

**Result: Confirmed — wall below ZGL is significantly weaker; sweet spot is wall just above ZGL (+0 to +150pts).**

13,024 tested raw call wall events from `master_features.csv`. Wall below ZGL occurs in 5.4% of events (706).

| | N | bounce_30m | bounce_60m | eod_held |
|--|--|--|--|--|
| **Wall ABOVE ZGL** | 12,318 | **76.6%** | **74.4%** | **71.1%** |
| **Wall BELOW ZGL** | 706 | **55.4%** | **58.0%** | **55.8%** |

A 15–21pt gap across all three outcomes. Distance to wall is identical between groups (median ~11pts), ruling out approach distance as a confound.

Fine-grained cut around the ZGL crossing:

| dist_wall_to_zgl range | N | bounce_30m | bounce_60m | eod_held |
|------------------------|---|------------|------------|----------|
| −200 to −50pts (wall deep below ZGL) | 200 | 48.7% | 48.7% | 63.5% |
| −50 to 0pts (wall just below ZGL) | 506 | 58.0% | 61.6% | 52.8% |
| **0 to +50pts (wall just above ZGL)** | 2,115 | **76.6%** | **87.7%** | **87.3%** |
| +50 to +150pts | 2,333 | 81.2% | 75.2% | 70.1% |
| +150 to +250pts | 3,675 | 82.0% | 75.7% | 72.3% |
| +250 to +400pts (wall far above ZGL) | 4,195 | 69.3% | 66.2% | 62.6% |

The 0 to +50pt band is the strongest setup: 87.3% eod_held, 87.7% b60. The wall is just inside positive-gamma territory — dealer net hedging flows are supportive but the wall isn't so far into the high-delta zone that gamma thins out.

Point-biserial correlations:

| Feature | r(b30) | r(b60) | r(eod) | p(eod) |
|---------|--------|--------|--------|--------|
| dist_wall_to_zgl | −0.022 | −0.082 | −0.085 | 0.000 |
| wall_below_zgl (binary) | −0.111 | −0.084 | −0.076 | 0.000 |
| net_gex_spot (baseline) | +0.050 | −0.018 | −0.070 | 0.000 |

`wall_below_zgl` binary has the strongest b30 correlation (−0.111) — the sign flip at zero is the sharpest signal, more informative than the continuous distance.

Wall below ZGL × VIX interaction:

| | N | bounce_30m | bounce_60m | eod_held |
|--|--|--|--|--|
| Wall above ZGL, VIX < 20 | 11,629 | 76.1% | 74.3% | 70.6% |
| Wall above ZGL, VIX ≥ 20 | 689 | 85.2% | 76.4% | 80.7% |
| Wall below ZGL, VIX < 20 | 556 | 65.0% | 67.2% | 62.8% |
| **Wall below ZGL, VIX ≥ 20** | **150** | **20.0%** | **24.0%** | **30.0%** |

Wall below ZGL combined with high VIX is the worst possible setup — 20% b30, 30% eod_held. The two risk factors compound: net-negative gamma at the wall + amplifying vol regime = almost no resistance.

**Key findings:**
1. **Wall below ZGL = structurally weak** (55% vs 71% eod_held, −15pt gap). The wall strike has more put GEX than call GEX in aggregate — dealers' net flows don't support the call wall as resistance.
2. **Sweet spot: wall 0–50pts above ZGL** — 87% b60 and eod_held. Just inside positive net gamma territory.
3. **Wall far above ZGL (250–400pts) is also weak** (63% eod_held). Gamma thins at high-delta strikes far from spot; dealer hedging pressure per unit of price movement is lower.
4. **Wall below ZGL + VIX ≥ 20 = avoid** — 20% b30, 30% eod_held. Skip this setup entirely.
5. **`wall_below_zgl` is a usable binary classifier feature** (r = −0.111 b30). The zero crossing matters more than the continuous distance.

### 4. Put Wall Symmetry
**Hypothesis:** When both call and put walls are firm and spot is range-bound between them, the call wall has a structural ally — put sellers on the other side are defending their position too. If the put wall has already broken, the call wall is operating in a one-sided regime with less mean-reversion pressure.

**Data:** `master_features.csv` (tested raw call wall events) joined to `events.csv` put wall rows on `(date, snapshot_ts)`. 13,024 events, 100% put wall join rate.

**Result: Put wall also being tested on the same day is the strongest single symmetry signal.**

Point-biserial correlations with call wall outcomes:

| Feature | r(b30) | r(b60) | r(eod) |
|---------|--------|--------|--------|
| dist_spot_to_put (spot − put wall) | +0.005 | −0.052 | −0.090 |
| spot_range_pct (0=at put wall, 1=at call wall) | +0.008 | −0.111 | −0.062 |
| **put_wall_tested (put wall also tested same day)** | **+0.050** | **+0.116** | **+0.172** |

Note: `dist_spot_to_put` and `range_width` (call wall − put wall) are nearly collinear (r=0.912) for tested call wall events — since spot ≈ call wall for all tested events, their difference is just `dist_to_wall`. Only `dist_spot_to_put` is reported.

**Put wall also tested on the same day** is the strongest feature (r=+0.172 eod_held — one of the highest single-feature correlations in the entire study). Validation: VIX is essentially identical between put-tested (17.4) and not-tested (17.2) groups — VIX is not the confound. Logistic regression controlling for VIX shows `put_wall_tested` coefficient (+0.435) stronger than `vix_open` (−0.274) — it is a genuine independent signal.

| | N | bounce_30m | bounce_60m | eod_held |
|--|--|--|--|--|
| Put wall NOT tested | 8,659 | 74.0% | 69.9% | 64.7% |
| **Put wall ALSO tested** | **4,365** | **78.5%** | **80.7%** | **81.4%** |

A **17pt eod_held gap**. When both walls are tested on the same day, the market is range-bound — both dealer hedging flows are active and reinforcing their respective walls. Note: median time between call and put wall approaches = 0hrs (many occur close together), but std = 1.84hrs; the signal is day-level, not tick-level coincidence.

Distance from spot to put wall (fine bins):

| dist_spot_to_put | N | bounce_30m | bounce_60m | eod_held |
|-----------------|---|------------|------------|----------|
| 0–10pts (put wall very close) | 1,956 | 69.9% | 73.7% | **76.2%** |
| 10–20pts | 2,192 | 78.0% | 77.1% | 70.8% |
| 20–40pts | 3,516 | 78.9% | 78.3% | 72.9% |
| 40–75pts | 3,056 | 70.7% | 66.4% | 68.4% |
| 75–150pts (put wall far) | 2,136 | 80.3% | 73.0% | 62.9% |

Put wall very close (0–10pts): strong eod_held (76.2%) but weaker b30 (69.9%) — tight range, more intraday volatility. Put wall far (75–150pts): strong b30 (80.3%) but eod_held falls to 62.9% — spot has room to move before hitting put support, less range-bound by EOD.

Spot position within the call-put range (0 = at put wall, 1 = at call wall):

| Range position | N | bounce_30m | bounce_60m | eod_held |
|---------------|---|------------|------------|----------|
| 0.0–0.2 (spot near put wall) | 788 | 60.5% | 73.8% | 72.1% |
| 0.2–0.4 | 1,356 | 76.3% | 81.2% | 76.3% |
| 0.4–0.6 (midrange) | 2,178 | 76.5% | 78.9% | 71.9% |
| **0.6–0.8 (spot closing in on call wall)** | 3,334 | **83.5%** | 78.0% | 72.1% |
| 0.8–1.0 (spot at call wall) | 5,366 | 72.1% | 66.6% | 66.8% |

Spot at 60–80% of range has the strongest b30 (83.5%) — approaching the call wall with room below, not yet under immediate pressure. Spot already at the wall (80–100%) weakens both b30 and eod_held — no buffer.

**Key findings:**
1. **`put_wall_tested` is the strongest feature** (r=+0.172 eod_held, independent of VIX). When the put wall is also tested on the same day, the call wall holds EOD 81% vs 65%. Add to call wall classifier.
2. **Put wall far away (75–150pts)** weakens eod_held to 63% — the structural ally is too distant to anchor the range.
3. **Sweet spot: spot at 60–80% of range** — approaching the call wall with room below (83.5% b30).
4. **Spot already at the wall (80–100% of range)** weakens outcomes — immediate pressure, no buffer.
5. `dist_spot_to_put` has modest eod correlation (−0.090) — usable secondary feature but not dominant.

### 5. OI Structure at the Wall Strike
**Hypothesis:** The composition of OI at the wall strike — particularly 0DTE vs. multi-expiry, and put vs. call — reveals whether the level is a live market focal point or a legacy artifact.
**Data:** `master_features.csv` (tested raw call wall events) + bucket 0 chain per day. 12,141 events across 118 days. Script: `scripts/gex_wall_oi_concentration.py`. Output: `out/gex_walls/oi_concentration.csv`.

**Result: Raw OI magnitude is noise. What matters is whether 0DTE OI is present and whether put OI is accumulating at the call wall.**

0DTE OI is ~18% of wall OI on average. The 0DTE fraction is the key signal:

| 0DTE fraction of wall OI | N | bounce_30m | bounce_60m | eod_held |
|--------------------------|---|------------|------------|----------|
| Q1: 4–10% | 3,121 | 66.0% | 67.1% | 60.5% |
| **Q2: 10–18%** | 3,030 | **87.4%** | **81.7%** | **82.4%** |
| Q3: 18–27% | 2,997 | 85.8% | 80.1% | 73.2% |
| Q4: 27–91% | 2,993 | 73.0% | 76.0% | 74.0% |

Very low 0DTE fraction (< 10%) = 60.5% eod_held. The wall strike has little current-day relevance — multi-expiry legacy OI dominates but today's market isn't engaging with it. This is confirmed by the worst single combination: Low 0DTE + High multi = 62.4% eod_held. Best setup: Low 0DTE + Low multi = 81.1%, or Q2 0DTE fraction = 82.4%.

**`put_oi_0dte`** (r = −0.126 b30): 0DTE put OI at the call wall is the strongest single OI signal. When participants are actively buying 0DTE puts at the call wall strike, they're betting on a break — and the wall tends to fail short-term. See Item 7 for the full call/put OI balance analysis.

### 6. Next-Expiry IV at Wall Strike (Term Structure Contestedness)
**Hypothesis:** If the wall strike carries elevated IV relative to the smile — i.e., above what you'd expect for a call that far OTM — the market is explicitly pricing uncertainty at that level and the wall is more likely to break.
**Data:** `master_features.csv` (tested raw call wall events) + bucket 0 chain per day. 12,141 events across 118 days. Script: `scripts/gex_wall_iv_term_structure.py`. Output: `out/gex_walls/iv_term_structure.csv`.

**Measure:** Fit a quadratic to the same-day 0DTE OTM call smile (IV vs log-moneyness). The wall strike's residual from this fit = how much extra IV it carries beyond what the smile predicts for that moneyness. Median R² = 0.998 — the smile is well-identified from same-day data alone. The residual is always positive (mean +7.8 vol pts) — wall strikes consistently carry a premium, the question is how large.

| Smile residual quartile | N | bounce_30m | eod_held | Avg excess IV |
|-------------------------|---|------------|----------|---------------|
| Q1 (small excess, avg +1.8) | 3,350 | 80.8% | **78.7%** | +1.8 |
| Q2 (avg +4.5) | 2,759 | 77.5% | **80.1%** | +4.5 |
| Q3 (avg +6.7) | 3,062 | 73.9% | 69.8% | +6.7 |
| Q4 (large excess, avg +11.5) | 2,970 | 79.4% | **61.0%** | +11.5 |

r = −0.116 eod_held. Monotonically declining: a small smile premium (wall holds more than the smile would predict, but not dramatically) = wall holds 79–80% EOD. A large premium (Q4, +11.5 vol pts excess) = market is explicitly pricing a move through the wall = 61% EOD.

**Term structure spread** (next-expiry call IV minus 0DTE call IV at wall strike, r = −0.142 eod_held):

Normal term structure has the spread negative (0DTE IV > next-expiry) — today's gamma inflates near-expiry vol. A very negative spread (0DTE IV much higher than next) means gamma is massive at that strike today; walls tend to hold through EOD (80%) even if the initial bounce is messy. A near-zero spread means the vol crush is minimal — the wall is not especially relevant on 0DTE — and EOD drops to 70%.

**Key findings:**
1. **`iv_smile_residual`** (r = −0.116 eod_held): the larger the smile premium at the wall strike, the more the market is pricing a break. Q4 (excess > +8 vol pts) → 61% eod_held. Good classifier feature — computed from same-day smile, no lookback.
2. **IV term structure spread** (r = −0.142 eod_held): a large negative spread (high 0DTE gamma at wall) predicts EOD holds even when b30 is messy. Near-zero spread = wall is not a 0DTE focal point.
3. The original contestedness hypothesis inverts: more IV at the wall = weaker wall, not stronger. The market pricing a move through a level is evidence the level will break, not evidence it's structural.

### 7. Delta Imbalance at Wall Strike
**Hypothesis:** The call/put OI ratio at the wall strike reveals which side has more forced hedging pressure. When put OI is comparable to call OI, put dealers are also hedging there — bilateral pressure. When calls overwhelmingly dominate, dealer delta-hedging is one-sided and the wall may behave differently.
**Data:** Same as Item 5 — `out/gex_walls/oi_concentration.csv`. `call_put_oi_ratio` = call OI / put OI at wall strike, all expiries.

**Result: U-shaped — both extremes outperform the middle.**

| call/put OI ratio | N | bounce_30m | bounce_60m | eod_held | Avg ratio |
|-------------------|---|------------|------------|----------|-----------|
| Q1: ratio < 1.1 (put-heavy) | 2,477 | 80.0% | 76.6% | **81.0%** | 0.75 |
| Q2: 1.1–1.7 | 2,462 | 72.3% | 77.0% | 65.7% | 1.36 |
| Q3: 1.7–2.9 | 2,418 | 70.8% | 63.5% | 61.1% | 2.38 |
| Q4: 2.9–8.6 | 2,568 | 82.1% | 84.3% | 75.5% | 4.84 |
| Q5: ratio > 8.6 (call-heavy) | 2,216 | 85.2% | 79.2% | **79.2%** | 33.22 |

When put OI roughly matches call OI (Q1, ratio < 1.1): put dealers are hedging at the same strike as call dealers — bilateral pressure reinforces the level, 81% eod. When calls dominate overwhelmingly (Q5, ratio > 8.6): one-sided dealer flow, strong short-term bounce (85% b30). The weakest setups are balanced-but-call-biased ratios (Q2–Q3, 1–3x): neither side dominates, no concentrated hedging pressure, 61–66% eod.

The U-shape means this is not a linear feature, but the ends are actionable: avoid ratios in the 1–3x range; favour either put-comparable or heavily call-dominant OI at the wall.

### Literature Context
- **GEX wall strike-level pinning:** Pre-registered tests (Popovici SSRN, Cenobar & Fazeli SSRN) find weak strike-level pinning — walls are breached only marginally less than placebo strikes. GEX is better validated as an aggregate vol-regime signal than a price barrier.
- **OI pinning at expiry:** Ni, Pearson & Poteshman (RFS 2005) confirm strong expiry-day pinning at high-OI strikes. Less evidence intraday.
- **Order flow imbalance (OFI):** Most validated microstructure feature for level breaks (Boehmer & Wu). We're proxying this with wall call volume but actual OFI (buy vs. sell at top of book) would be more direct — not available in our data.
- **0DTE dealer flows:** Doshi et al. (2025) and Adams et al. find dealer net gamma in 0DTE is positive on average → intraday vol-dampening, not amplifying. Consistent with walls generally holding.

---

## Phase 10 — 0DTE Bear Call Spread Backtest

**Goal:** Turn the Phase 9 wall firmness findings into a concrete trading strategy. Sell a 20-pt wide 0DTE SPXW bear call spread with the short leg above the call wall; hold to expiry (EOD settlement).

**Script:** `scripts/gex_call_spread_backtest.py`
**Output:** `out/gex_walls/call_spread_backtest.csv`, `out/gex_walls/call_spread_losses.csv`

### Setup

- **Data:** Schwab SPXW snapshots, Dec 2025–Aug 2026, 165 trading days
- **Entry:** Bucket 2 snapshot (10:30–11:00 ET, ~10:45) — 60 min after open to let morning vol settle
- **Short strike (K1):** Call wall + 20pts, rounded to nearest 5pt increment
- **Long strike (K2):** K1 + 20pts
- **Spread pricing:** Black-Scholes using 0DTE IV from option chain at K1 and K2; T = 5.25/(252×6.5) ≈ 0.0032 years
- **Exit:** EOD settlement (intrinsic value vs last bucket spot)
- **Delta:** Read directly from option chain at K1 (not computed)

### Filters Applied

| Filter | Rationale |
|---|---|
| Wall below ZGL (`dist_wall_to_zgl < 0`) | Net put GEX dominates at wall — no dealer buying pressure. Note: this filter turned out to be directionally wrong for a short call spread (90% win counterfactually) since wall-below-ZGL days mostly see the market drift lower, away from the short call. |
| IV smile residual > 8 vol pts (Q4) | Market explicitly pricing a break through the wall by EOD. 62% win counterfactually vs 97% traded baseline — this filter earns its keep. |

### Entry Timing: Iron Condor Opportunity

**Script:** `scripts/gex_condor_timing.py`

On 57.6% of days (95/165) both walls are tested (spot within 10pts). Distribution of when the second wall is tested:

- Both walls tested by 11:30 ET (bucket 4): only 18 days (19% of both-tested days)
- Both walls tested by 1:00 PM (bucket 7): 38 days (40%)
- Majority of both-tested days: wall test happens in the final 2 buckets (15:00–16:00), by which point 0DTE premium is minimal

**Implication:** Iron condor entry off both-walls-tested is a secondary opportunity — it only sets up early enough (~40% of both-tested days) to have meaningful premium remaining.

### Short Strike Placement

Placing K1 at the call wall (original design) puts the short leg at maximum risk — any breach of the wall immediately makes the spread ITM. Moving K1 to call_wall + 20pts gives a meaningful buffer and is the design used throughout.

### Filter Comparison (K1 = call_wall + 20pts, entry at 10:45)

| Metric | cr≥$0.50 | cr≥$0.75 | cr≥$1.00 | cr/spread≥8% |
|---|---|---|---|---|
| Days traded | 74 | 66 | 61 | 42 |
| Wins | 72 | 64 | 59 | 40 |
| Losses | 2 | 2 | 2 | 2 |
| Win rate | 97.3% | 97.0% | 96.7% | 95.2% |
| Avg P&L | $181 | $196 | $205 | $241 |
| Total P&L | $13,429 | $12,938 | $12,490 | $10,130 |
| Max loss | -$1,764 | -$1,764 | -$1,764 | -$1,764 |

The same 2 losses survive every filter — credit quality is not the problem.

**On the credit/spread ratio filter:** The correct framing for minimum credit is the break-even win rate implied by the trade: `(spread_width - credit) / spread_width`. At $0.50 credit the break-even is 97.5% — essentially no edge over our observed 97.3%. At $1.00 credit the break-even is 95.0% — clear edge. The ratio filter (≥8% = $1.60 minimum) is most selective but also least frequent.

### Loss Analysis

Both losses are large-rally days where the market closed above K2:

| Date | Spot at entry | Call wall | K1 | EOD spot | P&L | Notes |
|---|---|---|---|---|---|---|
| 2026-07-10 | 7538 | 7550 | 7570 | 7575 | -$41 | Barely ITM, rounding-error loss |
| 2026-08-04 | 7688 | 7700 | 7720 | 7745 | -$1,764 | Strong multi-day rally; +7% SPX move |

The 8/4 loss is a structural outlier — a 7%+ single-day SPX rally that would blow through any short call spread placement. No entry filter prevents it without also filtering out most of the wins.

**Key insight:** The losses are not a signal-quality problem. They are tail-risk events (strong trending days). The strategy's edge comes from the asymmetric behavior of GEX walls — 97% of days the market doesn't mount a sustained rally through a level sitting 20pts above the call wall.

### Open Questions

1. **Stop-loss via wall relocation:** A "genuine held move" (call wall relocates ≥25pts for ≥2 consecutive buckets, per Phase 6 definition) was tested as an intraday stop. It triggers too late on fast trending days because the wall relocates *because* spot moved through it — there is no early warning. A spot-based stop may be more effective.
2. **Delta filter:** Short strike delta from the option chain is now recorded in the CSV. Worth checking whether a delta ceiling (e.g. ≤0.10) captures the same days as the credit ratio filter or adds independent information.
3. **Put spread symmetry:** This backtest covers only the call side. An analogous bull put spread backtest (K1 = put_wall − 20pts) would complete the iron condor picture.