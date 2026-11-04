# GEX Wall Study Notes

## Strategy

**GEX Wall Options Spread Strategy**

Use GEX call and put walls as structural price levels to trade mean-reversion via defined-risk options spreads on SPXW:

- **Bear call spread:** sell a call spread at or just above the call wall when high-conviction the wall will hold (price reverses down from resistance). Collect premium if SPX stays below the short strike.
- **Bull put spread:** sell a put spread at or just below the put wall when high-conviction the wall will hold (price bounces up from support). Collect premium if SPX stays above the short strike.

Entry trigger: price comes within ~10pts of the wall level. A logistic regression classifier trained on features (VIX regime, GEX structure, IV levels, time-of-day) assigns a hold probability — only enter when probability exceeds a threshold calibrated for statistical significance.

Target outcome: `bounce_60m` (price remains on the original side of the wall 60 minutes after first approach). EOD held is the ultimate exit anchor.

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
| `scripts/fetch_vix_dec2025.py` | One-time: fetch Dec 2025 VIX 1-min from IBKR → `data/candles/` |

### Data (data/candles/)

| File | Purpose |
|------|---------|
| `data/candles/vix_1min_dec2025.parquet` | Dec 2025 VIX 1-min bars fetched from IBKR (not in lakehouse) |
| `data/candles/spy_1min_dec2025.parquet` | Dec 2025 SPY 1-min bars fetched from IBKR (not in lakehouse) |

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
| `out/gex_reversal/inflections_lb10_mm8.csv` | Reversal study: 1,511 inflection points with GEX distances |

---

## Phase 1 — Forward Wall Study

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

**Data:** Dec 2025–Aug 2026, 138 days with all three snapshots (morning ~9:45, midday ~13:00, close ~15:30 ET).
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