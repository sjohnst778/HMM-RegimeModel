# CLAUDE.md — HMM-RegimeModel Project Context

This file provides persistent context for Claude across sessions.
Share it at the start of any new session on this project.

---

## Objective

Build a deployable early warning system for **persistent market downturns** — not just
volatility spikes, but sustained bear and crisis regimes that are likely to persist.

The system produces a weekly **downturn risk score [0, 1]** combining:
- A Hidden Markov Model (HMM) layer for regime identification
- A Bayesian Online Changepoint Detection (BOCPD) layer for structural break detection
- An XGBoost ensemble layer combining both signals with macro features
- A self-contained HTML dashboard for presenting results

The key insight: BOCPD detects structural breaks in cross-asset relationships
*before* prices move, HMM confirms the regime has changed. Together they give
both early warning and regime confirmation.

---

## Project Status: COMPLETE ✅

All layers built, validated, running end-to-end, and outputting a live dashboard.

| Component | Status | Key metric |
|-----------|--------|------------|
| HMM | ✅ Complete | 3 states, avg run 54-166d |
| BOCPD | ✅ Complete | Leads HMM by 28-109d across 4 crises |
| Ensemble | ✅ Complete | AUC 0.650 ± 0.055 |
| Pipeline | ✅ Complete | Live reading confirmed 2026-04-17 |
| Dashboard | ✅ Complete | HTML output, historical date support |
| Model persistence | ✅ Complete | Versioned bundles, Eikon-ready |
| Data cache | ✅ Complete | CSV cache, 6-month refresh, 11 tickers |

**Next work:** Eikon data integration when access is available (all code scaffolded,
one-line config change to activate).

---

## Architecture

```
Raw market & macro data
  Equity:  ^GSPC (S&P500), ^FTSE (FTSE100), ^N225 (Nikkei), ^HSI (Hang Seng)
           ^STOXX50E dropped from BOCPD — 59% zero returns pre-2007 (yfinance)
  Macro:   HYG, LQD, TIP, ^TNX, ^VIX, ^IRX  (yfinance ETF proxies)
           CDX/iTraxx, VIX3M, VVIX, yield curve, TED spread  (Eikon — gated off)
            │
    data/cache/  ← CSV cache (one file per ticker)
    Read CSV → refresh last 6 months from provider → merge → save
            │
    ┌───────┴────────┐
    │                │
  HMM              BOCPD
(regime state     (structural breaks in
 from ^GSPC)       cross-market correlations
                   + equity/rates)
    │                │
    └───────┬────────┘
            │
    XGBoost ensemble (27 features, 1-day lagged)
    + isotonic calibration
            │
    ┌───────┴────────────────┐
    │                        │
  risk_score_raw        risk_score_calibrated
  (continuous,          (step function — use for
   use for alerts)       probability statements)
            │
    output/scores.csv  →  dashboard.html
```

---

## Data Layer

### Equity Tickers
| Ticker | Index | Eikon RIC | BOCPD status |
|--------|-------|-----------|--------------|
| `^GSPC` | S&P 500 (US) | `.SPX` | ✓ used |
| `^FTSE` | FTSE 100 (UK) | `.FTSE` | ✓ used |
| `^N225` | Nikkei 225 (Japan) | `.N225` | ✓ passes quality filter |
| `^HSI` | Hang Seng (HK/China) | `.HSI` | ✓ passes quality filter |
| `^STOXX50E` | Euro Stoxx 50 | `.STOXX50E` | ✗ dropped — 59% zero returns |

### Data Cache Layer

**File:** `data/cache.py`

All data fetches go through a local CSV cache in `data/cache/`. On each run:
1. Read existing CSV for the ticker
2. Fetch only the **last 6 months** from the provider
3. Merge (fresh data wins on overlap — handles late revisions)
4. Save updated CSV back to cache
5. Return full combined series to caller

This means the provider (yfinance or Eikon) is only ever called for recent data
after the initial backfill, regardless of `DATA_START=1995-01-01`.

**Cache files (one per ticker):**
| File | Ticker | History | Size |
|------|--------|---------|------|
| `_GSPC.csv` | ^GSPC | 1995-present | ~217KB |
| `_FTSE.csv` | ^FTSE | 1995-present | ~198KB |
| `_N225.csv` | ^N225 | 1995-present | ~196KB |
| `_HSI.csv` | ^HSI | 1995-present | ~197KB |
| `_STOXX50E.csv` | ^STOXX50E | 2007-present | ~128KB |
| `_TNX.csv` | ^TNX | 1995-present | ~224KB |
| `_VIX.csv` | ^VIX | 1995-present | ~224KB |
| `_IRX.csv` | ^IRX | 1995-present | ~227KB |
| `HYG.csv` | HYG | 2007-present | ~136KB |
| `LQD.csv` | LQD | 2002-present | ~170KB |
| `TIP.csv` | TIP | 2003-present | ~160KB |

**Cache management commands:**
```bash
python data/cache.py --backfill          # initial population (run once per environment)
python data/cache.py --backfill --ticker ^GSPC  # single ticker
python data/cache.py --status            # show coverage per ticker
python data/cache.py --clear             # wipe all (requires re-fetch)
```

**Providing historical CSVs for Eikon:**
If you have historical data from another source, drop CSVs directly into `data/cache/`
with format: `date` index, `close` column. The cache layer will read them and only
fetch the last 6 months from Eikon, staying within API limits.

**CSV format:**
```
date,close
1995-01-03,470.42
1995-01-04,472.49
```

**gitignore:** `data/cache/*.csv` is excluded from git — regenerate with `--backfill`.

### Data Quality Filter
`build_bocpd_features()` automatically drops any ticker with <90% non-zero returns.
`^STOXX50E` has 3,293 runs of repeated prices in yfinance pre-2007 data — undefined
rolling correlations would truncate BOCPD history. Clean with Eikon data.

### Data Provider Abstraction
`DATA_PROVIDER` in `config.py` controls which provider is used:
- `"yfinance"` — free, current, development use
- `"eikon"` — production, one-line switch, all code already scaffolded

To switch: `DATA_PROVIDER = "eikon"` in config.py, then `python pipeline/score.py`.

---

## HMM Layer

**File:** `models/hmm.py` — `RegimeHMM` class

**Features (v2):** 3 features on `^GSPC` only (`build_hmm_features_v2`):
- `ret_1d` — 1-day log return
- `vol_21d` — 21-day rolling annualised vol
- `drawdown_252d` — drawdown from 252-day rolling high (critical for state separation)

**Decoding:** Forward-backward smoothed posteriors → 20-day rolling mean → persistence
filter (min 20 days). Prevents single-day state flips.

**States:** Re-ordered by mean return. 0=crisis, 1=bear, 2=bull.

**Validated state characteristics:**
| State | Vol | Drawdown | Avg run |
|-------|-----|----------|---------|
| Crisis | 0.261 | -16.9% | 166d |
| Bear | 0.134 | -3.1% | 54d |
| Bull | 0.093 | -0.6% | 66d |

**Known limitation:** Bear/bull oscillation due to proximity in feature space.
Acceptable — P(crisis) is the primary ensemble signal, not the state label.

---

## BOCPD Layer

**File:** `models/bocpd.py` — `BOCPDetector` class

**Signal extraction:** `P(run_length ≤ 4)` from the R matrix — NOT `R[0,t]` which
is always near-zero. Short-run-length probability mass spikes at structural breaks.

**Critical implementation note:** `StudentT` likelihood is stateful — instantiate
a **fresh copy per series** in `run()`. Sharing one instance causes shape errors
when series have different lengths.

**Input series (22 total with Nikkei + Hang Seng):**
- Pairwise equity correlations (FTSE/GSPC, FTSE/N225, GSPC/N225, etc.)
- Per-index vol and mean-return series
- Equity/rates cross-correlations (`^TNX`) — flight-to-safety signal
- Credit spread series (Eikon only — CDX/iTraxx via `credit_prices` parameter)

**Config:** `BOCPD_HAZARD_LAMBDA=60` (~1 break per quarter)

**Warmup:** Needs ~180 trading days before posterior is stable. `DATA_START=1995-01-01`
gives 12+ years before GFC. Never deploy on a fresh series without burn-in.

**Validated lead times vs HMM:**
| Crisis | BOCPD lead |
|--------|-----------|
| Dot-com (2000) | 43 days |
| GFC (2007) | 28 days |
| COVID (2020) | 83 days |
| Rate hikes (2022) | 109 days |

---

## Ensemble Layer

**File:** `models/ensemble.py` — `DownturnEnsemble` class

**Label:** Binary — 1 if S&P 500 drops >10% within next 20 trading days.
Best config from grid search: 20d/-10% (AUC 0.650) vs 42d/-10% (0.607), 63d/-7% (0.581).

**Class imbalance:** ~8% positive rate. `scale_pos_weight = (1-pos_rate)/pos_rate ≈ 11.6`.
Without this the model predicts all-zero.

**Walk-forward:** Expanding window on FULL history (1995-present). Must include
dot-com + GFC in training — post-2018 only gives AUC ~0.4 (almost no crisis examples).

**Two score outputs:**
- `risk_score_raw` — raw XGBoost, varies continuously day-to-day. Use for alerts and tracking.
- `risk_score_calibrated` — isotonic calibration applied. Use for probability statements
  ("~13% chance of -10% drawdown in 20 days"). Warning: creates step function — score
  can freeze at one value for days/weeks when inputs are stable.

**Thresholding:** Fixed 0.5 is meaningless for 8% positive rate models.
Use `p90` (top 10% of risk days) and `p95` (top 5%) computed from full history.

**Validated performance:**
| Metric | Value |
|--------|-------|
| Mean AUC (5-fold walk-forward) | 0.650 ± 0.055 |
| Dot-com recall (p90) | ✓ 268 days above threshold |
| GFC recall (p90) | ✓ 183 days above threshold |
| COVID recall | ✗ missed — pure exogenous shock |
| Rate hikes recall (p90) | ✓ 121 days above threshold |

**COVID limitation:** Architecture detects slow-building structural deterioration.
COVID was an exogenous shock with zero build-up — inherently unpredictable. Accept
and document; do not overfit around it.

**Feature importance (top 5):**
1. `hmm_state` (19.5%)
2. `vol_63d` (10.8%)
3. `macro_vol_ig_spread` (6.7%)
4. `hmm_p_state_0` (6.6%)
5. `drawdown_252d` (6.5%)

`bocpd_break_freq_30d` at 4.3% — frequency of breaks more useful than daily spike.

**1-day lag:** All features shifted by 1 day before training to prevent look-ahead bias.

---

## Pipeline

**File:** `pipeline/score.py`

**Modes:**
```bash
python pipeline/score.py                          # full retrain + score (~5 min)
python pipeline/score.py --score-only             # load saved bundle, score only (~2 min)
python pipeline/score.py --score-only --tag eikon_v1  # specific bundle
```

**Output — `output/scores.csv`:**
| Column | Description | Use for |
|--------|-------------|---------|
| `hmm_state` | 0=crisis, 1=bear, 2=bull | Regime label |
| `hmm_p_crisis` | P(crisis state) | Leading HMM signal |
| `bocpd_p_change` | P(structural break today) | Daily break spike |
| `bocpd_break_freq_30d` | Break days in last 30 | Sustained stress measure |
| `risk_score_raw` | Raw XGBoost score | Alerts, trend tracking |
| `risk_score_calibrated` | Calibrated probability | Stakeholder communication |
| `alert_p90` | 1 if raw > p90 threshold | Alert flag |
| `alert_p95` | 1 if raw > p95 threshold | High conviction alert |

**Latest validated reading (2026-04-17):**
- Regime: BEAR, P(crisis)=0.0005
- Risk score raw: 0.2419 (p90=0.396, no alert)
- Risk score calibrated: 0.2176
- BOCPD break freq 30d: 11
- Interpretation: tariff-driven correction, not systemic crisis
- Cache: reading from CSV, only refreshing last 6 months from provider

**Weekly cron:**
```bash
# crontab -e
0 7 * * 1 /bin/bash ~/projects/HMM-RegimeModel/pipeline/run_weekly.sh
```

---

## Dashboard

**File:** `dashboard.py`

```bash
python dashboard.py                        # latest dashboard
python dashboard.py --date 2008-10-10      # historical view
python dashboard.py --date 2020-03-23      # COVID crash
```

Output: `output/dashboard_YYYYMMDD.html` (self-contained, embeds charts as base64).
Running without `--date` also writes `output/dashboard.html` as a convenience latest.

**Thresholds use full history** even in historical view — consistent meaning across dates.

**Dashboard sections:**
1. Alert banner (green/amber/red)
2. Five KPI cards with day-on-day deltas
3. 3-year risk score chart with p90/p95 lines and regime colour bar
4. 90-day signal breakdown (P(crisis), BOCPD, risk score)
5. 8-week history table
6. Signal guide (plain English for non-technical stakeholders)

---

## Model Persistence

**File:** `utils/persist.py`

```python
# Save after retraining
save_model_bundle(hmm, ensemble, feature_names, tag="yfinance_v1", metadata={...})

# Load for scoring
bundle = load_model_bundle("latest")

# Compare yfinance vs Eikon
compare_bundles("yfinance_v1", "eikon_v1")  # shows features added/removed, AUC delta
```

**Baseline:** `yfinance_v1`, feature hash `3d3ec0c5` (27 features, with Nikkei + Hang Seng).
Previous baseline before Asian indices: hash `5dd8c027` (23 features).

---

## Eikon Migration

All Eikon feature code is already written and gated behind `FEATURE_FLAGS`.
To activate:

1. `DATA_PROVIDER = "eikon"` in `config.py`
2. Set `EIKON_API_KEY` (env var or config)
3. `python pipeline/score.py` — full retrain with new data
4. Compare: `compare_bundles("yfinance_v1", "eikon_v1")`

**What activates automatically:**
| Feature group | Flag | Expected benefit |
|---------------|------|-----------------|
| Credit spreads (CDX/iTraxx) | `credit_spreads` | GFC lead 28d → 60+d, COVID detection |
| VIX term structure | `vix_term_structure` | COVID early warning, fold 3 AUC |
| Yield curve shape | `yield_curve_shape` | 2022 lead improvement |
| Funding stress (TED/FRA-OIS) | `funding_stress` | GFC confirmation signal |

**Eikon RICs configured for:**
- Equity: `.SPX`, `.FTSE`, `.STOXX50E`, `.N225`, `.HSI`, `.KS11`, `.AXJO`
- Credit: `CDXHY5Y=GFI`, `CDXIG5Y=GFI`, `ITXEB5Y=GFI`, `ITXEXO5Y=GFI`
- Vol: `.VIX`, `.VIX3M`, `.VVIX`
- Rates: `USBMK=RR`, `US2YT=RR`, `US3MT=RR`
- Funding: `TEDRATE=`, `FRAOIS3M=`

---

## Project Structure

```
HMM-RegimeModel/
├── config.py                  # All parameters + feature flags + provider switch
├── dashboard.py               # HTML dashboard generator
├── validate_hmm.py            # HMM validation + regime charts
├── validate_bocpd.py          # BOCPD validation + lead/lag analysis
├── validate_ensemble.py       # Ensemble validation + crisis recall
├── diagnose.py                # File version diagnostic (use when updates not landing)
│
├── data/
│   ├── fetch.py               # Provider abstraction (yfinance / Eikon)
│   ├── cache.py               # CSV cache layer — read/write/merge/backfill
│   ├── features.py            # All feature engineering:
│   │                          #   build_hmm_features_v2()       ← HMM input
│   │                          #   build_bocpd_features()        ← BOCPD input
│   │                          #   build_ensemble_features()     ← ensemble input
│   │                          #   build_labels()                ← training labels
│   └── cache/                 # CSV files, one per ticker (gitignored)
│       ├── _GSPC.csv          # ^GSPC S&P 500
│       ├── _FTSE.csv          # ^FTSE FTSE 100
│       ├── _N225.csv          # ^N225 Nikkei 225
│       ├── _HSI.csv           # ^HSI Hang Seng
│       └── ...                # (11 files total)
│
├── models/
│   ├── hmm.py                 # RegimeHMM — COMPLETE
│   ├── bocpd.py               # BOCPDetector — COMPLETE
│   └── ensemble.py            # DownturnEnsemble — COMPLETE
│
├── utils/
│   └── persist.py             # Model versioning + bundle save/load/compare
│
├── pipeline/
│   ├── score.py               # End-to-end scoring pipeline
│   └── run_weekly.sh          # Cron-ready weekly runner
│
├── tests/
│   └── test_hmm.py            # pytest unit tests for RegimeHMM
│
├── models/saved/              # Versioned model bundles (gitignore large files)
│   ├── yfinance_v1_*/         # Baseline bundle (hash: 3d3ec0c5, 27 features)
│   └── latest -> yfinance_v1_*
│
├── output/                    # All outputs (gitignore)
│   ├── scores.csv             # Full score history
│   ├── dashboard.html         # Latest dashboard
│   ├── dashboard_YYYYMMDD.html # Historical snapshots
│   └── *.png                  # Validation charts
│
├── requirements.txt
└── README.md
```

---

## Key Config Parameters

```python
DATA_PROVIDER            = "yfinance"      # switch to "eikon" when ready
DATA_START               = "1995-01-01"    # BOCPD needs pre-2007 warmup
HMM_N_RESTARTS           = 50             # restarts to escape local optima
HMM_MIN_PERSISTENCE      = 20             # min days per regime
BOCPD_HAZARD_LAMBDA      = 60             # ~1 break per quarter
ENSEMBLE_TARGET_WINDOW   = 20             # predict -10% drawdown in 20 days
ENSEMBLE_DRAWDOWN_THRESH = -0.10
ENSEMBLE_TRAIN_CUTOFF    = "2018-01-01"
```

---

## Technical Notes

- Always run from **project root** so relative imports work
- **First run in a new environment:** `python data/cache.py --backfill` before pipeline
- Cache CSVs live in `data/cache/` — gitignored, must be regenerated per environment
- Activate venv: `source venv/bin/activate`
- `matplotlib.use("Agg")` set in all scripts — no GUI needed
- `hmmlearn` convergence warnings are **expected and harmless** — best restart wins
- `diagnose.py` confirms correct file versions when updates aren't being picked up
- `^STOXX50E` is in `EQUITY_TICKERS` but auto-dropped from BOCPD by the quality filter
- `risk_score_calibrated` may freeze for days — use `risk_score_raw` for tracking
- All features lag 1 day — today's BOCPD spike feeds tomorrow's risk score

---

## Environment

```
Python:       3.12.3
Key packages: hmmlearn==0.3.3, bayesian_changepoint_detection==0.2.dev1,
              xgboost==3.2.0, scikit-learn==1.8.0, pandas==3.0.2,
              numpy==2.4.4, yfinance==1.2.0
OS:           Linux (developed on Ubuntu / tested on local machine)
```
