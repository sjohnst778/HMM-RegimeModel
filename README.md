# HMM-RegimeModel

Early warning system for persistent market downturns using a blended
HMM + BOCPD architecture with an XGBoost ensemble scoring layer.

## Architecture

```
Raw market & macro data
        │
   ┌────┴────┐
   │         │
 HMM       BOCPD
(regime   (structural
 state)    breaks)
   │         │
   └────┬────┘
        │
  XGBoost ensemble
        │
  Downturn risk score [0, 1]
```

**HMM layer** — Gaussian HMM with 3 states (bull / bear / crisis) fitted on
daily returns and rolling volatility. States are re-ordered by mean return so
state 0 is always the worst-return (crisis) regime.

**BOCPD layer** — Bayesian Online Changepoint Detection applied to rolling
pairwise cross-asset correlations. Structural breaks in correlation regimes
often precede price-based regime transitions.

**Ensemble layer** — XGBoost classifier combining HMM state probabilities,
BOCPD changepoint signal, returns, vol, and macro features to predict
P(significant drawdown in next N days).

## Project Structure

```
HMM-RegimeModel/
├── data/
│   ├── fetch.py        # yfinance data download
│   └── features.py     # returns, vol, correlations, labels
├── models/
│   ├── hmm.py          # RegimeHMM class
│   ├── bocpd.py        # BOCPDetector class
│   └── ensemble.py     # DownturnEnsemble class
├── pipeline/
│   └── score.py        # end-to-end scoring pipeline
├── tests/
│   └── test_hmm.py
├── config.py           # all parameters in one place
└── requirements.txt
```

## Quickstart

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python pipeline/score.py

# Run tests
pytest tests/
```

## Configuration

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `HMM_N_STATES` | 3 | Number of HMM regimes |
| `BOCPD_HAZARD_LAMBDA` | 250 | Expected observations between changepoints |
| `ENSEMBLE_TARGET_WINDOW` | 20 | Forward window for drawdown prediction (days) |
| `ENSEMBLE_DRAWDOWN_THRESH` | -0.10 | Drawdown threshold for positive label |
| `ENSEMBLE_TRAIN_CUTOFF` | 2018-01-01 | Walk-forward train/test split date |

## Validation Notes

- Use `DownturnEnsemble.walk_forward_evaluate()` for time-series cross-validation
- Pay particular attention to performance on known crisis periods: 2000-02, 2008-09, 2020, 2022
- Average AUC is less important than recall on actual crisis events
