# config.py
# Central configuration for tickers, lookback windows, and model parameters.

# ── Data provider ──────────────────────────────────────────────────────────────
# 'yfinance' — free, development use
# 'eikon'    — Thomson Reuters Eikon, production use
DATA_PROVIDER = "yfinance"

# Start date — far enough back to give BOCPD warmup before GFC (2007)
DATA_START = "1995-01-01"

# ── Equity tickers ─────────────────────────────────────────────────────────────
EQUITY_TICKERS = [
    "^GSPC",      # S&P 500 (US)
    "^FTSE",      # FTSE 100 (UK)
    "^STOXX50E",  # Euro Stoxx 50 (Europe) -- pre-2007 yfinance data is poor, auto-dropped from BOCPD
    "^N225",      # Nikkei 225 (Japan)
    "^HSI",       # Hang Seng (Hong Kong/China proxy)
]
# Note: KOSPI (^KS11) and ASX 200 (^AXJO) are lower priority
# Add them here if you want broader Asian coverage
HMM_BENCHMARK  = "^GSPC"   # Single ticker for HMM features

# ── Macro tickers (yfinance) ───────────────────────────────────────────────────
# These are ETF proxies — available on yfinance but lower quality than Eikon
MACRO_TICKERS_YFINANCE = {
    "hy_spread":  "HYG",    # High yield proxy (ETF price, not spread)
    "ig_spread":  "LQD",    # Investment grade proxy (ETF price, not spread)
    "tips":       "TIP",    # Inflation expectations proxy
    "rates":      "^TNX",   # US 10y yield
    "vix":        "^VIX",   # VIX (1m implied vol)
    "yield_2y":   "^IRX",   # US 2y yield proxy (13-week T-bill as proxy)
}

# ── Macro tickers (Eikon RICs) ─────────────────────────────────────────────────
# Activated when DATA_PROVIDER = 'eikon'
# These are the actual instruments — proper spreads, not ETF proxies
MACRO_TICKERS_EIKON = {
    # Credit spreads (CDS indices) — Priority 1
    "hy_spread":      "CDXHY5Y=GFI",     # CDX HY 5y CDS spread (bps)
    "ig_spread":      "CDXIG5Y=GFI",     # CDX IG 5y CDS spread (bps)
    "itraxx_main":    "ITXEB5Y=GFI",     # iTraxx Main 5y (European IG)
    "itraxx_xover":   "ITXEXO5Y=GFI",   # iTraxx Crossover 5y (European HY)

    # Rates — Priority 3
    "rates":          "USBMK=RR",        # US 10y benchmark yield
    "yield_2y":       "US2YT=RR",        # US 2y yield
    "yield_3m":       "US3MT=RR",        # US 3m yield
    "tips":           "USBEI10Y=RR",     # 10y breakeven inflation

    # Vol surface — Priority 2
    "vix":            ".VIX",            # VIX 1m implied vol
    "vix3m":          ".VIX3M",          # VIX 3m implied vol
    "vvix":           ".VVIX",           # Vol of vol

    # Funding stress — Priority 5
    "ted_spread":     "TEDRATE=",        # TED spread (Libor vs T-bill)
    "fra_ois":        "FRAOIS3M=",       # 3m FRA-OIS spread
}

# Active macro tickers — switches automatically with DATA_PROVIDER
def get_macro_tickers() -> dict:
    if DATA_PROVIDER == "eikon":
        return MACRO_TICKERS_EIKON
    return MACRO_TICKERS_YFINANCE

MACRO_TICKERS = get_macro_tickers()

# ── Eikon credentials (fill in when switching) ────────────────────────────────
# EIKON_API_KEY = "your-api-key-here"  # or set env var EIKON_API_KEY
# Optional RIC overrides if your subscription uses different codes:
# EIKON_EQUITY_RICS = {"^GSPC": ".SPX", "^FTSE": ".FTSE"}

# ── Feature engineering ────────────────────────────────────────────────────────
RETURN_WINDOWS = [1, 5, 21]
VOL_WINDOW     = 21
CORR_WINDOW    = 21   # Shorter = more responsive to structural breaks

# ── Feature flags — controls which feature groups are built ───────────────────
# These are all ON when Eikon data is available, subset ON for yfinance
FEATURE_FLAGS = {
    # Always available
    "credit_etf_proxies":    True,   # HYG/LQD vol as credit stress proxy (yfinance)

    # Eikon-only features — set True when DATA_PROVIDER = 'eikon'
    "credit_spreads":        DATA_PROVIDER == "eikon",  # CDX/iTraxx spread levels + changes
    "vix_term_structure":    DATA_PROVIDER == "eikon",  # VIX/VIX3M slope, VVIX
    "yield_curve_shape":     DATA_PROVIDER == "eikon",  # 2s10s, 3m10y spreads, inversion flag
    "funding_stress":        DATA_PROVIDER == "eikon",  # TED spread, FRA-OIS
}

# ── HMM ───────────────────────────────────────────────────────────────────────
HMM_N_STATES       = 3
HMM_N_ITER         = 1000
HMM_COVARIANCE     = "full"
HMM_RANDOM_STATE   = 42
HMM_N_RESTARTS     = 50
HMM_MIN_PERSISTENCE = 20

# ── BOCPD ─────────────────────────────────────────────────────────────────────
BOCPD_HAZARD_LAMBDA = 60  # ~1 structural break per quarter

# ── Ensemble ──────────────────────────────────────────────────────────────────
ENSEMBLE_TARGET_WINDOW   = 20
ENSEMBLE_DRAWDOWN_THRESH = -0.10
ENSEMBLE_TRAIN_CUTOFF    = "2018-01-01"

XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
}

# ── Pipeline ──────────────────────────────────────────────────────────────────
SCORE_OUTPUT_PATH = "output/scores.csv"
