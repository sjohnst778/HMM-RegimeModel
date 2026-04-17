# data/features.py
# Feature engineering: returns, volatility, rolling correlations,
# and the composite feature matrix fed into the HMM and BOCPD layers.

import numpy as np
import pandas as pd
from config import RETURN_WINDOWS, VOL_WINDOW, CORR_WINDOW


def compute_returns(prices: pd.DataFrame, windows: list[int] = RETURN_WINDOWS) -> pd.DataFrame:
    """
    Compute log returns over multiple rolling windows for each ticker.

    Returns
    -------
    pd.DataFrame
        Columns: '{ticker}_ret_{window}d' for each ticker x window combination.
    """
    log_prices = np.log(prices)
    frames = []
    for window in windows:
        rets = log_prices.diff(window)
        rets.columns = [f"{col}_ret_{window}d" for col in rets.columns]
        frames.append(rets)
    return pd.concat(frames, axis=1)


def compute_volatility(prices: pd.DataFrame, window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Compute rolling annualised volatility (std of daily log returns).

    Returns
    -------
    pd.DataFrame
        Columns: '{ticker}_vol_{window}d'
    """
    daily_rets = np.log(prices).diff()
    vol = daily_rets.rolling(window).std() * np.sqrt(252)
    vol.columns = [f"{col}_vol_{window}d" for col in vol.columns]
    return vol


def compute_rolling_correlations(prices: pd.DataFrame, window: int = CORR_WINDOW) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations between all tickers.
    These are fed into BOCPD to detect structural breaks in correlation regime.

    Returns
    -------
    pd.DataFrame
        Columns: 'corr_{ticker_a}_{ticker_b}' for each unique pair.
    """
    daily_rets = np.log(prices).diff()
    tickers = prices.columns.tolist()
    pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i + 1:]]

    records = {}
    for a, b in pairs:
        col = f"corr_{a}_{b}"
        records[col] = daily_rets[a].rolling(window).corr(daily_rets[b])

    return pd.DataFrame(records, index=prices.index)


def build_hmm_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for HMM fitting.
    Uses 1-day returns and rolling volatility as the observation sequence.

    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix (no NaNs), index aligned to prices.
    """
    rets = compute_returns(prices, windows=[1])
    vols = compute_volatility(prices)
    features = pd.concat([rets, vols], axis=1).dropna()
    return features


def build_bocpd_features(
    equity_prices: pd.DataFrame,
    rates_prices: pd.DataFrame | None = None,
    credit_prices: pd.DataFrame | None = None,
    window: int = CORR_WINDOW,
) -> pd.DataFrame:
    """
    Build the feature matrix for BOCPD using equity tickers only.

    Deliberately uses equity_prices (not macro) to preserve the long history
    needed for BOCPD warmup before the GFC (2007). Macro ETFs like HYG only
    launched in 2007, so including them truncates the series via dropna().

    Monitors three complementary signal types per equity ticker:
      1. Rolling pairwise correlations  — cross-market contagion signal
      2. Rolling annualised volatility  — vol regime break
      3. Rolling mean return            — trend/drift break

    Parameters
    ----------
    equity_prices : pd.DataFrame — equity index prices only
    rates_prices  : pd.DataFrame, optional — rates series (e.g. ^TNX).
                    Used for equity/rates cross-correlations (flight-to-safety signal).
                    Aligned to equity dates — no truncation of equity history.
    credit_prices : pd.DataFrame, optional — CDS spread series (Eikon: CDX/iTraxx).
                    Used as direct BOCPD input series — breaks in credit spread
                    behaviour fire ahead of equity crisis by 4-12 weeks.
                    Aligned to equity dates — no truncation.
    window        : int — rolling window in trading days (default CORR_WINDOW=21)

    Returns
    -------
    pd.DataFrame, NaNs dropped. Starts ~window days after equity data begins.
    """
    daily_rets = np.log(equity_prices).diff()

    # Drop any ticker whose price series has long constant runs — rolling
    # correlations against a zero-variance segment are undefined (NaN) and
    # will cause dropna() to truncate the entire history. ^STOXX50E has
    # 3000+ repeated-price runs in the pre-2007 yfinance data.
    MIN_UNIQUE_RATIO = 0.90  # at least 90% of returns must be non-zero
    clean_tickers = []
    for col in equity_prices.columns:
        nonzero_ratio = (daily_rets[col].dropna() != 0).mean()
        if nonzero_ratio >= MIN_UNIQUE_RATIO:
            clean_tickers.append(col)
        else:
            print(f"  Dropping {col} from BOCPD features "
                  f"(only {100*nonzero_ratio:.1f}% non-zero returns — "
                  f"constant price runs cause undefined correlations)")

    daily_rets = daily_rets[clean_tickers]
    tickers = clean_tickers
    series = {}

    # 1. Rolling pairwise correlations
    pairs = [(a, b) for i, a in enumerate(tickers) for b in tickers[i + 1:]]
    for a, b in pairs:
        series[f"corr_{a}_{b}"] = daily_rets[a].rolling(window).corr(daily_rets[b])

    # 2. Rolling annualised volatility per index
    for col in tickers:
        series[f"vol_{col}"] = daily_rets[col].rolling(window).std() * np.sqrt(252)

    # 3. Rolling mean return per index (drift break signal)
    for col in tickers:
        series[f"mean_ret_{col}"] = daily_rets[col].rolling(window).mean()

    # 4. Cross-asset correlations: equity vs rates
    # Equity/rates correlation inverts during flight-to-safety crises.
    # Joined separately so shorter history doesn't truncate equity history.
    if rates_prices is not None:
        rates_rets = np.log(rates_prices.clip(lower=1e-6)).diff()
        rates_rets = rates_rets.reindex(equity_prices.index).ffill()
        for eq_col in tickers:
            for r_col in rates_rets.columns:
                cross_corr = daily_rets[eq_col].rolling(window).corr(rates_rets[r_col])
                series[f"corr_{eq_col}_{r_col}"] = cross_corr

    # 5. Credit spread series as BOCPD inputs (Eikon only)
    # CDS spread changes are structural break signals that fire ahead of equity.
    # Joined separately to avoid truncating pre-Eikon history.
    if credit_prices is not None:
        # Use spread changes (not levels) as BOCPD observations —
        # BOCPD detects breaks in the *behaviour* of the series
        credit_rets = credit_prices.diff()  # spread changes in bps, not log returns
        credit_rets = credit_rets.reindex(equity_prices.index).ffill()
        for c_col in credit_rets.columns:
            # Rolling mean of spread changes — structural break when mean shifts
            series[f"credit_chg_{c_col}"] = credit_rets[c_col].rolling(window).mean()
            # Rolling vol of spread changes — vol regime break in credit
            series[f"credit_vol_{c_col}"] = credit_rets[c_col].rolling(window).std()

    df = pd.DataFrame(series, index=equity_prices.index).dropna()
    return df


def build_ensemble_features(
    equity_prices: pd.DataFrame,
    macro_prices: pd.DataFrame,
    hmm_states: pd.Series,
    hmm_probs: pd.DataFrame,
    bocpd_signal: pd.Series,
    bocpd_freq: pd.Series,
    feature_flags: dict | None = None,
) -> pd.DataFrame:
    """
    Build the composite feature matrix for the XGBoost ensemble layer.

    Combines four groups of features:
      1. HMM outputs — state probabilities, current state, regime duration
      2. BOCPD outputs — composite changepoint probability, 30d break frequency
      3. Market features — multi-window returns, vol, correlations on benchmark
      4. Macro features — HY spread proxy, rates, TIPS (from macro_prices)

    All features are lagged by 1 day to prevent look-ahead bias:
    the model predicts tomorrow's drawdown using only today's information.

    Parameters
    ----------
    equity_prices : equity price DataFrame
    macro_prices  : macro price DataFrame (HY, IG, TIPS, rates)
    hmm_states    : pd.Series of integer regime labels
    hmm_probs     : pd.DataFrame of state probabilities (state_0, state_1, state_2)
    bocpd_signal  : pd.Series of composite P(changepoint)
    bocpd_freq    : pd.Series of 30d rolling break frequency

    Returns
    -------
    pd.DataFrame, feature matrix aligned on date index, NaNs dropped.
    """
    # Resolve feature flags — defaults to config, can be overridden for testing
    if feature_flags is None:
        try:
            from config import FEATURE_FLAGS
            feature_flags = FEATURE_FLAGS
        except ImportError:
            feature_flags = {}
    flags = feature_flags

    frames = []

    # --- Group 1: HMM features ---
    hmm_df = hmm_probs.copy()
    hmm_df.columns = [f"hmm_p_{c}" for c in hmm_probs.columns]  # hmm_p_state_0 etc
    # Regime duration: how many consecutive days in current state
    run_id = (hmm_states != hmm_states.shift()).cumsum()
    hmm_df["hmm_state"] = hmm_states
    hmm_df["hmm_duration"] = run_id.groupby(run_id).cumcount() + 1
    frames.append(hmm_df)

    # --- Group 2: BOCPD features ---
    bocpd_df = pd.concat([
        bocpd_signal.rename("bocpd_p_change"),
        bocpd_freq.rename("bocpd_break_freq_30d"),
    ], axis=1)
    frames.append(bocpd_df)

    # --- Group 3: Market features (benchmark = first equity column) ---
    benchmark_col = equity_prices.columns[0]
    px = equity_prices[benchmark_col]
    log_ret = np.log(px).diff()

    market = pd.DataFrame(index=equity_prices.index)
    for w in [1, 5, 21, 63]:
        market[f"ret_{w}d"] = log_ret.rolling(w).sum()
    for w in [5, 21, 63]:  # vol needs min 2 obs; skip window=1
        market[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)
    # Drawdown from 252d high
    market["drawdown_252d"] = (px / px.rolling(252, min_periods=21).max() - 1)
    frames.append(market)

    # --- Group 4: Macro features (always-on) ---
    # For yfinance: ETF price vol as credit stress proxies
    # For Eikon: replaced by proper spread levels in Groups 5-8 below
    if macro_prices is not None and len(macro_prices):
        macro_rets = np.log(macro_prices.clip(lower=1e-6)).diff()
        for col in macro_prices.columns:
            frames.append(macro_rets[[col]].rename(columns={col: f"macro_ret_{col}"}))
            frames.append(
                macro_rets[col].rolling(21).std().rename(f"macro_vol_{col}").to_frame()
            )

    # --- Group 5: Credit spreads (Eikon only) ---
    # Priority 1 enhancement — proper CDS spread levels and changes
    # Detects GFC/COVID credit stress weeks before equity falls
    if macro_prices is not None and flags.get("credit_spreads"):
        cs = pd.DataFrame(index=equity_prices.index)
        for col in ["hy_spread", "ig_spread", "itraxx_main", "itraxx_xover"]:
            if col in macro_prices.columns:
                cs[f"cs_{col}_level"]   = macro_prices[col]
                cs[f"cs_{col}_chg_5d"]  = macro_prices[col].diff(5)
                cs[f"cs_{col}_chg_21d"] = macro_prices[col].diff(21)
        # HY/IG ratio — widens when stress is credit-specific not just rates
        if "hy_spread" in macro_prices.columns and "ig_spread" in macro_prices.columns:
            cs["cs_hy_ig_ratio"] = (
                macro_prices["hy_spread"] / macro_prices["ig_spread"].clip(lower=0.01)
            )
        frames.append(cs)

    # --- Group 6: VIX term structure (Eikon only) ---
    # Priority 2 enhancement — inverted VIX curve is a leading crisis signal
    # VIX3M > VIX = normal. VIX > VIX3M = fear concentrated near-term = stress
    if macro_prices is not None and flags.get("vix_term_structure"):
        vts = pd.DataFrame(index=equity_prices.index)
        if "vix" in macro_prices.columns:
            vts["vix_level"]      = macro_prices["vix"]
            vts["vix_chg_5d"]     = macro_prices["vix"].diff(5)
            # Risk premium: implied vol / realised vol. Compression precedes crises
            vts["vix_rp"] = macro_prices["vix"] / (
                np.log(equity_prices[equity_prices.columns[0]]).diff()
                .rolling(21).std() * np.sqrt(252) * 100
            ).clip(lower=1)
        if "vix3m" in macro_prices.columns and "vix" in macro_prices.columns:
            # Negative = inverted = near-term fear > long-term uncertainty
            vts["vix_ts_slope"]   = macro_prices["vix"] - macro_prices["vix3m"]
            vts["vix_ts_inverted"] = (vts["vix_ts_slope"] > 0).astype(int)
        if "vvix" in macro_prices.columns:
            vts["vvix_level"] = macro_prices["vvix"]
            vts["vvix_chg_5d"] = macro_prices["vvix"].diff(5)
        frames.append(vts)

    # --- Group 7: Yield curve shape (Eikon only) ---
    # Priority 3 enhancement — inversion is a lagging but high-precision crisis signal
    if macro_prices is not None and flags.get("yield_curve_shape"):
        yc = pd.DataFrame(index=equity_prices.index)
        if "rates" in macro_prices.columns and "yield_2y" in macro_prices.columns:
            yc["yc_2s10s"]     = macro_prices["rates"] - macro_prices["yield_2y"]
            yc["yc_inverted"]  = (yc["yc_2s10s"] < 0).astype(int)
            # Days since inversion started (0 if not inverted)
            inv = yc["yc_inverted"]
            yc["yc_inv_duration"] = inv.groupby(
                (inv != inv.shift()).cumsum()
            ).cumcount() * inv
        if "rates" in macro_prices.columns and "yield_3m" in macro_prices.columns:
            yc["yc_3m10y"]    = macro_prices["rates"] - macro_prices["yield_3m"]
            yc["yc_3m_inv"]   = (yc["yc_3m10y"] < 0).astype(int)
        frames.append(yc)

    # --- Group 8: Funding stress (Eikon only) ---
    # Priority 5 enhancement — bank funding stress, leads equity in credit crises
    if macro_prices is not None and flags.get("funding_stress"):
        fs = pd.DataFrame(index=equity_prices.index)
        if "ted_spread" in macro_prices.columns:
            fs["ted_level"]   = macro_prices["ted_spread"]
            fs["ted_chg_21d"] = macro_prices["ted_spread"].diff(21)
        if "fra_ois" in macro_prices.columns:
            fs["fra_ois_level"]   = macro_prices["fra_ois"]
            fs["fra_ois_chg_21d"] = macro_prices["fra_ois"].diff(21)
        frames.append(fs)

    # Concatenate and apply 1-day lag to prevent look-ahead bias
    features = pd.concat(frames, axis=1)
    features = features.shift(1)   # <-- critical: use yesterday's signals
    features = features.dropna()
    return features


def build_labels(px: pd.Series, window: int, threshold: float) -> pd.Series:
    """
    Build binary labels: 1 if the minimum return over the NEXT `window` trading
    days falls below `threshold`.

    Uses vectorised rolling min for speed (avoids slow apply).

    Parameters
    ----------
    px        : pd.Series — benchmark price series
    window    : int — forward-looking window in trading days
    threshold : float — e.g. -0.10 for -10% drawdown

    Returns
    -------
    pd.Series of {0, 1}, named 'crisis_label'.
    NaN for the last `window` rows (no future data available).
    """
    log_px = np.log(px)
    # For each day t, find the min log-return over [t+1, t+window]
    # = rolling min of log_px shifted forward, minus log_px[t]
    fwd_min_logpx = log_px[::-1].rolling(window, min_periods=window).min()[::-1].shift(-(window))
    fwd_min_ret   = fwd_min_logpx - log_px   # log return to trough

    labels = (fwd_min_ret < np.log(1 + threshold)).astype(float)
    labels[labels == 0] = 0  # explicit
    labels.name = "crisis_label"
    # NaN out the last `window` rows where we have no future
    labels.iloc[-window:] = np.nan
    return labels


def build_hmm_features_benchmark(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Build a compact 2-feature HMM observation matrix using only a single
    benchmark series (e.g. S&P 500). Keeping the observation space small
    (just return + vol) avoids the regime-switching noise that arises when
    too many correlated features are fed into the HMM.

    Features:
        - 1-day log return
        - 21-day rolling annualised volatility

    Parameters
    ----------
    prices    : full price DataFrame (multi-ticker)
    benchmark : column name of the benchmark ticker (e.g. "^GSPC")

    Returns
    -------
    pd.DataFrame with 2 columns, no NaNs.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in prices columns: {list(prices.columns)}")

    px = prices[[benchmark]]
    ret = compute_returns(px, windows=[1])
    vol = compute_volatility(px)
    features = pd.concat([ret, vol], axis=1).dropna()
    features.columns = ["ret_1d", "vol_21d"]
    return features


def build_hmm_features_v2(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """
    Improved 3-feature HMM observation matrix that better separates
    bear from crisis regimes by adding drawdown from peak.

    Features:
        - 1-day log return         (captures daily vol regime)
        - 21-day rolling ann. vol  (captures vol level)
        - drawdown from 252d high  (captures sustained loss regime)

    The drawdown feature is key: crisis has deep drawdown AND high vol,
    bear has moderate drawdown AND moderate vol, bull has near-zero drawdown.
    This triangular separation prevents bear/crisis oscillation.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' not in columns: {list(prices.columns)}")

    px = prices[benchmark]
    log_px = np.log(px)

    ret_1d = log_px.diff().rename("ret_1d")
    vol_21d = ret_1d.rolling(21).std() * np.sqrt(252)
    vol_21d.name = "vol_21d"

    # Drawdown from rolling 252-day high (negative number, range [-inf, 0])
    rolling_high = px.rolling(252, min_periods=21).max()
    drawdown = (px / rolling_high - 1.0).rename("drawdown_252d")

    features = pd.concat([ret_1d, vol_21d, drawdown], axis=1).dropna()
    return features
