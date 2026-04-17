# pipeline/score.py
# End-to-end weekly scoring pipeline.
#
# Two modes:
#   TRAIN mode  — fits all models on historical data, saves bundle, scores full history
#   SCORE mode  — loads saved bundle, fetches latest data, appends new scores
#
# Usage:
#   python pipeline/score.py              # train + score (first run or retraining)
#   python pipeline/score.py --score-only # score only using saved bundle
#   python pipeline/score.py --tag eikon_v1  # load a specific model bundle

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch import fetch_equity_prices, fetch_macro_prices
from data.features import (
    build_hmm_features_v2,
    build_bocpd_features,
    build_ensemble_features,
    build_labels,
)
from models.hmm import RegimeHMM
from models.bocpd import BOCPDetector
from models.ensemble import DownturnEnsemble
from utils.persist import save_model_bundle, load_model_bundle
from config import (
    DATA_START, DATA_PROVIDER,
    HMM_N_STATES, HMM_BENCHMARK, HMM_MIN_PERSISTENCE,
    BOCPD_HAZARD_LAMBDA,
    ENSEMBLE_TARGET_WINDOW, ENSEMBLE_DRAWDOWN_THRESH, ENSEMBLE_TRAIN_CUTOFF,
    SCORE_OUTPUT_PATH,
)


def run_pipeline(score_only: bool = False, bundle_tag: str = "latest") -> pd.DataFrame:
    """
    Full end-to-end pipeline.

    Parameters
    ----------
    score_only : if True, load saved bundle instead of retraining
    bundle_tag : model bundle to load when score_only=True

    Returns
    -------
    pd.DataFrame with columns:
        date, hmm_state, hmm_p_crisis, bocpd_p_change,
        bocpd_break_freq_30d, downturn_risk_score, alert_p90, alert_p95
    """
    os.makedirs("output", exist_ok=True)

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    _log("Fetching data...")
    equity_prices = fetch_equity_prices(start=DATA_START)
    macro_prices  = fetch_macro_prices(start=DATA_START)
    benchmark     = equity_prices[HMM_BENCHMARK]
    rates         = macro_prices[["rates"]].rename(columns={"rates": "^TNX"}) \
                    if "rates" in macro_prices.columns else None

    _log(f"  Equity: {equity_prices.shape}  "
         f"{equity_prices.index[0].date()} -> {equity_prices.index[-1].date()}")

    # ── 2. HMM ────────────────────────────────────────────────────────────────
    _log("Running HMM...")
    hmm_features = build_hmm_features_v2(equity_prices, benchmark=HMM_BENCHMARK)

    if score_only:
        bundle   = load_model_bundle(bundle_tag)
        hmm      = bundle["hmm"]
        ensemble = bundle["ensemble"]
        _log(f"  Loaded bundle: {bundle_tag}  "
             f"(trained {bundle['metadata'].get('timestamp', 'unknown')})")
    else:
        hmm = RegimeHMM(n_states=HMM_N_STATES)
        hmm.fit(hmm_features)

    hmm_states = hmm.predict(hmm_features, min_persistence=HMM_MIN_PERSISTENCE)
    hmm_probs  = hmm.predict_proba(hmm_features)
    _log(f"  Regime today: {_regime_label(int(hmm_states.iloc[-1]))}  "
         f"(P(crisis)={hmm_probs['state_0'].iloc[-1]:.3f})")

    # ── 3. BOCPD ──────────────────────────────────────────────────────────────
    _log("Running BOCPD...")
    # Pass credit prices to BOCPD if available (Eikon only — CDX/iTraxx)
    credit_cols   = ["hy_spread", "ig_spread", "itraxx_main", "itraxx_xover"]
    credit_prices = macro_prices[[c for c in credit_cols if c in macro_prices.columns]]                     if any(c in macro_prices.columns for c in credit_cols) else None
    bocpd_input  = build_bocpd_features(
        equity_prices,
        rates_prices=rates,
        credit_prices=credit_prices,
    )
    detector     = BOCPDetector(hazard_lambda=BOCPD_HAZARD_LAMBDA)
    bocpd_df     = detector.run(bocpd_input)
    bocpd_signal = detector.composite_signal(bocpd_df)
    bocpd_freq   = detector.rolling_break_frequency(bocpd_signal)
    _log(f"  P(changepoint today): {bocpd_signal.iloc[-1]:.4f}  "
         f"Break freq 30d: {bocpd_freq.iloc[-1]:.0f}")

    # ── 4. Ensemble features ──────────────────────────────────────────────────
    _log("Building ensemble features...")
    from config import FEATURE_FLAGS
    ens_features = build_ensemble_features(
        equity_prices=equity_prices,
        macro_prices=macro_prices,
        hmm_states=hmm_states,
        hmm_probs=hmm_probs,
        bocpd_signal=bocpd_signal,
        bocpd_freq=bocpd_freq,
        feature_flags=FEATURE_FLAGS,
    )

    # ── 5. Train ensemble (if not score-only) ─────────────────────────────────
    if not score_only:
        _log(f"Training ensemble (cutoff: {ENSEMBLE_TRAIN_CUTOFF})...")
        labels = build_labels(
            benchmark,
            window=ENSEMBLE_TARGET_WINDOW,
            threshold=ENSEMBLE_DRAWDOWN_THRESH,
        )
        train_mask = ens_features.index < ENSEMBLE_TRAIN_CUTOFF
        aligned    = ens_features.join(labels).dropna()
        X_train    = aligned[ens_features.columns][aligned.index < ENSEMBLE_TRAIN_CUTOFF]
        y_train    = aligned[labels.name][aligned.index < ENSEMBLE_TRAIN_CUTOFF]

        ensemble = DownturnEnsemble()
        ensemble.fit(X_train, y_train)

        # Save bundle
        save_model_bundle(
            hmm=hmm,
            ensemble=ensemble,
            feature_names=list(ens_features.columns),
            tag=f"{DATA_PROVIDER}_v1",
            metadata={
                "data_source":     DATA_PROVIDER,
                "data_start":      DATA_START,
                "train_cutoff":    ENSEMBLE_TRAIN_CUTOFF,
                "label_window":    ENSEMBLE_TARGET_WINDOW,
                "label_threshold": ENSEMBLE_DRAWDOWN_THRESH,
                "positive_rate":   round(float(y_train.mean()), 4),
            },
        )

    # ── 6. Score ──────────────────────────────────────────────────────────────
    _log("Scoring...")
    scores_cal = ensemble.predict_proba(ens_features)      # calibrated (step function)
    scores_raw = ensemble.predict_proba_raw(ens_features)  # raw XGBoost (continuous)

    # Use RAW score for thresholds and alerts — varies continuously day-to-day
    # Calibrated score is retained for probability interpretation in reporting
    p90 = float(np.percentile(scores_raw.values, 90))
    p95 = float(np.percentile(scores_raw.values, 95))

    # ── 7. Assemble output ────────────────────────────────────────────────────
    output = pd.concat([
        hmm_states.rename("hmm_state"),
        hmm_probs["state_0"].rename("hmm_p_crisis"),
        bocpd_signal.rename("bocpd_p_change"),
        bocpd_freq.rename("bocpd_break_freq_30d"),
        scores_raw.rename("risk_score_raw"),        # primary: use for alerts + tracking
        scores_cal.rename("risk_score_calibrated"),  # secondary: use for probability reporting
    ], axis=1).dropna()

    output["alert_p90"] = (output["risk_score_raw"] > p90).astype(int)
    output["alert_p95"] = (output["risk_score_raw"] > p95).astype(int)
    output.index.name = "date"

    # ── 8. Save scores ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SCORE_OUTPUT_PATH), exist_ok=True)
    output.to_csv(SCORE_OUTPUT_PATH)
    _log(f"Scores saved to {SCORE_OUTPUT_PATH}  ({len(output)} rows)")

    # ── 9. Print latest reading ───────────────────────────────────────────────
    latest = output.iloc[-1]
    _log("=" * 55)
    _log(f"LATEST READING  {output.index[-1].date()}")
    _log("=" * 55)
    _log(f"  Risk score (raw)     : {latest['risk_score_raw']:.4f}  (day-to-day tracking)")
    _log(f"  Risk score (calib)   : {latest['risk_score_calibrated']:.4f}  (probability interpretation)")
    _log(f"  Alert (p90={p90:.3f}) : {'🔴 YES' if latest['alert_p90'] else '🟢 NO'}")
    _log(f"  Alert (p95={p95:.3f}) : {'🔴 YES' if latest['alert_p95'] else '🟢 NO'}")
    _log(f"  HMM regime          : {_regime_label(int(latest['hmm_state']))}")
    _log(f"  P(crisis)           : {latest['hmm_p_crisis']:.4f}")
    _log(f"  BOCPD P(break)      : {latest['bocpd_p_change']:.4f}")
    _log(f"  BOCPD break freq 30d: {latest['bocpd_break_freq_30d']:.0f}")
    _log("=" * 55)

    return output


def _regime_label(state: int) -> str:
    return {0: "CRISIS", 1: "BEAR", 2: "BULL"}.get(state, "UNKNOWN")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM-RegimeModel scoring pipeline")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Load saved model bundle instead of retraining (faster)",
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="Model bundle tag to load (default: latest)",
    )
    args = parser.parse_args()

    results = run_pipeline(score_only=args.score_only, bundle_tag=args.tag)
    print(f"\nTail of output:\n{results.tail(5).to_string()}")
