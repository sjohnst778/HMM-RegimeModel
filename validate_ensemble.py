# validate_ensemble.py
# Trains and validates the XGBoost ensemble layer.
# Run from project root: python validate_ensemble.py
#
# Produces:
#   output/ensemble_score.png        — risk score vs price history
#   output/ensemble_feature_imp.png  — SHAP / feature importance
#   output/ensemble_crisis_recall.txt — per-crisis recall table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

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
from utils.persist import save_model_bundle
from config import (
    DATA_START, HMM_N_STATES, HMM_BENCHMARK, HMM_MIN_PERSISTENCE,
    BOCPD_HAZARD_LAMBDA,
    ENSEMBLE_TARGET_WINDOW, ENSEMBLE_DRAWDOWN_THRESH, ENSEMBLE_TRAIN_CUTOFF,
)

CRISIS_BANDS = [
    ("2000-03-24", "2002-10-09", "Dot-com"),
    ("2007-10-09", "2009-03-09", "GFC"),
    ("2020-02-19", "2020-03-23", "COVID"),
    ("2022-01-03", "2022-10-12", "Rate hikes"),
]


def main():
    os.makedirs("output", exist_ok=True)

    # ── 1. Fetch data ──────────────────────────────────────────────────────────
    print("Fetching data...")
    equity_prices = fetch_equity_prices(start=DATA_START)
    macro_prices  = fetch_macro_prices(start=DATA_START)
    benchmark     = equity_prices[HMM_BENCHMARK]
    print(f"  Equity: {equity_prices.shape}  Macro: {macro_prices.shape}")

    # ── 2. HMM ────────────────────────────────────────────────────────────────
    print("\nFitting HMM...")
    hmm_features = build_hmm_features_v2(equity_prices, benchmark=HMM_BENCHMARK)
    hmm = RegimeHMM(n_states=HMM_N_STATES)
    hmm.fit(hmm_features)
    hmm_states = hmm.predict(hmm_features, min_persistence=HMM_MIN_PERSISTENCE)
    hmm_probs  = hmm.predict_proba(hmm_features)
    print(f"  HMM transitions: {int((hmm_states != hmm_states.shift()).sum()) - 1}")
    print(f"  Transition matrix:\n{hmm.transition_matrix().round(3)}")

    # ── 3. BOCPD ──────────────────────────────────────────────────────────────
    print("\nRunning BOCPD...")
    rates = macro_prices[["rates"]].rename(columns={"rates": "^TNX"})             if "rates" in macro_prices.columns else None
    credit_cols   = ["hy_spread", "ig_spread", "itraxx_main", "itraxx_xover"]
    credit_prices = macro_prices[[c for c in credit_cols if c in macro_prices.columns]]                     if any(c in macro_prices.columns for c in credit_cols) else None
    bocpd_input   = build_bocpd_features(
        equity_prices, rates_prices=rates, credit_prices=credit_prices,
    )
    detector      = BOCPDetector(hazard_lambda=BOCPD_HAZARD_LAMBDA)
    bocpd_df      = detector.run(bocpd_input)
    bocpd_signal  = detector.composite_signal(bocpd_df)
    bocpd_freq    = detector.rolling_break_frequency(bocpd_signal)
    print(f"  BOCPD date range: {bocpd_signal.index[0].date()} -> {bocpd_signal.index[-1].date()}")
    print(f"  Days P>0.5: {(bocpd_signal > 0.5).sum()} ({100*(bocpd_signal>0.5).mean():.1f}%)")

    # ── 4. Ensemble features ──────────────────────────────────────────────────
    print("\nBuilding ensemble features...")
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
    labels = build_labels(
        benchmark,
        window=ENSEMBLE_TARGET_WINDOW,
        threshold=ENSEMBLE_DRAWDOWN_THRESH,
    )

    print(f"  Feature matrix: {ens_features.shape}")
    print(f"  Label period:   {labels.index[0].date()} -> {labels.index[-1].date()}")
    print(f"  Positive rate:  {labels.mean():.1%}  ({labels.sum()} crisis days)")

    # Align
    aligned = ens_features.join(labels).dropna()
    X = aligned[ens_features.columns]
    y = aligned[labels.name]
    print(f"  Aligned shape:  {X.shape}")

    # ── 5. Walk-forward evaluation ────────────────────────────────────────────
    print(f"\nWalk-forward evaluation (train cutoff: {ENSEMBLE_TRAIN_CUTOFF})...")
    ensemble = DownturnEnsemble()
    cv_results = ensemble.walk_forward_evaluate(X, y, min_train_years=5, n_splits=5)
    print(f"\n  Mean AUC: {cv_results['mean_auc']:.3f}  ±  {cv_results['std_auc']:.3f}")

    # ── 6. Fit on full pre-cutoff history ─────────────────────────────────────
    print(f"\nFitting on data before {ENSEMBLE_TRAIN_CUTOFF}...")
    train_mask = X.index < ENSEMBLE_TRAIN_CUTOFF
    ensemble.fit(X[train_mask], y[train_mask])

    # ── 7. Score full history ─────────────────────────────────────────────────
    scores_cal = ensemble.predict_proba(X)
    scores_raw = ensemble.predict_proba_raw(X)
    scores = scores_raw  # use raw for thresholds and recall — varies continuously
    print(f"\nLatest risk score raw  ({scores_raw.index[-1].date()}): {scores_raw.iloc[-1]:.4f}")
    print(f"Latest risk score calib ({scores_cal.index[-1].date()}): {scores_cal.iloc[-1]:.4f}")
    print(f"(Calibrated score may be frozen if in isotonic plateau — use raw for tracking)")

    # ── 8. Threshold analysis + crisis recall ────────────────────────────────
    print("\n── Score Distribution ───────────────────────────────────")
    for pct in [50, 75, 90, 92, 95, 99]:
        val = np.percentile(scores.values, pct)
        print(f"  {pct}th percentile: {val:.4f}")

    # Use the 90th percentile as threshold — flags top 10% of risk days
    # More meaningful than 0.5 for a calibrated model with 8% positive rate
    p90_threshold = np.percentile(scores.values, 90)
    p95_threshold = np.percentile(scores.values, 95)
    print(f"\n  Using p90 threshold = {p90_threshold:.4f}  (flags ~10% of days)")
    print(f"  Using p95 threshold = {p95_threshold:.4f}  (flags ~5% of days)")

    print("\n── Crisis Recall (p90 threshold) ────────────────────────")
    _crisis_recall(scores, CRISIS_BANDS, threshold=p90_threshold)
    print("\n── Crisis Recall (p95 threshold) ────────────────────────")
    _crisis_recall(scores, CRISIS_BANDS, threshold=p95_threshold)

    # ── 9. Feature importance ─────────────────────────────────────────────────
    print("\n── Top 15 Features ──────────────────────────────────────")
    imp = ensemble.feature_importance()
    for feat, val in imp.head(15).items():
        bar = "█" * int(val * 200)
        print(f"  {feat:40s} {val:.4f}  {bar}")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    _plot_score(benchmark, scores, hmm_states, y, p90_threshold, p95_threshold)
    _plot_feature_importance(imp)

    # ── 11. Save model bundle ──────────────────────────────────────────────────
    from config import DATA_PROVIDER
    save_model_bundle(
        hmm=hmm,
        ensemble=ensemble,
        feature_names=list(ens_features.columns),
        tag=f"{DATA_PROVIDER}_v1",
        metadata={
            "data_source":       DATA_PROVIDER,
            "data_start":        DATA_START,
            "train_cutoff":      ENSEMBLE_TRAIN_CUTOFF,
            "mean_auc":          round(cv_results["mean_auc"], 4),
            "std_auc":           round(cv_results["std_auc"], 4),
            "label_window":      ENSEMBLE_TARGET_WINDOW,
            "label_threshold":   ENSEMBLE_DRAWDOWN_THRESH,
            "positive_rate":     round(float(y.mean()), 4),
            "n_train_days":      int((X.index < ENSEMBLE_TRAIN_CUTOFF).sum()),
            "hmm_transitions":   int((hmm_states != hmm_states.shift()).sum()) - 1,
            "bocpd_days_flagged": int((bocpd_signal > 0.5).sum()),
        },
    )

    print("\nDone. Charts saved to output/")


def _crisis_recall(scores, crisis_bands, threshold=0.5):
    """For each crisis, report: max score, days above threshold, first alert date."""
    for start, end, name in crisis_bands:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s < scores.index[0]:
            continue
        window = scores.loc[s:e]
        if len(window) == 0:
            continue
        max_score  = window.max()
        days_above = int((window > threshold).sum())
        status     = "✓" if days_above > 0 else "✗"

        if days_above > 0:
            first_alert = window[window > threshold].index[0]
            lead_days   = (s - first_alert).days
            lead_str    = f"lead={lead_days:+d}d" if lead_days != 0 else "on crisis date"
            print(f"  {status} {name:14s}: max={max_score:.4f}  "
                  f"days_above={days_above:3d}  "
                  f"first_alert={first_alert.date()}  {lead_str}")
        else:
            print(f"  {status} {name:14s}: max={max_score:.4f}  "
                  f"days_above={days_above:3d}  no alert")


def _plot_score(benchmark, scores, hmm_states, labels, p90=0.1, p95=0.05):
    """Three-panel: price with crisis bands, risk score, label."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 0.5]})
    fig.suptitle("Ensemble Downturn Risk Score", fontsize=13, fontweight="bold")

    colours = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}

    # Panel 1: Price + crisis bands
    ax0 = axes[0]
    ax0.plot(benchmark.index, benchmark.values, color="black", linewidth=0.8, zorder=3)
    ax0.set_yscale("log")
    ax0.set_ylabel(HMM_BENCHMARK)
    for start, end, name in CRISIS_BANDS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s >= benchmark.index[0]:
            ax0.axvspan(s, e, alpha=0.12, color="red", zorder=1)
            ax0.text(s + (e-s)/2, benchmark.quantile(0.12), name,
                     ha="center", fontsize=7, color="darkred", rotation=90)

    # Panel 2: Risk score
    ax1 = axes[1]
    ax1.fill_between(scores.index, scores.values, color="#d62728", alpha=0.6)
    ax1.axhline(p90, color="orange", linestyle="--", linewidth=0.8, label=f"p90={p90:.3f}")
    ax1.axhline(p95, color="red",    linestyle=":",  linewidth=0.8, label=f"p95={p95:.3f}")
    ax1.set_ylabel("Risk Score")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left", fontsize=8)
    for start, end, _ in CRISIS_BANDS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s >= scores.index[0]:
            ax1.axvspan(s, e, alpha=0.08, color="red", zorder=0)

    # Panel 3: Actual label (was there a significant drawdown?)
    ax2 = axes[2]
    ax2.fill_between(labels.index, labels.values.astype(float),
                     color="#d62728", alpha=0.7, step="mid")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["no", "yes"], fontsize=7)
    ax2.set_ylabel(f">{abs(ENSEMBLE_DRAWDOWN_THRESH):.0%}\ndrawdown")

    plt.tight_layout()
    plt.savefig("output/ensemble_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: output/ensemble_score.png")


def _plot_feature_importance(imp):
    """Horizontal bar chart of top 20 features."""
    top = imp.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    colours = ["#d62728" if "hmm" in f else
               "#1f77b4" if "bocpd" in f else
               "#2ca02c" for f in top.index]
    ax.barh(range(len(top)), top.values, color=colours, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Ensemble Feature Importance\n(red=HMM, blue=BOCPD, green=market)")

    patches = [
        mpatches.Patch(color="#d62728", label="HMM"),
        mpatches.Patch(color="#1f77b4", label="BOCPD"),
        mpatches.Patch(color="#2ca02c", label="Market"),
    ]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    plt.savefig("output/ensemble_feature_imp.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Chart saved: output/ensemble_feature_imp.png")


if __name__ == "__main__":
    main()
