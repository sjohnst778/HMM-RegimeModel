# validate_bocpd.py
# Validates the BOCPD layer against known crisis periods.
# Run from project root: python validate_bocpd.py
#
# Key checks:
#   1. Does BOCPD fire AHEAD of HMM crisis transitions? (leading indicator test)
#   2. Is the signal clean enough outside crisis periods? (false positive rate)
#   3. Which correlation pairs are most informative?

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

from data.fetch import fetch_equity_prices, fetch_macro_prices
from data.features import build_bocpd_features, build_hmm_features_v2
from models.bocpd import BOCPDetector
from models.hmm import RegimeHMM
from config import (
    HMM_N_STATES, HMM_BENCHMARK, HMM_MIN_PERSISTENCE,
    BOCPD_HAZARD_LAMBDA, DATA_START,
)

# Known crisis entry dates for lead/lag analysis
CRISIS_ENTRIES = {
    "Dot-com":    pd.Timestamp("2000-03-24"),
    "GFC":        pd.Timestamp("2007-10-09"),
    "COVID":      pd.Timestamp("2020-02-19"),
    "Rate hikes": pd.Timestamp("2022-01-03"),
}

BOCPD_THRESHOLD = 0.5   # P(changepoint) threshold for signal firing


def main():
    print(f"[CONFIG CHECK] DATA_START = {DATA_START}")
    os.makedirs("output", exist_ok=True)

    # --- Data ---
    print("Fetching data...")
    prices = fetch_equity_prices(start=DATA_START)  # DATA_START = 1995-01-01
    print(f"  Data start: {DATA_START}  (need pre-2000 for GFC warmup)")
    benchmark = prices[HMM_BENCHMARK]

    # --- HMM (for comparison) ---
    print("Fitting HMM...")
    hmm_features = build_hmm_features_v2(prices, benchmark=HMM_BENCHMARK)
    hmm = RegimeHMM(n_states=HMM_N_STATES)
    hmm.fit(hmm_features)
    hmm_states = hmm.predict(hmm_features, min_persistence=HMM_MIN_PERSISTENCE)
    hmm_probs  = hmm.predict_proba(hmm_features)

    # --- BOCPD ---
    print(f"\nRunning BOCPD (hazard_lambda={BOCPD_HAZARD_LAMBDA})...")
    # Fetch rates separately — shorter history but used only for cross-asset correlation
    # reindex() aligns to equity dates so no truncation of pre-2007 history
    macro = fetch_macro_prices(start=DATA_START)
    rates = macro[["rates"]].rename(columns={"rates": "^TNX"}) if "rates" in macro.columns else None
    if rates is not None:
        print(f"  Rates (^TNX) available from: {rates.first_valid_index().date()}")
    bocpd_features = build_bocpd_features(prices, rates_prices=rates)
    n_corr = sum(1 for c in bocpd_features.columns if c.startswith('corr_'))
    n_vol  = sum(1 for c in bocpd_features.columns if c.startswith('vol_'))
    n_ret  = sum(1 for c in bocpd_features.columns if c.startswith('mean_ret_'))
    print(f"  {bocpd_features.shape[1]} series total: {n_corr} correlations, {n_vol} vol, {n_ret} mean-return")
    print(f"  Columns: {list(bocpd_features.columns)}")

    detector  = BOCPDetector()
    bocpd_df  = detector.run(bocpd_features)
    signal    = detector.composite_signal(bocpd_df)
    mean_sig  = detector.mean_signal(bocpd_df)
    break_freq = detector.rolling_break_frequency(signal, window=30)

    # --- Console diagnostics ---
    print("\n-- BOCPD Signal Summary --")
    print(f"  Date range:  {signal.index[0].date()} -> {signal.index[-1].date()}")
    print(f"  Days > {BOCPD_THRESHOLD}: {(signal > BOCPD_THRESHOLD).sum()} "
          f"({100*(signal > BOCPD_THRESHOLD).mean():.1f}% of all days)")
    print(f"  Signal mean: {signal.mean():.4f}  max: {signal.max():.4f}")

    print("\n-- Lead/Lag vs HMM Crisis Transitions --")
    min_reliable = signal.index[0] + pd.Timedelta(days=180)
    print(f"  (First reliable date after warmup: {min_reliable.date()})")
    _lead_lag_analysis(signal, hmm_states, hmm_probs)

    # --- Parameter sweep ---
    print("\n-- Lambda Sensitivity Sweep (CORR_WINDOW=21) --")
    print(f"  {'Lambda':>8}  {'Days>0.5':>10}  {'GFC':>6}  {'COVID':>7}  {'2022':>6}")
    for lam in [30, 60, 90, 120, 180, 250]:
        d2   = BOCPDetector(hazard_lambda=lam)
        bdf2 = d2.run(bocpd_features)
        sig2 = d2.composite_signal(bdf2)
        pct  = 100 * (sig2 > BOCPD_THRESHOLD).mean()
        def hit(name):
            if name not in CRISIS_ENTRIES: return "N/A"
            cd = CRISIS_ENTRIES[name]
            if cd < sig2.index[0]: return "N/A"
            w = sig2.loc[cd - pd.Timedelta(days=90): cd + pd.Timedelta(days=30)]
            return "YES" if (w > BOCPD_THRESHOLD).any() else "NO"
        print(f"  {lam:>8}  {pct:>9.1f}%  {hit('GFC'):>6}  {hit('COVID'):>7}  {hit('Rate hikes'):>6}")
    print(f"  Using lambda={BOCPD_HAZARD_LAMBDA} for main analysis")

    print("\n-- Top 15 Signal Dates --")
    for dt, val in signal.nlargest(15).items():
        print(f"  {str(dt.date()):12s}  p={val:.4f}")

    print("\n-- Per-Series Breakdown --")
    for col in bocpd_df.columns:
        s = bocpd_df[col]
        pct = 100 * (s > BOCPD_THRESHOLD).mean()
        print(f"  {col:35s}: {pct:.1f}% days above threshold")

    # --- Plots ---
    _plot_main(benchmark, signal, mean_sig, hmm_states, hmm_probs)
    _plot_per_series(bocpd_df, benchmark)
    _plot_break_frequency(break_freq, benchmark, hmm_states)

    print("\nCharts saved to output/")


def _lead_lag_analysis(signal, hmm_states, hmm_probs):
    """
    For each known crisis entry, find:
    - When did HMM P(crisis) first exceed 0.5?
    - When did BOCPD first exceed threshold in the 60 days before/after?
    - Is BOCPD leading or lagging?
    """
    p_crisis = hmm_probs["state_0"]

    for name, entry_date in CRISIS_ENTRIES.items():
        if entry_date < signal.index[0]:
            continue

        window_start = entry_date - pd.Timedelta(days=90)
        window_end   = entry_date + pd.Timedelta(days=90)

        # HMM: first day P(crisis) > 0.5 in window
        hmm_window = p_crisis.loc[window_start:window_end]
        hmm_fire   = hmm_window[hmm_window > 0.5].index
        hmm_date   = hmm_fire[0] if len(hmm_fire) else None

        # BOCPD: first day signal > threshold in window
        bocpd_window = signal.loc[window_start:window_end]
        bocpd_fire   = bocpd_window[bocpd_window > BOCPD_THRESHOLD].index
        bocpd_date   = bocpd_fire[0] if len(bocpd_fire) else None

        if hmm_date and bocpd_date:
            lead = (hmm_date - bocpd_date).days
            leader = "BOCPD leads by" if lead > 0 else "HMM leads by"
            print(f"  {name:12s}: market peak {entry_date.date()} | "
                  f"BOCPD fired {bocpd_date.date()} | "
                  f"HMM fired {hmm_date.date()} | "
                  f"{leader} {abs(lead)} days")
        elif bocpd_date:
            print(f"  {name:12s}: BOCPD fired {bocpd_date.date()} | HMM did not fire in window")
        elif hmm_date:
            print(f"  {name:12s}: HMM fired {hmm_date.date()} | BOCPD did not fire in window")
        else:
            print(f"  {name:12s}: neither fired in window")


def _plot_main(benchmark, signal, mean_sig, hmm_states, hmm_probs):
    """Four-panel chart: price, BOCPD composite, HMM P(crisis), regime bar."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 0.5]})
    fig.suptitle("BOCPD vs HMM — Structural Break Detection", fontsize=13, fontweight="bold")

    # Panel 1: Price
    ax0 = axes[0]
    ax0.plot(benchmark.index, benchmark.values, color="black", linewidth=0.8)
    ax0.set_yscale("log")
    ax0.set_ylabel(HMM_BENCHMARK)
    for name, date in CRISIS_ENTRIES.items():
        if date >= benchmark.index[0]:
            ax0.axvline(date, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
            ax0.text(date, benchmark.max() * 0.6, name, fontsize=7,
                     color="red", rotation=90, va="top")

    # Panel 2: BOCPD composite signal
    ax1 = axes[1]
    ax1.fill_between(signal.index, signal.values, color="#1f77b4", alpha=0.7)
    ax1.fill_between(mean_sig.index, mean_sig.values, color="#aec7e8", alpha=0.5,
                     label="mean signal")
    ax1.axhline(BOCPD_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
                label=f"threshold={BOCPD_THRESHOLD}")
    ax1.set_ylabel("P(changepoint)\nBOCPD")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", fontsize=8)
    for date in CRISIS_ENTRIES.values():
        if date >= signal.index[0]:
            ax1.axvline(date, color="red", linewidth=0.6, linestyle="--", alpha=0.4)

    # Panel 3: HMM P(crisis)
    ax2 = axes[2]
    ax2.fill_between(hmm_probs.index, hmm_probs["state_0"].values,
                     color="#d62728", alpha=0.7)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("P(crisis)\nHMM")
    ax2.set_ylim(0, 1)
    for date in CRISIS_ENTRIES.values():
        if date >= signal.index[0]:
            ax2.axvline(date, color="red", linewidth=0.6, linestyle="--", alpha=0.4)

    # Panel 4: HMM regime colour bar
    ax3 = axes[3]
    colours = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}
    prev, start = hmm_states.iloc[0], hmm_states.index[0]
    for date, state in hmm_states.items():
        if state != prev:
            ax3.axvspan(start, date, facecolor=colours[prev], alpha=0.8, linewidth=0)
            start, prev = date, state
    ax3.axvspan(start, hmm_states.index[-1], facecolor=colours[prev], alpha=0.8, linewidth=0)
    ax3.set_yticks([])
    ax3.set_ylabel("Regime")

    plt.tight_layout()
    plt.savefig("output/bocpd_vs_hmm.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_per_series(bocpd_df, benchmark):
    """Plot BOCPD signal for each correlation series individually."""
    n = bocpd_df.shape[1]
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3 * (n + 1)), sharex=True)
    fig.suptitle("BOCPD Per-Series Signals", fontsize=12, fontweight="bold")

    axes[0].plot(benchmark.index, benchmark.values, color="black", linewidth=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(HMM_BENCHMARK)
    for date in CRISIS_ENTRIES.values():
        if date >= benchmark.index[0]:
            axes[0].axvline(date, color="red", linewidth=0.7, linestyle="--", alpha=0.5)

    for i, col in enumerate(bocpd_df.columns):
        ax = axes[i + 1]
        ax.fill_between(bocpd_df.index, bocpd_df[col].values, alpha=0.7, color="#1f77b4")
        ax.axhline(BOCPD_THRESHOLD, color="red", linestyle="--", linewidth=0.7)
        ax.set_ylabel(col.replace("bocpd_corr_", "").replace("_", "/"), fontsize=8)
        ax.set_ylim(0, 1)
        for date in CRISIS_ENTRIES.values():
            if date >= bocpd_df.index[0]:
                ax.axvline(date, color="red", linewidth=0.6, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("output/bocpd_per_series.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_break_frequency(break_freq, benchmark, hmm_states):
    """Plot 30-day rolling break frequency — captures sustained instability."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 0.4]})
    fig.suptitle("BOCPD 30-Day Break Frequency", fontsize=12, fontweight="bold")

    axes[0].plot(benchmark.index, benchmark.values, color="black", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(HMM_BENCHMARK)

    axes[1].fill_between(break_freq.index, break_freq.values,
                         color="#9467bd", alpha=0.7)
    axes[1].set_ylabel("Break days\n(last 30d)")
    for date in CRISIS_ENTRIES.values():
        if date >= benchmark.index[0]:
            axes[0].axvline(date, color="red", linewidth=0.7, linestyle="--", alpha=0.5)
            axes[1].axvline(date, color="red", linewidth=0.7, linestyle="--", alpha=0.5)

    # Regime bar
    ax3 = axes[2]
    colours = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}
    prev, start = hmm_states.iloc[0], hmm_states.index[0]
    for date, state in hmm_states.items():
        if state != prev:
            ax3.axvspan(start, date, facecolor=colours[prev], alpha=0.8, linewidth=0)
            start, prev = date, state
    ax3.axvspan(start, hmm_states.index[-1], facecolor=colours[prev], alpha=0.8, linewidth=0)
    ax3.set_yticks([])
    ax3.set_ylabel("Regime")

    plt.tight_layout()
    plt.savefig("output/bocpd_break_frequency.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
