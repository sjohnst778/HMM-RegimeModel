# validate_hmm.py
# Run from project root: python validate_hmm.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import sys, os

from data.fetch import fetch_equity_prices
from data.features import build_hmm_features_v2
from models.hmm import RegimeHMM, STATE_LABELS, STATE_COLOURS
from config import HMM_N_STATES, HMM_BENCHMARK, HMM_MIN_PERSISTENCE

CRISIS_BANDS = [
    ("2000-03-01", "2002-10-01", "Dot-com"),
    ("2007-10-01", "2009-03-01", "GFC"),
    ("2020-02-15", "2020-04-01", "COVID"),
    ("2022-01-01", "2022-10-01", "Rate hikes"),
]

def main():
    # --- Data ---
    print("Fetching equity data...")
    prices = fetch_equity_prices(start="2000-01-01")
    if prices.empty:
        print("No data returned."); sys.exit(1)

    benchmark = prices[HMM_BENCHMARK]
    print(f"Downloaded {len(prices)} days | {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Benchmark: {HMM_BENCHMARK}  |  Min persistence: {HMM_MIN_PERSISTENCE} days\n")

    # --- Features ---
    features = build_hmm_features_v2(prices, benchmark=HMM_BENCHMARK)
    print(f"Features: {list(features.columns)}  shape: {features.shape}\n")

    # --- Fit ---
    print("Fitting HMM...")
    model = RegimeHMM(n_states=HMM_N_STATES)
    model.fit(features)

    states = model.predict(features, min_persistence=HMM_MIN_PERSISTENCE)
    probs  = model.predict_proba(features)

    n_trans = int((states != states.shift()).sum()) - 1
    print(f"\nRegime transitions after filtering: {n_trans}")

    # --- Diagnostics ---
    print("\n-- Transition Matrix --")
    print(model.transition_matrix().round(3))

    print("\n-- State Summary --")
    for s in range(HMM_N_STATES):
        mask = states == s
        print(f"  {STATE_LABELS[s]:6s}: vol={features.loc[mask,'vol_21d'].mean():.3f} "
              f"dd={features.loc[mask,'drawdown_252d'].mean():.3f} "
              f"n={mask.sum()} ({100*mask.mean():.1f}%)  "
              f"avg_run={states.groupby((states!=states.shift()).cumsum())[states==s].count().mean() if False else _avg_run(states,s):.0f}d")

    # --- Plot ---
    _plot(benchmark, states, probs)
    _plot_runlengths(states)
    print("\nCharts saved to output/")

def _avg_run(states, s):
    runs = []
    count = 0
    for x in states:
        if x == s: count += 1
        elif count: runs.append(count); count = 0
    if count: runs.append(count)
    return np.mean(runs) if runs else 0

def _plot(benchmark, states, probs):
    os.makedirs("output", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("HMM Market Regime Detection", fontsize=14, fontweight="bold")

    dates = states.index
    state_rgb = np.array([mcolors.to_rgb(STATE_COLOURS[s]) for s in states.values])

    # --- Panel 1: Price ---
    ax0 = axes[0]
    ax0.set_yscale("log")
    ax0.set_ylabel(HMM_BENCHMARK)

    # Draw regime blocks as coloured rectangles directly
    _draw_regime_blocks(ax0, states, benchmark, alpha=0.28)

    ax0.plot(benchmark.index, benchmark.values, color="black", linewidth=0.8, zorder=5)

    # Crisis band markers (vertical lines only, no axvspan)
    for start, end, label in CRISIS_BANDS:
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        if mid >= dates[0]:
            ax0.axvline(mid, color="grey", linewidth=0.6, alpha=0.5, zorder=2)
            ax0.text(mid, benchmark.min() * 1.05, label,
                     ha="center", va="bottom", fontsize=7, color="dimgrey", rotation=90)

    _add_legend(ax0)

    # --- Panel 2: Regime colour bar ---
    ax1 = axes[1]
    _draw_regime_blocks(ax1, states, None, alpha=0.95)
    ax1.set_yticks([])
    ax1.set_ylabel("Regime")
    ax1.set_ylim(0, 1)
    _add_legend(ax1)

    # --- Panel 3: P(crisis) ---
    ax2 = axes[2]
    ax2.fill_between(probs.index, probs["state_0"].values,
                     color=STATE_COLOURS[0], alpha=0.7)
    ax2.axhline(0.5, color="black", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("P(crisis)")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("output/hmm_regimes.png", dpi=150, bbox_inches="tight")
    plt.close()

def _draw_regime_blocks(ax, states, prices=None, alpha=0.3):
    """Draw one Rectangle patch per contiguous regime block."""
    from matplotlib.patches import Rectangle

    # Build blocks
    blocks = []
    prev, start = states.iloc[0], states.index[0]
    for date, state in states.items():
        if state != prev:
            blocks.append((start, date, prev))
            start, prev = date, state
    blocks.append((start, states.index[-1], prev))

    # Get y-range
    if prices is not None:
        ymin, ymax = prices.min() * 0.95, prices.max() * 1.05
    else:
        ymin, ymax = 0, 1

    for blk_start, blk_end, state in blocks:
        width = (blk_end - blk_start).days
        rect = Rectangle(
            (blk_start, ymin),
            pd.Timedelta(days=width),
            ymax - ymin,
            facecolor=STATE_COLOURS[state],
            alpha=alpha,
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(rect)

    ax.set_xlim(states.index[0], states.index[-1])
    if prices is not None:
        ax.set_ylim(ymin, ymax)

def _add_legend(ax):
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=STATE_COLOURS[i], label=STATE_LABELS[i]) for i in range(3)],
        loc="upper left", fontsize=9
    )

def _plot_runlengths(states):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    fig.suptitle("Regime Run-Length Distributions", fontsize=12)
    for s in range(3):
        runs, count = [], 0
        for x in states:
            if x == s: count += 1
            elif count: runs.append(count); count = 0
        if count: runs.append(count)
        axes[s].hist(runs, bins=30, color=STATE_COLOURS[s], edgecolor="white", alpha=0.85)
        axes[s].set_title(f"State {s}: {STATE_LABELS[s]}")
        axes[s].set_xlabel("Consecutive days")
        if s == 0: axes[s].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("output/hmm_run_lengths.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
