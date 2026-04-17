# dashboard.py
# Generates a self-contained HTML dashboard from scores.csv
# Run from project root: python dashboard.py
# Output: output/dashboard.html  (open in any browser)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import SCORE_OUTPUT_PATH, ENSEMBLE_DRAWDOWN_THRESH, ENSEMBLE_TARGET_WINDOW

REGIME_LABELS  = {0: "CRISIS", 1: "BEAR", 2: "BULL"}
REGIME_COLOURS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}
REGIME_BG      = {0: "#fde8e8", 1: "#fff3e0", 2: "#e8f5e9"}
OUTPUT_DIR     = "output"


def load_scores(path: str = SCORE_OUTPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def current_status(
    scores: pd.DataFrame,
    scores_full: pd.DataFrame | None = None,
) -> dict:
    """Extract latest reading into a plain dict.

    Parameters
    ----------
    scores      : filtered scores (up to as-of date)
    scores_full : full history for consistent threshold calculation
    """
    latest    = scores.iloc[-1]
    thresh_df = scores_full if scores_full is not None else scores
    p90       = float(np.percentile(thresh_df["risk_score_raw"].values, 90))
    p95       = float(np.percentile(thresh_df["risk_score_raw"].values, 95))
    return {
        "date":            scores.index[-1].strftime("%d %b %Y"),
        "regime":          REGIME_LABELS.get(int(latest["hmm_state"]), "UNKNOWN"),
        "regime_int":      int(latest["hmm_state"]),
        "p_crisis":        float(latest["hmm_p_crisis"]),
        "bocpd_today":     float(latest["bocpd_p_change"]),
        "bocpd_freq":      int(latest["bocpd_break_freq_30d"]),
        "score_raw":       float(latest["risk_score_raw"]),
        "score_calib":     float(latest["risk_score_calibrated"]),
        "alert_p90":       bool(latest["alert_p90"]),
        "alert_p95":       bool(latest["alert_p95"]),
        "p90_threshold":   p90,
        "p95_threshold":   p95,
        "prev_score":      float(scores["risk_score_raw"].iloc[-2]) if len(scores) > 1 else None,
        "prev_freq":       int(scores["bocpd_break_freq_30d"].iloc[-2]) if len(scores) > 1 else None,
    }


def make_charts(
    scores: pd.DataFrame,
    scores_full: pd.DataFrame | None = None,
    lookback_days: int = 756,
) -> dict:
    """Generate all charts, return dict of file paths.

    Parameters
    ----------
    scores      : filtered scores (up to as-of date)
    scores_full : full scores history for consistent threshold calculation.
                  If None, uses scores for thresholds.
    lookback_days : how many days to show in the main chart
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    recent   = scores.tail(lookback_days)
    thresh_df = scores_full if scores_full is not None else scores
    paths    = {}

    # ── Chart 1: Risk score with alert thresholds ──────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 0.4]})
    fig.suptitle("Downturn Risk Monitor", fontsize=13, fontweight="bold",
                 color="#1a1a2e")

    p90 = float(np.percentile(thresh_df["risk_score_raw"].values, 90))
    p95 = float(np.percentile(thresh_df["risk_score_raw"].values, 95))

    # Panel 1: Risk score
    ax0 = axes[0]
    ax0.fill_between(recent.index, recent["risk_score_raw"].values,
                     alpha=0.35, color="#1f77b4")
    ax0.plot(recent.index, recent["risk_score_raw"].values,
             color="#1f77b4", linewidth=0.8)
    ax0.axhline(p90, color="#ff7f0e", linestyle="--", linewidth=1,
                label=f"p90 alert ({p90:.3f})")
    ax0.axhline(p95, color="#d62728", linestyle="--", linewidth=1,
                label=f"p95 alert ({p95:.3f})")

    # Shade p90 alert periods
    alert = recent["alert_p90"].values
    for i in range(len(recent)):
        if alert[i]:
            ax0.axvline(recent.index[i], color="#ff7f0e", alpha=0.15,
                        linewidth=2)
    ax0.set_ylabel("Risk Score (raw)")
    ax0.set_ylim(0, min(1, recent["risk_score_raw"].max() * 1.2))
    ax0.legend(loc="upper left", fontsize=8)

    # Panel 2: BOCPD break frequency
    ax1 = axes[1]
    ax1.fill_between(recent.index, recent["bocpd_break_freq_30d"].values,
                     alpha=0.6, color="#9467bd", step="mid")
    ax1.axhline(10, color="#d62728", linestyle=":", linewidth=0.8,
                label="freq=10")
    ax1.set_ylabel("Break freq\n(30d)")
    ax1.legend(loc="upper left", fontsize=8)

    # Panel 3: Regime colour bar
    ax2 = axes[2]
    prev, start = int(recent["hmm_state"].iloc[0]), recent.index[0]
    for date, state in recent["hmm_state"].items():
        state = int(state)
        if state != prev:
            ax2.axvspan(start, date, facecolor=REGIME_COLOURS[prev],
                        alpha=0.7, linewidth=0)
            start, prev = date, state
    ax2.axvspan(start, recent.index[-1], facecolor=REGIME_COLOURS[prev],
                alpha=0.7, linewidth=0)
    ax2.set_yticks([])
    ax2.set_ylabel("Regime")

    plt.tight_layout()
    path1 = f"{OUTPUT_DIR}/chart_risk_score.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["risk_score"] = path1

    # ── Chart 2: Signal breakdown (last 90 days) ───────────────────────────
    recent90 = scores.tail(90)
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("Signal Breakdown — Last 90 Days", fontsize=12,
                 fontweight="bold", color="#1a1a2e")

    axes[0].plot(recent90.index, recent90["hmm_p_crisis"].values,
                 color="#d62728", linewidth=1)
    axes[0].fill_between(recent90.index, recent90["hmm_p_crisis"].values,
                         alpha=0.4, color="#d62728")
    axes[0].set_ylabel("HMM P(crisis)")
    axes[0].set_ylim(0, 1)

    axes[1].fill_between(recent90.index, recent90["bocpd_p_change"].values,
                         alpha=0.6, color="#1f77b4", step="mid")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("BOCPD\nP(break)")
    axes[1].set_ylim(0, 1)

    axes[2].fill_between(recent90.index, recent90["risk_score_raw"].values,
                         alpha=0.6, color="#2ca02c")
    axes[2].axhline(p90, color="#ff7f0e", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Risk score\n(raw)")
    axes[2].set_ylim(0, 1)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    path2 = f"{OUTPUT_DIR}/chart_signals_90d.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    paths["signals_90d"] = path2

    return paths


def make_html(status: dict, chart_paths: dict, scores: pd.DataFrame, is_historical: bool = False) -> str:
    """Render a self-contained HTML dashboard."""

    regime      = status["regime"]
    regime_col  = REGIME_COLOURS[status["regime_int"]]
    regime_bg   = REGIME_BG[status["regime_int"]]
    alert_html  = ""
    if status["alert_p95"]:
        alert_html = '<div class="alert alert-red">🔴 HIGH ALERT — Risk score above p95 threshold</div>'
    elif status["alert_p90"]:
        alert_html = '<div class="alert alert-amber">🟠 ELEVATED ALERT — Risk score above p90 threshold</div>'
    else:
        alert_html = '<div class="alert alert-green">🟢 No alert — Risk score within normal range</div>'

    score_delta = ""
    if status["prev_score"] is not None:
        delta = status["score_raw"] - status["prev_score"]
        arrow = "▲" if delta > 0 else "▼"
        col   = "#d62728" if delta > 0 else "#2ca02c"
        score_delta = f'<span style="color:{col};font-size:0.85em"> {arrow} {abs(delta):.3f}</span>'

    freq_delta = ""
    if status["prev_freq"] is not None:
        delta = status["bocpd_freq"] - status["prev_freq"]
        if delta != 0:
            arrow = "▲" if delta > 0 else "▼"
            col   = "#d62728" if delta > 0 else "#2ca02c"
            freq_delta = f'<span style="color:{col};font-size:0.85em"> {arrow} {abs(delta)}</span>'

    # Weekly history table (last 8 weeks, Fridays only)
    weekly = scores.resample("W-FRI").last().tail(8)
    table_rows = ""
    for date, row in weekly.iterrows():
        s     = int(row["hmm_state"])
        rcol  = REGIME_COLOURS[s]
        score = row["risk_score_raw"]
        a90   = "🔴" if row["alert_p90"] else "🟢"
        table_rows += f"""
        <tr>
            <td>{date.strftime("%d %b %Y")}</td>
            <td><span style="color:{rcol};font-weight:600">{REGIME_LABELS[s]}</span></td>
            <td>{row['hmm_p_crisis']:.4f}</td>
            <td>{row['bocpd_p_change']:.3f}</td>
            <td>{int(row['bocpd_break_freq_30d'])}</td>
            <td>{score:.3f}</td>
            <td>{row['risk_score_calibrated']:.3f}</td>
            <td>{a90}</td>
        </tr>"""

    # Embed charts as base64
    import base64
    def img_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    img1 = img_b64(chart_paths["risk_score"])
    img2 = img_b64(chart_paths["signals_90d"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Downturn Risk Monitor — {status['date']}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f6fa; color: #2c3e50; }}
  .header {{ background: #1a1a2e; color: white; padding: 20px 32px;
             display: flex; justify-content: space-between; align-items: center; }}
  .header h1 {{ font-size: 1.4em; font-weight: 600; }}
  .header .date {{ font-size: 0.9em; color: #aaa; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  .alert {{ padding: 12px 20px; border-radius: 8px; margin-bottom: 20px;
            font-weight: 500; font-size: 1.05em; }}
  .alert-red   {{ background: #fde8e8; color: #c0392b; border-left: 4px solid #d62728; }}
  .alert-amber {{ background: #fff3e0; color: #d35400; border-left: 4px solid #ff7f0e; }}
  .alert-green {{ background: #e8f5e9; color: #27ae60; border-left: 4px solid #2ca02c; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px; margin-bottom: 24px; }}
  .card {{ background: white; border-radius: 10px; padding: 18px;
           box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .card .label {{ font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.05em;
                  color: #7f8c8d; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.8em; font-weight: 700; }}
  .card .sub   {{ font-size: 0.8em; color: #95a5a6; margin-top: 4px; }}
  .regime-card .value {{ color: {regime_col}; }}
  .section {{ background: white; border-radius: 10px; padding: 20px;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 20px; }}
  .section h2 {{ font-size: 1em; font-weight: 600; color: #34495e;
                 margin-bottom: 16px; padding-bottom: 8px;
                 border-bottom: 1px solid #ecf0f1; }}
  .section img {{ width: 100%; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
  th {{ background: #f8f9fa; padding: 10px 12px; text-align: left;
        font-weight: 600; color: #555; border-bottom: 2px solid #dee2e6; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; }}
  tr:hover td {{ background: #fafafa; }}
  .footer {{ text-align: center; color: #aaa; font-size: 0.8em; padding: 20px; }}
  .signal-explainer {{ display: grid; grid-template-columns: 1fr 1fr;
                        gap: 12px; margin-top: 8px; }}
  .signal-item {{ background: #f8f9fa; border-radius: 6px; padding: 12px; }}
  .signal-item .name {{ font-weight: 600; font-size: 0.85em; color: #2c3e50; }}
  .signal-item .desc {{ font-size: 0.8em; color: #7f8c8d; margin-top: 4px; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Downturn Risk Monitor</h1>
  <span class="date">As of {status['date']} &nbsp;|&nbsp; Model: yfinance_v1 &nbsp;|&nbsp;
  Target: {abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} drawdown / {ENSEMBLE_TARGET_WINDOW}d</span>
</div>

<div class="container">

  {alert_html}

  <!-- KPI Cards -->
  <div class="cards">
    <div class="card regime-card">
      <div class="label">HMM Regime</div>
      <div class="value">{regime}</div>
      <div class="sub">P(crisis) = {status['p_crisis']:.4f}</div>
    </div>
    <div class="card">
      <div class="label">Risk Score (raw)</div>
      <div class="value">{status['score_raw']:.3f}{score_delta}</div>
      <div class="sub">p90={status['p90_threshold']:.3f} &nbsp; p95={status['p95_threshold']:.3f}</div>
    </div>
    <div class="card">
      <div class="label">Risk Score (calib)</div>
      <div class="value">{status['score_calib']:.3f}</div>
      <div class="sub">~{status['score_calib']:.0%} prob of -{abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} dd</div>
    </div>
    <div class="card">
      <div class="label">BOCPD Break Freq</div>
      <div class="value">{status['bocpd_freq']}{freq_delta}</div>
      <div class="sub">days flagged / last 30</div>
    </div>
    <div class="card">
      <div class="label">BOCPD Today</div>
      <div class="value">{status['bocpd_today']:.3f}</div>
      <div class="sub">P(structural break)</div>
    </div>
  </div>

  <!-- Risk Score Chart -->
  <div class="section">
    <h2>Risk Score History (3 Years)</h2>
    <img src="data:image/png;base64,{img1}" alt="Risk score chart">
  </div>

  <!-- Signal Breakdown -->
  <div class="section">
    <h2>Signal Breakdown — Last 90 Days</h2>
    <img src="data:image/png;base64,{img2}" alt="Signal breakdown">
  </div>

  <!-- Weekly History Table -->
  <div class="section">
    <h2>Weekly History (Last 8 Weeks)</h2>
    <table>
      <thead>
        <tr>
          <th>Date</th><th>Regime</th><th>P(crisis)</th>
          <th>BOCPD today</th><th>Break freq 30d</th>
          <th>Score (raw)</th><th>Score (calib)</th><th>Alert p90</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>

  <!-- Signal Guide -->
  <div class="section">
    <h2>Signal Guide</h2>
    <div class="signal-explainer">
      <div class="signal-item">
        <div class="name">HMM Regime</div>
        <div class="desc">Hidden Markov Model state: BULL / BEAR / CRISIS.
        Based on return, volatility and drawdown patterns.
        Changes slowly — average 54-166 days per state.</div>
      </div>
      <div class="signal-item">
        <div class="name">P(crisis)</div>
        <div class="desc">Probability of being in the crisis state.
        Rises before the regime officially flips.
        Key leading indicator within the HMM layer.</div>
      </div>
      <div class="signal-item">
        <div class="name">BOCPD P(break)</div>
        <div class="desc">Probability of a structural break today in
        cross-market correlations and rates relationships.
        Historically leads HMM by 28-109 days at crisis onset.</div>
      </div>
      <div class="signal-item">
        <div class="name">Break freq 30d</div>
        <div class="desc">Days in the last 30 where BOCPD fired above 0.5.
        Smoothed version of the daily spike — better for
        sustained stress monitoring. &gt;10 = elevated.</div>
      </div>
      <div class="signal-item">
        <div class="name">Risk score (raw)</div>
        <div class="desc">XGBoost ensemble combining all signals.
        Varies continuously day-to-day.
        Use for alerts and trend tracking.</div>
      </div>
      <div class="signal-item">
        <div class="name">Risk score (calibrated)</div>
        <div class="desc">Probability-calibrated version of the raw score.
        Interpret as: approximate probability of a
        {abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} drawdown in the next {ENSEMBLE_TARGET_WINDOW} trading days.</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  Generated {datetime.now().strftime("%d %b %Y %H:%M")} &nbsp;|&nbsp;
  HMM-RegimeModel &nbsp;|&nbsp; Not investment advice
</div>

</body>
</html>"""
    return html


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate risk monitor dashboard")
    parser.add_argument(
        "--date",
        default=None,
        help="View dashboard as of a specific date (YYYY-MM-DD). "
             "Defaults to latest available date.",
    )
    parser.add_argument(
        "--scores",
        default=SCORE_OUTPUT_PATH,
        help=f"Path to scores CSV (default: {SCORE_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading scores...")
    scores_full = load_scores(args.scores)
    print(f"  {len(scores_full)} rows, "
          f"{scores_full.index[0].date()} -> {scores_full.index[-1].date()}")

    # ── Date filtering ─────────────────────────────────────────────────────
    if args.date:
        as_of = pd.Timestamp(args.date)
        # Find the last available score on or before the requested date
        available = scores_full[scores_full.index <= as_of]
        if available.empty:
            print(f"No scores available on or before {args.date}. "
                  f"Earliest available: {scores_full.index[0].date()}")
            return
        scores = available
        actual_date = scores.index[-1].date()
        if actual_date != as_of.date():
            print(f"  Note: no score for {args.date} "
                  f"(non-trading day?) — using {actual_date} instead")
        # Percentile thresholds use FULL history so comparisons remain consistent
        # but current_status() only reads from the filtered scores
        print(f"  Viewing as of: {actual_date}")
    else:
        scores = scores_full

    # ── Generate ───────────────────────────────────────────────────────────
    print("Generating charts...")
    # Charts use filtered scores but thresholds anchored to full history
    chart_paths = make_charts(scores, scores_full=scores_full)

    print("Building dashboard...")
    status   = current_status(scores, scores_full=scores_full)
    html     = make_html(status, chart_paths, scores, is_historical=bool(args.date))

    # Name output file by date so historical dashboards don't overwrite each other
    date_str = scores.index[-1].strftime("%Y%m%d")
    out_path = f"{OUTPUT_DIR}/dashboard_{date_str}.html"
    # Also write a "latest" version for convenience
    latest_path = f"{OUTPUT_DIR}/dashboard.html"
    for path in [out_path, latest_path] if not args.date else [out_path]:
        with open(path, "w") as f:
            f.write(html)

    print(f"\nDashboard saved: {out_path}")
    print(f"Open in browser: file://{os.path.abspath(out_path)}")
    print()
    print(f"  As of:      {scores.index[-1].date()}")
    print(f"  Regime:     {status['regime']}")
    print(f"  Risk score: {status['score_raw']:.3f}  (p90={status['p90_threshold']:.3f})")
    print(f"  Alert p90:  {'YES 🔴' if status['alert_p90'] else 'NO 🟢'}")


if __name__ == "__main__":
    main()
