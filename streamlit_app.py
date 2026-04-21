# streamlit_app.py
# Web dashboard for the Downturn Risk Monitor.
# Run from project root:
#   streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# Inline the constants we need so this file has no dependency on the ML stack.
# Keep these in sync with config.py.
SCORE_OUTPUT_PATH        = "output/scores.csv"
ENSEMBLE_DRAWDOWN_THRESH = -0.10
ENSEMBLE_TARGET_WINDOW   = 20

REGIME_LABELS  = {0: "CRISIS", 1: "BEAR", 2: "BULL"}
REGIME_COLOURS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Downturn Risk Monitor",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_scores(path: str = SCORE_OUTPUT_PATH) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def current_status(scores: pd.DataFrame, scores_full: pd.DataFrame) -> dict:
    latest = scores.iloc[-1]
    p90    = float(np.percentile(scores_full["risk_score_raw"].values, 90))
    p95    = float(np.percentile(scores_full["risk_score_raw"].values, 95))
    return {
        "date":          scores.index[-1].strftime("%d %b %Y"),
        "regime":        REGIME_LABELS.get(int(latest["hmm_state"]), "UNKNOWN"),
        "regime_int":    int(latest["hmm_state"]),
        "p_crisis":      float(latest["hmm_p_crisis"]),
        "bocpd_today":   float(latest["bocpd_p_change"]),
        "bocpd_freq":    int(latest["bocpd_break_freq_30d"]),
        "score_raw":     float(latest["risk_score_raw"]),
        "score_calib":   float(latest["risk_score_calibrated"]),
        "alert_p90":     bool(latest["alert_p90"]),
        "alert_p95":     bool(latest["alert_p95"]),
        "p90_threshold": p90,
        "p95_threshold": p95,
        "prev_score":    float(scores["risk_score_raw"].iloc[-2]) if len(scores) > 1 else None,
        "prev_freq":     int(scores["bocpd_break_freq_30d"].iloc[-2]) if len(scores) > 1 else None,
    }


def make_risk_score_chart(scores: pd.DataFrame, scores_full: pd.DataFrame, lookback_days: int = 756) -> go.Figure:
    recent    = scores.tail(lookback_days)
    p90       = float(np.percentile(scores_full["risk_score_raw"].values, 90))
    p95       = float(np.percentile(scores_full["risk_score_raw"].values, 95))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.30, 0.15],
        vertical_spacing=0.04,
        subplot_titles=("Risk Score (raw)", "BOCPD Break Freq (30d)", "Regime"),
    )

    # ── Panel 1: Risk score ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["risk_score_raw"],
        fill="tozeroy", fillcolor="rgba(31,119,180,0.25)",
        line=dict(color="#1f77b4", width=1),
        name="Risk score",
    ), row=1, col=1)

    fig.add_hline(y=p90, line=dict(color="#ff7f0e", dash="dash", width=1.5),
                  annotation_text=f"p90 ({p90:.3f})",
                  annotation_position="top right", row=1, col=1)
    fig.add_hline(y=p95, line=dict(color="#d62728", dash="dash", width=1.5),
                  annotation_text=f"p95 ({p95:.3f})",
                  annotation_position="top right", row=1, col=1)

    # Shade p90 alert periods as vertical spans
    alert_days = recent[recent["alert_p90"].astype(bool)].index
    for day in alert_days:
        fig.add_vrect(
            x0=day - pd.Timedelta(hours=12),
            x1=day + pd.Timedelta(hours=12),
            fillcolor="rgba(255,127,14,0.15)", line_width=0,
            row=1, col=1,
        )

    # ── Panel 2: BOCPD break frequency ───────────────────────────────────────
    fig.add_trace(go.Bar(
        x=recent.index, y=recent["bocpd_break_freq_30d"],
        marker_color="rgba(148,103,189,0.7)",
        name="Break freq 30d",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=10, line=dict(color="#d62728", dash="dot", width=1),
                  row=2, col=1)

    # ── Panel 3: Regime colour bar ────────────────────────────────────────────
    # Build contiguous spans for each regime state
    prev_state = int(recent["hmm_state"].iloc[0])
    span_start = recent.index[0]
    for date, state in recent["hmm_state"].items():
        state = int(state)
        if state != prev_state:
            fig.add_vrect(
                x0=span_start, x1=date,
                fillcolor=REGIME_COLOURS[prev_state],
                opacity=0.8, line_width=0,
                row=3, col=1,
            )
            span_start = date
            prev_state = state
    fig.add_vrect(
        x0=span_start, x1=recent.index[-1],
        fillcolor=REGIME_COLOURS[prev_state],
        opacity=0.8, line_width=0,
        row=3, col=1,
    )
    # invisible scatter just to silence "no traces" warning on row 3
    fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=3, col=1)

    fig.update_yaxes(range=[0, max(1.05, recent["risk_score_raw"].max() * 1.1)],
                     row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=3, col=1)
    fig.update_layout(
        height=520,
        margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def make_signals_chart(scores: pd.DataFrame, scores_full: pd.DataFrame) -> go.Figure:
    recent90 = scores.tail(90)
    p90      = float(np.percentile(scores_full["risk_score_raw"].values, 90))

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[1, 1, 1],
        vertical_spacing=0.07,
        subplot_titles=("HMM P(crisis)", "BOCPD P(structural break)", "Risk Score (raw)"),
    )

    fig.add_trace(go.Scatter(
        x=recent90.index, y=recent90["hmm_p_crisis"],
        fill="tozeroy", fillcolor="rgba(214,39,40,0.3)",
        line=dict(color="#d62728", width=1.5),
        name="P(crisis)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=recent90.index, y=recent90["bocpd_p_change"],
        fill="tozeroy", fillcolor="rgba(31,119,180,0.35)",
        line=dict(color="#1f77b4", width=1),
        name="BOCPD P(break)",
    ), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color="red", dash="dash", width=0.8), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=recent90.index, y=recent90["risk_score_raw"],
        fill="tozeroy", fillcolor="rgba(44,160,44,0.35)",
        line=dict(color="#2ca02c", width=1),
        name="Risk score",
    ), row=3, col=1)
    fig.add_hline(y=p90, line=dict(color="#ff7f0e", dash="dash", width=1),
                  annotation_text=f"p90={p90:.3f}", annotation_position="top right",
                  row=3, col=1)

    for row in [1, 2, 3]:
        fig.update_yaxes(range=[0, 1.05], row=row, col=1)

    fig.update_layout(
        height=480,
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Sidebar — date picker
    st.sidebar.title("📊 Risk Monitor")
    st.sidebar.caption("HMM-RegimeModel")

    scores_full = load_scores()

    if scores_full is None:
        st.error(
            f"**No scores file found** at `{SCORE_OUTPUT_PATH}`.  \n\n"
            "Run the pipeline to generate it:\n"
            "```\npython pipeline/score.py\n```\n"
            "then commit and push `output/scores.csv`."
        )
        return

    min_date = scores_full.index[0].date()
    max_date = scores_full.index[-1].date()

    as_of = st.sidebar.date_input(
        "View as of date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select a historical date to see how the dashboard looked then.",
    )

    # Filter to the selected date
    scores = scores_full[scores_full.index <= pd.Timestamp(as_of)]
    if scores.empty:
        st.error(f"No data available on or before {as_of}.")
        return

    actual_date = scores.index[-1].date()
    is_historical = actual_date != max_date

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Data range:** {min_date} → {max_date}  \n"
        f"**Viewing:** {actual_date}"
    )
    st.sidebar.caption("Scores refresh every 5 min. Run `pipeline/score.py` to update.")

    status = current_status(scores, scores_full)

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.title("📊 Downturn Risk Monitor")
    with col_meta:
        st.markdown(
            f"**As of:** {status['date']}  \n"
            f"Model: yfinance_v1  \n"
            f"Target: {abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} dd / {ENSEMBLE_TARGET_WINDOW}d"
        )

    if is_historical:
        st.info(f"Viewing historical snapshot: **{actual_date}**")

    # ── Alert banner ──────────────────────────────────────────────────────────
    if status["alert_p95"]:
        st.error("🔴 HIGH ALERT — Risk score above p95 threshold")
    elif status["alert_p90"]:
        st.warning("🟠 ELEVATED ALERT — Risk score above p90 threshold")
    else:
        st.success("🟢 No alert — Risk score within normal range")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    regime_col = REGIME_COLOURS[status["regime_int"]]
    c1.metric(
        label="HMM Regime",
        value=status["regime"],
        help=f"P(crisis) = {status['p_crisis']:.4f}",
    )

    score_delta = None
    if status["prev_score"] is not None:
        score_delta = round(status["score_raw"] - status["prev_score"], 3)
    c2.metric(
        label="Risk Score (raw)",
        value=f"{status['score_raw']:.3f}",
        delta=score_delta,
        delta_color="inverse",
        help=f"p90={status['p90_threshold']:.3f}  p95={status['p95_threshold']:.3f}",
    )

    c3.metric(
        label="Risk Score (calib)",
        value=f"{status['score_calib']:.3f}",
        help=f"~{status['score_calib']:.0%} prob of -{abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} drawdown in {ENSEMBLE_TARGET_WINDOW}d",
    )

    freq_delta = None
    if status["prev_freq"] is not None:
        freq_delta = status["bocpd_freq"] - status["prev_freq"]
    c4.metric(
        label="BOCPD Break Freq (30d)",
        value=status["bocpd_freq"],
        delta=freq_delta,
        delta_color="inverse",
        help="Days flagged in last 30 trading days",
    )

    c5.metric(
        label="BOCPD Today",
        value=f"{status['bocpd_today']:.3f}",
        help="P(structural break today)",
    )

    st.markdown("---")

    # ── Risk score chart (3 years) ────────────────────────────────────────────
    st.subheader("Risk Score History (3 Years)")
    fig1 = make_risk_score_chart(scores, scores_full)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Signal breakdown (90d) ────────────────────────────────────────────────
    st.subheader("Signal Breakdown — Last 90 Days")
    fig2 = make_signals_chart(scores, scores_full)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── Weekly history table ──────────────────────────────────────────────────
    st.subheader("Weekly History (Last 8 Weeks)")

    weekly = scores.resample("W-FRI").last().tail(8).copy()
    weekly["Regime"] = weekly["hmm_state"].map(REGIME_LABELS)
    weekly["Alert"] = weekly["alert_p90"].map({True: "p90 🔴", False: "🟢"})

    display = weekly[[
        "Regime", "hmm_p_crisis", "bocpd_p_change",
        "bocpd_break_freq_30d", "risk_score_raw", "risk_score_calibrated", "Alert",
    ]].rename(columns={
        "hmm_p_crisis":          "P(crisis)",
        "bocpd_p_change":        "BOCPD today",
        "bocpd_break_freq_30d":  "Break freq 30d",
        "risk_score_raw":        "Score (raw)",
        "risk_score_calibrated": "Score (calib)",
    }).sort_index(ascending=False)

    st.dataframe(
        display
        use_container_width=True,
        column_config={
            "_index": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        },
    )

    st.markdown("---")

    # ── Signal guide ──────────────────────────────────────────────────────────
    with st.expander("Signal Guide", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**HMM Regime**")
            st.caption("Hidden Markov Model state: BULL / BEAR / CRISIS. Based on return, volatility and drawdown patterns. Changes slowly — average 54–166 days per state.")
            st.markdown("**P(crisis)**")
            st.caption("Probability of being in the crisis state. Rises before the regime officially flips. Key leading indicator within the HMM layer.")
            st.markdown("**BOCPD P(break)**")
            st.caption("Probability of a structural break today in cross-market correlations and rates relationships. Historically leads HMM by 28–109 days at crisis onset.")
        with g2:
            st.markdown("**Break freq 30d**")
            st.caption("Days in the last 30 where BOCPD fired above 0.5. Smoothed version of the daily spike — better for sustained stress monitoring. >10 = elevated.")
            st.markdown("**Risk score (raw)**")
            st.caption("XGBoost ensemble combining all signals. Varies continuously day-to-day. Use for alerts and trend tracking.")
            st.markdown("**Risk score (calibrated)**")
            st.caption(
                f"Probability-calibrated version of the raw score. Interpret as: approximate probability of a "
                f"{abs(ENSEMBLE_DRAWDOWN_THRESH):.0%} drawdown in the next {ENSEMBLE_TARGET_WINDOW} trading days."
            )

    st.caption(
        f"Generated {datetime.now().strftime('%d %b %Y %H:%M')} | "
        "HMM-RegimeModel | Not investment advice"
    )


if __name__ == "__main__":
    main()
