# models/bocpd.py
# Bayesian Online Changepoint Detection (BOCPD) applied to rolling
# cross-asset correlations. Produces P(changepoint at t) as a signal
# for the ensemble layer.
#
# Key design notes:
#   - Uses P(run_length <= 4) as the changepoint signal, not R[0,t].
#     R[0,t] is always ~hazard_rate (tiny). The short-run-length probability
#     mass correctly spikes at structural breaks.
#   - Run on rolling pairwise correlations, not raw returns — correlation
#     regime shifts often PRECEDE price-based regime transitions.
#   - Composite signal = max across all series (fires if ANY pair breaks).

import numpy as np
import pandas as pd
from functools import partial
from bayesian_changepoint_detection.online_changepoint_detection import (
    online_changepoint_detection,
    constant_hazard,
    StudentT,
)
from config import BOCPD_HAZARD_LAMBDA


class BOCPDetector:
    """
    Applies BOCPD independently to each correlation time series and
    aggregates into a composite changepoint signal.

    Usage
    -----
    detector = BOCPDetector()
    bocpd_df = detector.run(corr_features)        # per-series signals
    signal   = detector.composite_signal(bocpd_df) # single composite
    """

    def __init__(
        self,
        hazard_lambda: float = BOCPD_HAZARD_LAMBDA,
        short_run_window: int = 5,
        warmup: int = 10,
    ):
        """
        Parameters
        ----------
        hazard_lambda    : Expected observations between changepoints.
                           ~250 = roughly one per year on daily data.
                           Lower = more sensitive (more false positives).
        short_run_window : P(run_length <= k) is used as the signal.
                           k=5 means "probability a new regime just started
                           within the last 5 days".
        warmup           : Number of initial observations to zero out
                           (BOCPD initialisation causes a spike at t=0).
        """
        self.hazard_lambda    = hazard_lambda
        self.short_run_window = short_run_window
        self.warmup           = warmup
        self._hazard_fn       = partial(constant_hazard, hazard_lambda)
        # NB: StudentT is stateful (accumulates observations) so we instantiate
        # a fresh copy per series in run() rather than storing a single instance.

    def run(self, series: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Run BOCPD on each column of `series` independently.

        Parameters
        ----------
        series : pd.DataFrame or pd.Series
            Observation sequence (rows = timesteps). No NaNs.

        Returns
        -------
        pd.DataFrame
            P(changepoint at t) for each column, indexed like input.
            Columns prefixed with 'bocpd_'.
        """
        if isinstance(series, pd.Series):
            series = series.to_frame()

        results = {}
        n_series = len(series.columns)
        for i, col in enumerate(series.columns):
            print(f"  Running BOCPD on series {i+1}/{n_series}: {col}", end="\r")
            data = series[col].values
            R, _ = online_changepoint_detection(
                data,
                self._hazard_fn,
                StudentT(alpha=0.1, beta=0.01, kappa=1, mu=0),  # fresh per series - StudentT is stateful
            )
            # P(changepoint) = P(run_length is very short)
            # Sum probability mass on run lengths 0..short_run_window-1
            signal = R[:self.short_run_window, :].sum(axis=0)
            # Zero out warmup period to remove initialisation spike
            signal[:self.warmup] = 0.0
            results[f"bocpd_{col}"] = signal[:len(data)]  # R is (T+1,T+1); trim to T

        print()  # newline after progress
        return pd.DataFrame(results, index=series.index)

    def composite_signal(self, bocpd_df: pd.DataFrame) -> pd.Series:
        """
        Max across all series — fires if ANY correlation pair breaks.
        This is the primary signal for the ensemble layer.
        """
        return bocpd_df.max(axis=1).rename("bocpd_p_changepoint")

    def mean_signal(self, bocpd_df: pd.DataFrame) -> pd.Series:
        """
        Mean across all series — less sensitive, fewer false positives.
        Use as an alternative or secondary signal.
        """
        return bocpd_df.mean(axis=1).rename("bocpd_p_changepoint_mean")

    def rolling_break_frequency(
        self, signal: pd.Series, window: int = 30, threshold: float = 0.5
    ) -> pd.Series:
        """
        Count of days in the past `window` days where P(changepoint) > threshold.
        Captures sustained structural instability rather than single spikes.
        Useful as an additional ensemble feature.
        """
        return (
            (signal > threshold)
            .rolling(window)
            .sum()
            .rename(f"bocpd_break_freq_{window}d")
        )
