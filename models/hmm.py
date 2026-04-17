# models/hmm.py
# Gaussian HMM for market regime detection.
# Fits a 3-state model (bull / bear / crisis) on return + vol features,
# labels historical regimes, and exposes state probabilities for the ensemble layer.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn.hmm import GaussianHMM
from config import HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE, HMM_RANDOM_STATE, HMM_N_RESTARTS

# Map state indices to human-readable labels.
# After fitting, states are re-ordered by mean return (low → high),
# so 0 = crisis, 1 = bear, 2 = bull.
STATE_LABELS = {0: "crisis", 1: "bear", 2: "bull"}
STATE_COLOURS = {0: "#d62728", 1: "#ff7f0e", 2: "#2ca02c"}


class RegimeHMM:
    """
    Wrapper around hmmlearn GaussianHMM providing:
    - fit / predict interface consistent with the ensemble pipeline
    - state re-ordering by mean return
    - state probability output for the ensemble layer
    - simple visualisation
    """

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        n_iter: int = HMM_N_ITER,
        covariance_type: str = HMM_COVARIANCE,
        random_state: int = HMM_RANDOM_STATE,
        n_restarts: int = HMM_N_RESTARTS,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.model = None
        self._state_order = None
        self.is_fitted = False

    def fit(self, features: pd.DataFrame) -> "RegimeHMM":
        """
        Fit HMM using multiple random restarts, keeping the model with
        the highest log-likelihood. This avoids local optima (e.g. one
        state collapsing to cover a multi-year period).

        Parameters
        ----------
        features : pd.DataFrame
            Observation sequence (rows = timesteps, cols = features). No NaNs.
        """
        X = features.values
        best_model = None
        best_score = -np.inf

        for i in range(self.n_restarts):
            seed = self.random_state + i
            candidate = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=seed,
            )
            try:
                candidate.fit(X)
                score = candidate.score(X)
                if score > best_score:
                    best_score = score
                    best_model = candidate
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("All HMM fitting restarts failed.")

        print(f"  Best log-likelihood: {best_score:.2f} (across {self.n_restarts} restarts)")
        self.model = best_model
        self._compute_state_order(features)
        self.is_fitted = True
        return self

    def _compute_state_order(self, features: pd.DataFrame) -> None:
        """
        Re-order states by mean of the first feature column (1-day return),
        so state 0 is always the worst-return (crisis) state.
        """
        raw_states = self.model.predict(features.values)
        first_feature = features.iloc[:, 0]
        mean_returns = {
            s: first_feature[raw_states == s].mean() for s in range(self.n_states)
        }
        ordered = sorted(mean_returns, key=mean_returns.get)   # ascending: worst first
        self._state_order = {orig: new for new, orig in enumerate(ordered)}

    def predict(self, features: pd.DataFrame, min_persistence: int = 5) -> pd.Series:
        """
        Predict regime labels using smoothed posterior decoding rather than
        raw Viterbi. Avoids the barcode pattern where states flip day-to-day.

        Method:
          1. Compute forward-backward smoothed state probabilities
          2. Assign each day to the argmax smoothed state
          3. Apply a minimum persistence filter: runs shorter than
             min_persistence days are absorbed into the surrounding state.

        Parameters
        ----------
        features        : pd.DataFrame of HMM observations
        min_persistence : minimum consecutive days before a state transition
                          is accepted (default 5 trading days = 1 week)
        """
        self._check_fitted()

        # Step 1: get forward-backward smoothed probabilities
        probs = self.predict_proba(features)

        # Step 2: apply a rolling mean to smooth out single-day probability spikes
        # before taking argmax — this is the key step to avoid barcode patterns
        smoothing_window = max(min_persistence, 5)
        smoothed = probs.rolling(window=smoothing_window, center=True, min_periods=1).mean()

        # Step 3: argmax of smoothed probabilities
        raw_states = smoothed.values.argmax(axis=1)

        # Step 4: persistence filter as a final cleanup
        filtered = self._apply_persistence_filter(raw_states, min_persistence)
        return pd.Series(filtered, index=features.index, name="hmm_state")

    @staticmethod
    def _apply_persistence_filter(states: np.ndarray, min_days: int) -> np.ndarray:
        """
        Absorb regime runs shorter than min_days into the preceding state.
        Single pass: when a run ends, if it was too short, backfill with
        the state that preceded it.
        """
        result = states.copy()
        n = len(result)
        i = 0
        while i < n:
            j = i
            while j < n and result[j] == result[i]:
                j += 1
            run_len = j - i
            if run_len < min_days and i > 0:
                result[i:j] = result[i - 1]
            i = j
        return result

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        State posterior probabilities for each timestep.

        Returns
        -------
        pd.DataFrame with columns [state_0, state_1, state_2], re-ordered.
        """
        self._check_fitted()
        raw_probs = self.model.predict_proba(features.values)
        # Re-order columns to match reordered state indices
        n = self.n_states
        reordered_cols = [None] * n
        for orig, new in self._state_order.items():
            reordered_cols[new] = raw_probs[:, orig]
        df = pd.DataFrame(
            np.column_stack(reordered_cols),
            index=features.index,
            columns=[f"state_{i}" for i in range(n)],
        )
        return df

    def label_series(self, states: pd.Series) -> pd.Series:
        """Map integer state indices to human-readable labels."""
        return states.map(STATE_LABELS)

    def plot_regimes(
        self,
        prices: pd.Series,
        states: pd.Series,
        title: str = "Market Regimes",
    ) -> plt.Figure:
        """
        Plot price series with regime background shading.

        Parameters
        ----------
        prices : pd.Series  — benchmark price index (e.g. S&P 500)
        states : pd.Series  — integer regime labels from predict()
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(prices.index, prices.values, color="black", linewidth=0.8, zorder=2)

        # Shade background by regime
        prev_state = states.iloc[0]
        start_date = states.index[0]
        for date, state in states.items():
            if state != prev_state:
                ax.axvspan(start_date, date, alpha=0.25, color=STATE_COLOURS[prev_state], zorder=1)
                start_date = date
                prev_state = state
        ax.axvspan(start_date, states.index[-1], alpha=0.25, color=STATE_COLOURS[prev_state], zorder=1)

        patches = [
            mpatches.Patch(color=STATE_COLOURS[i], alpha=0.5, label=STATE_LABELS[i])
            for i in range(self.n_states)
        ]
        ax.legend(handles=patches, loc="upper left")
        ax.set_title(title)
        ax.set_ylabel("Price")
        plt.tight_layout()
        return fig

    def transition_matrix(self) -> pd.DataFrame:
        """Return the fitted transition matrix as a labelled DataFrame."""
        self._check_fitted()
        labels = [STATE_LABELS[i] for i in range(self.n_states)]
        return pd.DataFrame(
            self.model.transmat_,
            index=labels,
            columns=labels,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)
    idx = pd.date_range("2000-01-01", periods=500, freq="B")
    features = pd.DataFrame(
        np.random.randn(500, 2), index=idx, columns=["ret_1d", "vol_21d"]
    )
    model = RegimeHMM()
    model.fit(features)
    states = model.predict(features)
    probs  = model.predict_proba(features)
    print(states.value_counts())
    print(model.transition_matrix())
