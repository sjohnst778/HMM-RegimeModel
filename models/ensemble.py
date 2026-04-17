# models/ensemble.py
# XGBoost ensemble layer combining HMM state probabilities,
# BOCPD changepoint signals, and macro features to produce
# a single "persistent downturn risk" score [0, 1].

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from config import XGB_PARAMS, ENSEMBLE_TRAIN_CUTOFF


class DownturnEnsemble:
    """
    XGBoost classifier trained on:
    - HMM state probabilities + regime duration
    - BOCPD composite changepoint probability
    - Rolling returns, volatility, cross-asset correlations

    Predicts: P(significant drawdown in next N days)
    """

    def __init__(self, xgb_params: dict = XGB_PARAMS, calibrate: bool = True):
        """
        Parameters
        ----------
        xgb_params : dict
            XGBoost hyperparameters.
        calibrate  : bool
            If True, wraps model in isotonic regression calibration so that
            output probabilities are well-calibrated (important for a risk score).
        """
        base = XGBClassifier(**xgb_params, eval_metric="logloss")
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3) if calibrate else base
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "DownturnEnsemble":
        """
        Fit ensemble on aligned features and binary labels.

        Parameters
        ----------
        features : pd.DataFrame  — output of build_ensemble_features()
        labels   : pd.Series     — output of build_labels(), binary {0, 1}
        """
        aligned = features.join(labels).dropna()
        X = aligned[features.columns]
        y = aligned[labels.name]

        # Handle class imbalance — crisis days are typically 5-15% of history
        pos_rate = y.mean()
        scale_pw = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
        print(f"  Class balance: {pos_rate:.1%} positive, scale_pos_weight={scale_pw:.1f}")

        # Rebuild base model with scale_pos_weight
        base = XGBClassifier(**{**XGB_PARAMS,
                                'eval_metric': 'logloss',
                                'scale_pos_weight': scale_pw})
        from sklearn.calibration import CalibratedClassifierCV
        self.model = CalibratedClassifierCV(base, method='isotonic', cv=3)

        self.feature_names = X.columns.tolist()
        self.model.fit(X.values, y.values)
        self.is_fitted = True
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """
        Produce calibrated downturn risk score for each date.

        Returns
        -------
        pd.Series named 'downturn_risk_score', range [0, 1].
        Note: isotonic calibration produces a step function — use
        predict_proba_raw() for a continuously varying score.
        """
        self._check_fitted()
        X = features[self.feature_names].values
        probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=features.index, name="downturn_risk_score")

    def predict_proba_raw(self, features: pd.DataFrame) -> pd.Series:
        """
        Produce raw (uncalibrated) XGBoost score for each date.

        Preferable for tracking day-to-day changes in risk level since
        isotonic calibration creates a step function that freezes the
        calibrated score when inputs fall in the same plateau.

        Returns
        -------
        pd.Series named 'risk_score_raw', range [0, 1].
        """
        self._check_fitted()
        X = features[self.feature_names].values
        try:
            # Extract base XGBoost estimators from calibrated wrapper
            raw_probs = np.mean([
                est.estimator.predict_proba(X)[:, 1]
                for est in self.model.calibrated_classifiers_
            ], axis=0)
        except AttributeError:
            # Model not calibrated — predict_proba is already raw
            raw_probs = self.model.predict_proba(X)[:, 1]
        return pd.Series(raw_probs, index=features.index, name="risk_score_raw")

    def walk_forward_evaluate(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        min_train_years: int = 5,
        n_splits: int = 5,
    ) -> dict:
        """
        Walk-forward cross-validation using an expanding training window
        on the FULL history.

        Correct approach for this problem:
        - Must include dot-com + GFC in training to learn crisis patterns
        - Restricting to post-2018 leaves almost no crisis examples to learn from
        - Each fold trains on all data up to a cutoff, tests on the next period

        Parameters
        ----------
        features         : full feature DataFrame (1995-present)
        labels           : full label Series
        min_train_years  : minimum years before first test fold (default 5)
        n_splits         : number of folds
        """
        aligned = features.join(labels).dropna()
        col     = labels.name
        n       = len(aligned)
        min_train = int(min_train_years * 252)
        step      = (n - min_train) // n_splits

        fold_results = []
        for i in range(n_splits):
            train_end  = min_train + i * step
            test_start = train_end
            test_end   = min(train_end + step, n)
            if test_end <= test_start:
                break

            train = aligned.iloc[:train_end]
            test  = aligned.iloc[test_start:test_end]

            if len(test[col].unique()) < 2:
                print(f"  Fold {i+1}: skipped — only one class in test set")
                continue

            # Handle class imbalance with scale_pos_weight
            pos_rate = train[col].mean()
            scale_pw = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

            feat_cols = [c for c in aligned.columns if c != col]
            clf = XGBClassifier(**XGB_PARAMS, eval_metric="logloss",
                                scale_pos_weight=scale_pw)
            clf.fit(train[feat_cols].values, train[col].values)
            probs = clf.predict_proba(test[feat_cols].values)[:, 1]
            auc   = roc_auc_score(test[col].values, probs)

            print(f"  Fold {i+1}: "
                  f"train {train.index[0].date()}->{train.index[-1].date()}  "
                  f"test {test.index[0].date()}->{test.index[-1].date()}  "
                  f"crisis={int(test[col].sum())}d  AUC={auc:.3f}")
            fold_results.append({"fold": i+1, "auc": auc,
                                  "test_start": test.index[0],
                                  "test_end":   test.index[-1]})

        aucs = [r["auc"] for r in fold_results]
        return {
            "fold_results": fold_results,
            "mean_auc":     np.mean(aucs) if aucs else np.nan,
            "std_auc":      np.std(aucs)  if aucs else np.nan,
        }

    def feature_importance(self) -> pd.Series:
        """
        Return feature importances from the fitted model.
        Handles both calibrated and uncalibrated XGBoost wrappers.
        """
        self._check_fitted()
        try:
            # CalibratedClassifierCV — extract base estimators
            importances = np.mean([
                est.estimator.feature_importances_
                for est in self.model.calibrated_classifiers_
            ], axis=0)
        except AttributeError:
            try:
                importances = self.model.feature_importances_
            except AttributeError:
                raise RuntimeError(
                    "Could not extract feature importances. "
                    "Try initialising DownturnEnsemble(calibrate=False)."
                )
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def plot_score(self, score: pd.Series, prices: pd.Series | None = None) -> plt.Figure:
        """
        Plot the downturn risk score over time, optionally with price overlay.
        """
        fig, axes = plt.subplots(
            2 if prices is not None else 1, 1,
            figsize=(14, 6 if prices is not None else 3),
            sharex=True,
        )
        axes = np.atleast_1d(axes)

        if prices is not None:
            axes[0].plot(prices.index, prices.values, color="black", linewidth=0.8)
            axes[0].set_ylabel("Price")
            axes[0].set_title("Price vs Downturn Risk Score")

        ax = axes[-1]
        ax.fill_between(score.index, score.values, alpha=0.5, color="#d62728")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="p=0.5")
        ax.axhline(0.7, color="red",   linestyle=":",  linewidth=0.8, label="p=0.7 alert")
        ax.set_ylabel("Downturn Risk Score")
        ax.set_ylim(0, 1)
        ax.legend()
        plt.tight_layout()
        return fig

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
