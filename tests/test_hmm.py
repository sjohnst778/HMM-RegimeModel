# tests/test_hmm.py
# Basic unit tests for the RegimeHMM class using synthetic data.

import numpy as np
import pandas as pd
import pytest
from models.hmm import RegimeHMM


@pytest.fixture
def synthetic_features():
    np.random.seed(0)
    idx = pd.date_range("2000-01-01", periods=300, freq="B")
    return pd.DataFrame(
        np.random.randn(300, 2), index=idx, columns=["ret_1d", "vol_21d"]
    )


def test_fit_predict_shape(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.fit(synthetic_features)
    states = model.predict(synthetic_features)
    assert len(states) == len(synthetic_features)


def test_state_values(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.fit(synthetic_features)
    states = model.predict(synthetic_features)
    assert set(states.unique()).issubset({0, 1, 2})


def test_proba_sums_to_one(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.fit(synthetic_features)
    probs = model.predict_proba(synthetic_features)
    np.testing.assert_allclose(probs.sum(axis=1).values, 1.0, atol=1e-5)


def test_transition_matrix_shape(synthetic_features):
    model = RegimeHMM(n_states=3)
    model.fit(synthetic_features)
    tm = model.transition_matrix()
    assert tm.shape == (3, 3)


def test_unfitted_raises(synthetic_features):
    model = RegimeHMM()
    with pytest.raises(RuntimeError):
        model.predict(synthetic_features)
