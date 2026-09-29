"""Cross-validation must not see SMOTE's synthetic rows."""
import copy

import numpy as np
import pytest

import config
import train_models_refactored
from train_models_refactored import ExoplanetMLPipeline


@pytest.mark.parametrize("dataset", ["k2pandc", "toi"])  # both imbalanced enough for SMOTE
def test_smote_only_runs_inside_training_folds(dataset, monkeypatch):
    params = copy.deepcopy(config.HYPERPARAMETERS)
    params["random_forest"]["n_estimators"] = 10
    monkeypatch.setattr(config, "HYPERPARAMETERS", params)

    calls = []
    real_cross_val_score = train_models_refactored.cross_val_score

    def recording_cross_val_score(estimator, X, y, **kwargs):
        calls.append((estimator, len(X)))
        return real_cross_val_score(estimator, X, y, **kwargs)

    monkeypatch.setattr(train_models_refactored, "cross_val_score", recording_cross_val_score)

    pipeline = ExoplanetMLPipeline()
    pipeline.train_dataset(dataset, model_types=["random_forest"])

    n_rows = len(pipeline.load_data(dataset))
    n_test = len(pipeline.test_indices[dataset])
    assert len(calls) == 1
    estimator, n_cv_rows = calls[0]
    # Only real train + validation rows are cross-validated ...
    assert n_cv_rows == n_rows - n_test
    # ... and resampling happens per fold, inside the estimator.
    assert "smote" in dict(estimator.steps)

    cv_mean = pipeline.results[dataset]["random_forest"]["training"]["cv_mean"]
    assert not np.isnan(cv_mean) and cv_mean > 0
