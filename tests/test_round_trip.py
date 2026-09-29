"""Train small models, save them, and evaluate them with the evaluation script."""
import copy

import pytest

import config
import test_models_refactored
from train_models_refactored import ExoplanetMLPipeline


@pytest.fixture
def small_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_FIXED_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "plots")
    (tmp_path / "plots").mkdir()
    params = copy.deepcopy(config.HYPERPARAMETERS)
    for model_params in params.values():
        model_params["n_estimators"] = 10
    # Single-threaded RF: with few trees, vote ties are common and threaded
    # probability summation can break them differently between two predicts.
    params["random_forest"]["n_jobs"] = 1
    monkeypatch.setattr(config, "HYPERPARAMETERS", params)

    pipeline = ExoplanetMLPipeline()
    for dataset in ["cumulative", "k2pandc", "toi"]:
        pipeline.train_dataset(dataset, model_types=config.MODEL_TYPES)
    pipeline.save_models()
    return pipeline


def test_train_then_evaluate_scores_every_model(small_models):
    tester = test_models_refactored.ModelTester()
    tester.load_models()
    assert tester.missing == []

    for dataset in ["cumulative", "k2pandc", "toi"]:
        for model_type in config.MODEL_TYPES:
            assert tester.evaluate_model(dataset, model_type), f"{dataset}_{model_type} was not scored"
    assert len(tester.results) == 3 * len(config.MODEL_TYPES)

    assert test_models_refactored.main() == 0


def test_evaluation_uses_the_held_out_test_split(small_models):
    tester = test_models_refactored.ModelTester()
    tester.load_models()
    for dataset in ["cumulative", "k2pandc", "toi"]:
        for model_type in config.MODEL_TYPES:
            scored = tester.evaluate_model(dataset, model_type)
            trained = small_models.results[dataset][model_type]["test"]
            assert scored["accuracy"] == pytest.approx(trained["accuracy"])
            assert scored["f1"] == pytest.approx(trained["f1"])
