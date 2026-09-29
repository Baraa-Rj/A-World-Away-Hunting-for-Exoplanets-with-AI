"""Features used for training must not contain the label in any form."""
import pandas as pd
import pytest

import config
from train_models_refactored import ExoplanetMLPipeline

DATASETS = ["cumulative", "k2pandc", "toi"]

# Every column that is the label or an encoding of it.
TARGET_DERIVED = {
    "koi_disposition", "koi_disposition_encoded",
    "koi_pdisposition", "koi_pdisposition_encoded",
    "disposition", "disposition_encoded",
    "tfopwg_disp", "tfopwg_disp_encoded",
}


@pytest.mark.parametrize("dataset", DATASETS)
def test_no_target_derived_column_in_features(dataset):
    df = pd.read_csv(config.get_cleaned_file_path(dataset))
    X, _, _ = ExoplanetMLPipeline().prepare_data(df, dataset)
    assert not TARGET_DERIVED & set(X.columns)


def test_k2_discovery_method_is_not_a_feature():
    df = pd.read_csv(config.get_cleaned_file_path("k2pandc"))
    X, _, _ = ExoplanetMLPipeline().prepare_data(df, "k2pandc")
    assert "discoverymethod_encoded" not in X.columns


@pytest.mark.parametrize("dataset", DATASETS)
def test_training_and_evaluation_prepare_identical_features(dataset):
    from test_models_refactored import ModelTester

    df = pd.read_csv(config.get_cleaned_file_path(dataset))
    pipeline = ExoplanetMLPipeline()
    X_train_side, _, _ = pipeline.prepare_data(df, dataset)

    tester = ModelTester()
    tester.label_encoders = pipeline.label_encoders
    X_eval_side, _, _, _ = tester.load_and_prepare_data(dataset)

    pd.testing.assert_frame_equal(X_train_side, X_eval_side)
