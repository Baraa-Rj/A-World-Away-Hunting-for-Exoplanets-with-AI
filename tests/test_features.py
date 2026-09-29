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
