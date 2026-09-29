"""
Feature preparation shared by training and evaluation, so both see exactly
the same columns and values.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import config

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a cleaned dataset into numeric features and the raw target.

    Args:
        df: Cleaned DataFrame
        dataset_name: Name of dataset

    Returns:
        Tuple of (features, original_target)

    Raises:
        ValueError: If target column not found or no features remain
    """
    dataset_config = config.get_dataset_config(dataset_name)
    target_col = dataset_config["target_column"]

    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    # Get target before dropping
    y = df[target_col].copy()

    # Get all leakage columns for this dataset
    leakage_cols = config.TARGET_COLUMNS.get(dataset_name, [])

    # Drop ALL target and leakage columns
    X = df.drop(columns=[col for col in leakage_cols if col in df.columns])

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])

    # Additional check: remove any column with 'disposition' (or, for
    # cumulative/toi, 'score') in its name
    suspicious_cols = [
        col for col in X.columns
        if 'disposition' in col.lower() or
           (dataset_name in ['cumulative', 'toi'] and 'score' in col.lower())
    ]

    if suspicious_cols:
        logger.warning(f"Removing suspicious columns: {suspicious_cols}")
        X = X.drop(columns=suspicious_cols)

    # Verify we have features
    if X.shape[1] == 0:
        raise ValueError("No features remaining after removing target columns!")

    logger.info(f"\n🔍 DATA LEAKAGE CHECK:")
    logger.info(f"  Removed columns: {[col for col in leakage_cols if col in df.columns]}")
    logger.info(f"  Remaining features: {X.shape[1]}")

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        logger.info(f"  Imputing {X.isnull().sum().sum()} missing values...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().sum().sum() > 0:
        logger.info(f"  Imputing {X.isnull().sum().sum()} inf values...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

    return X, y
