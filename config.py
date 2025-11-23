"""
Configuration module for NASA Exoplanet ML Analysis Project.
Contains all file paths, constants, and hyperparameters.
"""
import os
from pathlib import Path
from typing import Dict, Tuple, Any

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_FIXED_DIR = MODELS_DIR / "models_fixed"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MODELS_FIXED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Raw data file paths (update these with your actual filenames or set via environment variables)
RAW_FILES = {
    "cumulative": os.getenv(
        "CUMULATIVE_FILE",
        str(RAW_DATA_DIR / "cumulative_2025.10.03_00.50.03.csv")
    ),
    "k2pandc": os.getenv(
        "K2PANDC_FILE",
        str(RAW_DATA_DIR / "k2pandc_2025.10.03_00.53.03.csv")
    ),
    "toi": os.getenv(
        "TOI_FILE",
        str(RAW_DATA_DIR / "TOI_2025.10.03_00.51.26.csv")
    ),
}

# Processed data file paths
PROCESSED_FILES = {
    "cumulative": str(PROCESSED_DATA_DIR / "cumulative_processed.csv"),
    "k2pandc": str(PROCESSED_DATA_DIR / "k2pandc_processed.csv"),
    "toi": str(PROCESSED_DATA_DIR / "toi_processed.csv"),
}

# Cleaned data file paths
CLEANED_FILES = {
    "cumulative": str(PROCESSED_DATA_DIR / "cumulative_cleaned.csv"),
    "k2pandc": str(PROCESSED_DATA_DIR / "k2pandc_cleaned.csv"),
    "toi": str(PROCESSED_DATA_DIR / "toi_cleaned.csv"),
}

# Model file paths
MODEL_FILES = {
    "cumulative": {
        "rf": str(MODELS_FIXED_DIR / "cumulative_rf.pkl"),
        "xgb": str(MODELS_FIXED_DIR / "cumulative_xgb.pkl"),
        "lgb": str(MODELS_FIXED_DIR / "cumulative_lgb.pkl"),
    },
    "k2pandc": {
        "rf": str(MODELS_FIXED_DIR / "k2pandc_rf.pkl"),
        "xgb": str(MODELS_FIXED_DIR / "k2pandc_xgb.pkl"),
        "lgb": str(MODELS_FIXED_DIR / "k2pandc_lgb.pkl"),
    },
    "toi": {
        "rf": str(MODELS_FIXED_DIR / "toi_rf.pkl"),
        "xgb": str(MODELS_FIXED_DIR / "toi_xgb.pkl"),
        "lgb": str(MODELS_FIXED_DIR / "toi_lgb.pkl"),
    },
}

# Preprocessing constants
PREPROCESSING = {
    "IQR_THRESHOLD": 3.0,  # Multiplier for IQR outlier detection
    "RANDOM_STATE": 42,    # Random seed for reproducibility
    "HABITABLE_ZONE_TEMP": (200, 350),  # Temperature range in Kelvin
}

# Data splitting constants
DATA_SPLIT = {
    "TEST_SIZE": 0.15,           # 15% for test set
    "VALIDATION_SIZE": 0.15,     # 15% of remaining for validation (15% of 85% = ~12.75% of total)
    "RANDOM_STATE": 42,
    "STRATIFY": True,            # Use stratified splitting
}

# Class imbalance handling
IMBALANCE = {
    "USE_SMOTE": True,
    "SMOTE_RANDOM_STATE": 42,
    "SMOTE_K_NEIGHBORS": 5,
}

# Model hyperparameters
HYPERPARAMETERS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1,
    },
}

# Hyperparameter tuning grids
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9, -1],
        "learning_rate": [0.01, 0.1, 0.3],
        "num_leaves": [15, 31, 63],
    },
}

# Cross-validation settings
CROSS_VALIDATION = {
    "N_FOLDS": 5,
    "SHUFFLE": True,
    "RANDOM_STATE": 42,
}

# Logging configuration
LOGGING = {
    "LEVEL": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    "LOG_FILE": str(BASE_DIR / "exoplanet_ml.log"),
}

# Target columns that should be excluded from features (prevent data leakage)
TARGET_COLUMNS = {
    "cumulative": [
        "koi_disposition", "koi_pdisposition", "koi_score",
    ],
    "k2pandc": [
        "disposition", "pl_controv_flag",
    ],
    "toi": [
        "tfopwg_disp", "toi_disposition",
    ],
}

# Dataset-specific column mappings
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cumulative": {
        "target_column": "koi_disposition",
        "planetary_features": [
            "koi_period", "koi_prad", "koi_depth", "koi_duration",
            "koi_teq", "koi_insol", "koi_impact",
        ],
        "stellar_features": [
            "koi_steff", "koi_slogg", "koi_srad", "koi_smass",
        ],
        "derived_features": [
            "planet_star_radius_ratio", "planet_volume", "in_habitable_zone",
        ],
    },
    "k2pandc": {
        "target_column": "disposition",
        "planetary_features": [
            "pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt",
        ],
        "stellar_features": [
            "st_teff", "st_logg", "st_rad", "st_mass",
        ],
        "derived_features": [
            "planet_density", "surface_gravity", "escape_velocity",
        ],
    },
    "toi": {
        "target_column": "tfopwg_disp",
        "planetary_features": [
            "pl_orbper", "pl_rade", "pl_eqt", "pl_insol",
        ],
        "stellar_features": [
            "st_tmag", "st_dist", "st_rad",
        ],
        "derived_features": [
            "transit_snr_approx", "distance_modulus", "in_habitable_zone",
        ],
    },
}


def get_raw_file_path(dataset_name: str) -> str:
    """Get the raw file path for a given dataset."""
    if dataset_name not in RAW_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(RAW_FILES.keys())}")
    return RAW_FILES[dataset_name]


def get_processed_file_path(dataset_name: str) -> str:
    """Get the processed file path for a given dataset."""
    if dataset_name not in PROCESSED_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(PROCESSED_FILES.keys())}")
    return PROCESSED_FILES[dataset_name]


def get_cleaned_file_path(dataset_name: str) -> str:
    """Get the cleaned file path for a given dataset."""
    if dataset_name not in CLEANED_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(CLEANED_FILES.keys())}")
    return CLEANED_FILES[dataset_name]


def get_model_file_path(dataset_name: str, model_type: str) -> str:
    """Get the model file path for a given dataset and model type."""
    if dataset_name not in MODEL_FILES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(MODEL_FILES.keys())}")
    if model_type not in MODEL_FILES[dataset_name]:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of {list(MODEL_FILES[dataset_name].keys())}")
    return MODEL_FILES[dataset_name][model_type]


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get the configuration for a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]
