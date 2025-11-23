# Refactored Code Guide

This document explains the refactored codebase improvements and how to use the new modules.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Improvements](#key-improvements)
- [File Structure](#file-structure)
- [Configuration System](#configuration-system)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)

## 🎯 Overview

The codebase has been refactored with the following goals:
- ✅ Remove all hardcoded paths and constants
- ✅ Add comprehensive error handling
- ✅ Consolidate duplicate code
- ✅ Add type hints and logging
- ✅ Implement cross-validation and hyperparameter tuning

## 🚀 Key Improvements

### 1. Centralized Configuration (`config.py`)

All paths, constants, and hyperparameters are now in one place:

```python
import config

# Get file paths
raw_file = config.get_raw_file_path('cumulative')
processed_file = config.get_processed_file_path('cumulative')
model_file = config.get_model_file_path('cumulative', 'rf')

# Access constants
iqr_threshold = config.PREPROCESSING['IQR_THRESHOLD']
habitable_zone = config.PREPROCESSING['HABITABLE_ZONE_TEMP']

# Get hyperparameters
rf_params = config.HYPERPARAMETERS['random_forest']
```

**Environment Variable Support:**
```bash
# Override file paths via environment variables
export CUMULATIVE_FILE="/custom/path/to/cumulative.csv"
export K2PANDC_FILE="/custom/path/to/k2pandc.csv"
export TOI_FILE="/custom/path/to/toi.csv"
```

### 2. Type Hints

All functions now have type annotations:

```python
def load_data(self, dataset_name: str) -> pd.DataFrame:
    """Load dataset with proper type hints."""
    ...

def preprocess_dataset(
    self,
    dataset_name: str,
    save: bool = True
) -> pd.DataFrame:
    """Preprocess with typed parameters and return value."""
    ...
```

### 3. Logging Framework

Replace print statements with proper logging:

```python
import logging
logger = logging.getLogger(__name__)

# Configure in config.py
config.LOGGING = {
    "LEVEL": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_FILE": "exoplanet_ml.log"
}

# Use in code
logger.info("Processing dataset...")
logger.warning("Potential issue detected")
logger.error("Operation failed", exc_info=True)
```

### 4. Comprehensive Error Handling

All operations now handle errors gracefully:

```python
try:
    df = self.load_data(dataset_name)
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except ValueError as e:
    logger.error(f"Invalid data: {e}")
    raise
```

### 5. Consolidated Code

**Before:** Three separate methods (`preprocess_cumulative`, `preprocess_k2pandc`, `preprocess_toi`)

**After:** One generic method:

```python
def preprocess_dataset(self, dataset_name: str, save: bool = True) -> pd.DataFrame:
    """Generic preprocessing for any dataset."""
    dataset_config = config.get_dataset_config(dataset_name)
    # Use dataset-specific configuration
    ...
```

### 6. Cross-Validation & Hyperparameter Tuning

New ML capabilities:

```python
# Cross-validation
cv_results = pipeline.cross_validate_model(model, X, y, "Random Forest")

# Hyperparameter tuning
best_model, best_params = pipeline.tune_hyperparameters('random_forest', X_train, y_train)

# Use in training
pipeline.train_dataset('cumulative', model_types=['random_forest', 'xgboost'], tune=True)
```

## 📁 File Structure

### Refactored Files

```
├── config.py                                      # NEW: Centralized configuration
├── scripts/
│   ├── preprocess_data_refactored.py             # NEW: Refactored preprocessor
│   ├── train_models_refactored.py                # NEW: Refactored ML pipeline
│   ├── test_models_refactored.py                 # NEW: Refactored model tester
│   ├── create_cleaned_datasets_refactored.py     # NEW: Refactored dataset cleaner
│   │
│   ├── preprocess_data.py                         # OLD: Original version
│   ├── train_models_fixed.py                      # OLD: Original version
│   ├── test_models.py                             # OLD: Original version
│   └── create_cleaned_datasets.py                 # OLD: Original version
```

### When to Use Which Version

**Use Refactored Versions (`*_refactored.py`) for:**
- ✅ Production deployments
- ✅ New development
- ✅ Code maintenance
- ✅ Better error messages and debugging
- ✅ Configurability via environment variables

**Keep Original Versions for:**
- 📚 Reference
- 🔄 Backward compatibility (if needed)
- 📊 Reproducing original results

## ⚙️ Configuration System

### Dataset-Specific Configuration

Each dataset has its own configuration in `config.DATASET_CONFIGS`:

```python
config.DATASET_CONFIGS = {
    "cumulative": {
        "target_column": "koi_disposition",
        "planetary_features": ["koi_period", "koi_prad", ...],
        "stellar_features": ["koi_steff", "koi_slogg", ...],
        "derived_features": ["planet_star_radius_ratio", ...]
    },
    ...
}
```

### Hyperparameter Configuration

```python
config.HYPERPARAMETERS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        ...
    }
}
```

### Tuning Parameter Grids

```python
config.PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10]
    },
    ...
}
```

## 💻 Usage Examples

### 1. Preprocessing Data

```python
from scripts.preprocess_data_refactored import ExoplanetPreprocessor

# Initialize
preprocessor = ExoplanetPreprocessor()

# Load all datasets
preprocessor.load_all_datasets()

# Preprocess specific dataset
preprocessor.preprocess_dataset('cumulative', save=True)

# Or preprocess all
preprocessor.preprocess_all_datasets(save=True)

# Generate summary
summary = preprocessor.generate_summary()
```

### 2. Creating Cleaned Datasets

```python
from scripts.create_cleaned_datasets_refactored import DatasetCleaner

# Initialize
cleaner = DatasetCleaner()

# Clean specific dataset
df_clean = cleaner.clean_dataset('cumulative', save=True)

# Or clean all
cleaner.clean_all_datasets(save=True)

# Generate report
cleaner.generate_summary_report()
```

### 3. Training Models

```python
from scripts.train_models_refactored import ExoplanetMLPipeline

# Initialize
pipeline = ExoplanetMLPipeline()

# Train with default parameters (fast)
pipeline.train_dataset(
    'cumulative',
    model_types=['random_forest', 'xgboost'],
    tune=False
)

# Train with hyperparameter tuning (slower, better results)
pipeline.train_dataset(
    'cumulative',
    model_types=['random_forest', 'xgboost'],
    tune=True
)

# Train all datasets
for dataset in ['cumulative', 'k2pandc', 'toi']:
    pipeline.train_dataset(dataset, model_types=['random_forest'], tune=False)

# Generate report
pipeline.generate_summary_report()

# Save models
pipeline.save_models()
```

### 4. Testing Models

```python
from scripts.test_models_refactored import ModelTester

# Initialize
tester = ModelTester()

# Load models
tester.load_models(model_types=['rf', 'xgb'])

# Evaluate specific model
results = tester.evaluate_model('cumulative', 'rf')

# Evaluate all
for dataset in ['cumulative', 'k2pandc', 'toi']:
    for model_type in ['rf', 'xgb']:
        tester.evaluate_model(dataset, model_type)

# Generate summary
tester.generate_summary_report()
```

### 5. End-to-End Pipeline

```python
# Complete workflow
from scripts.preprocess_data_refactored import ExoplanetPreprocessor
from scripts.create_cleaned_datasets_refactored import DatasetCleaner
from scripts.train_models_refactored import ExoplanetMLPipeline
from scripts.test_models_refactored import ModelTester

# 1. Preprocess
preprocessor = ExoplanetPreprocessor()
preprocessor.load_all_datasets()
preprocessor.preprocess_all_datasets(save=True)

# 2. Clean
cleaner = DatasetCleaner()
cleaner.clean_all_datasets(save=True)

# 3. Train
pipeline = ExoplanetMLPipeline()
for dataset in ['cumulative', 'k2pandc', 'toi']:
    pipeline.train_dataset(dataset, model_types=['random_forest', 'xgboost'], tune=False)
pipeline.save_models()

# 4. Test
tester = ModelTester()
tester.load_models(model_types=['rf', 'xgb'])
for dataset in ['cumulative', 'k2pandc', 'toi']:
    tester.evaluate_model(dataset, 'rf')
tester.generate_summary_report()
```

## 🔄 Migration Guide

### From Original to Refactored

| Original | Refactored | Change |
|----------|------------|--------|
| `preprocess_data.py` | `preprocess_data_refactored.py` | Import path + use new methods |
| `train_models_fixed.py` | `train_models_refactored.py` | Import path + new parameters |
| `test_models.py` | `test_models_refactored.py` | Import path + model_types param |
| `create_cleaned_datasets.py` | `create_cleaned_datasets_refactored.py` | Import path |

### Breaking Changes

1. **Method Signatures:**
   - Old: `preprocessor.preprocess_cumulative()`
   - New: `preprocessor.preprocess_dataset('cumulative')`

2. **Model Keys:**
   - Old: `models['cumulative_rf']`
   - New: `models['cumulative_rf']` (same, but loaded differently)

3. **File Paths:**
   - Old: Hardcoded `'data/processed/cumulative_cleaned.csv'`
   - New: `config.get_cleaned_file_path('cumulative')`

### Backward Compatibility

If you need the old behavior:
- Keep using original files (`preprocess_data.py`, etc.)
- Original files are preserved for reference
- No breaking changes to original files

## 📊 Logging Configuration

Control logging verbosity:

```python
# In config.py
LOGGING = {
    "LEVEL": "INFO",  # Change to DEBUG for more details
    ...
}

# Or programmatically
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

View logs:
```bash
# Console output (always visible)
# File output
tail -f exoplanet_ml.log
```

## 🐛 Troubleshooting

### Issue: Import errors

```python
# Solution: Add parent directory to path (already in refactored files)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
```

### Issue: File not found

```bash
# Solution: Set environment variables or update config.py
export CUMULATIVE_FILE="/path/to/cumulative.csv"
# Or edit config.RAW_FILES directly
```

### Issue: Model not available

```python
# Check which models are installed
from scripts.train_models_refactored import HAS_XGBOOST, HAS_LIGHTGBM, HAS_SMOTE
print(f"XGBoost: {HAS_XGBOOST}, LightGBM: {HAS_LIGHTGBM}, SMOTE: {HAS_SMOTE}")

# Install missing packages
pip install xgboost lightgbm imbalanced-learn
```

## 📚 Additional Resources

- **Configuration:** See `config.py` for all available settings
- **Type Hints:** Python 3.7+ type hints documentation
- **Logging:** Python logging module documentation
- **ML Models:** Scikit-learn, XGBoost, LightGBM documentation

## ✅ Best Practices

1. **Always use refactored versions for new code**
2. **Configure via `config.py`, not hardcoded values**
3. **Use logging instead of print statements**
4. **Enable hyperparameter tuning for production models**
5. **Run cross-validation before final evaluation**
6. **Check logs for warnings and errors**
7. **Use type hints for new functions**

---

**Questions or Issues?**
Check the logs first, then review configuration settings in `config.py`.
