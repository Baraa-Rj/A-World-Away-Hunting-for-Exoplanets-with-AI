"""
Refactored ML Training Pipeline with cross-validation and hyperparameter tuning.
Includes type hints, logging, error handling, and uses config module.
"""
import sys
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# Optional imports with error handling
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features import prepare_features

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING["LEVEL"]),
    format=config.LOGGING["FORMAT"],
    datefmt=config.LOGGING["DATE_FORMAT"],
    handlers=[
        logging.FileHandler(config.LOGGING["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ExoplanetMLPipeline:
    """
    Complete ML training pipeline with cross-validation and hyperparameter tuning.

    Features:
    - Multiple model support (RF, XGBoost, LightGBM)
    - Cross-validation for robust evaluation
    - Hyperparameter tuning with GridSearchCV
    - SMOTE for class imbalance
    - Comprehensive logging and error handling
    - No data leakage
    """

    def __init__(self):
        """Initialize the ML pipeline."""
        self.results: Dict[str, Dict] = {}
        self.models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.best_params: Dict[str, Dict] = {}
        self.test_indices: Dict[str, List] = {}

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load cleaned dataset.

        Args:
            dataset_name: Name of dataset ('cumulative', 'k2pandc', or 'toi')

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset is empty or invalid
        """
        logger.info("="*80)
        logger.info(f"LOADING {dataset_name.upper()} DATASET")
        logger.info("="*80)

        try:
            file_path = config.get_cleaned_file_path(dataset_name)

            if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {file_path}\n"
                    f"Please run data cleaning first."
                )

            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(f"Dataset {dataset_name} is empty")

            logger.info(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {str(e)}")
            raise

    def prepare_data(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Prepare features and target without data leakage.

        Args:
            df: Input DataFrame
            dataset_name: Name of dataset

        Returns:
            Tuple of (features, encoded_target, original_target)

        Raises:
            ValueError: If target column not found or no features remain
        """
        logger.info("Preparing data...")

        try:
            X, y = prepare_features(df, dataset_name)

            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.label_encoders[dataset_name] = le

            logger.info(f"\n  ✓ Features: {X.shape[1]}")
            logger.info(f"  ✓ Samples: {len(X)}")
            logger.info(f"  ✓ Classes: {len(le.classes_)}")
            logger.info(f"  Class distribution:")
            for cls, count in zip(*np.unique(y_encoded, return_counts=True)):
                logger.info(f"    {le.classes_[cls]}: {count} ({count/len(y)*100:.1f}%)")

            return X, y_encoded, y

        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")
            raise

    def create_splits(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train/validation/test splits.

        Args:
            X: Features
            y: Target

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Creating train/validation/test splits...")

        test_size = config.DATA_SPLIT["TEST_SIZE"]
        val_size = config.DATA_SPLIT["VALIDATION_SIZE"]

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=config.DATA_SPLIT["RANDOM_STATE"],
            stratify=y if config.DATA_SPLIT["STRATIFY"] else None
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=config.DATA_SPLIT["RANDOM_STATE"],
            stratify=y_temp if config.DATA_SPLIT["STRATIFY"] else None
        )

        logger.info(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle class imbalance using SMOTE if available.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (resampled_X, resampled_y)
        """
        if not config.IMBALANCE["USE_SMOTE"] or not HAS_SMOTE:
            if not HAS_SMOTE:
                logger.warning("SMOTE not available. Install imbalanced-learn for class balancing.")
            return X_train, y_train

        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        imbalance_ratio = counts.max() / counts.min()

        if imbalance_ratio > 3:
            logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}). Applying SMOTE...")

            try:
                smote = SMOTE(
                    random_state=config.IMBALANCE["SMOTE_RANDOM_STATE"],
                    k_neighbors=min(
                        config.IMBALANCE["SMOTE_K_NEIGHBORS"],
                        counts.min() - 1  # Can't have more neighbors than samples
                    )
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

                logger.info(f"  Before SMOTE: {len(y_train)} samples")
                logger.info(f"  After SMOTE:  {len(y_resampled)} samples")

                return pd.DataFrame(X_resampled, columns=X_train.columns), y_resampled

            except Exception as e:
                logger.warning(f"SMOTE failed: {str(e)}. Proceeding without resampling.")
                return X_train, y_train
        else:
            logger.info("Classes relatively balanced. Skipping SMOTE.")
            return X_train, y_train

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, float]:
        """
        Perform cross-validation on a model.

        Args:
            model: Model to evaluate
            X: Features
            y: Target
            model_name: Name of the model for logging

        Returns:
            Dictionary with cross-validation scores
        """
        logger.info(f"\nPerforming {config.CROSS_VALIDATION['N_FOLDS']}-fold cross-validation for {model_name}...")

        try:
            cv = StratifiedKFold(
                n_splits=config.CROSS_VALIDATION["N_FOLDS"],
                shuffle=config.CROSS_VALIDATION["SHUFFLE"],
                random_state=config.CROSS_VALIDATION["RANDOM_STATE"]
            )

            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )

            logger.info(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")

            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {str(e)}")
            return {'cv_mean': 0.0, 'cv_std': 0.0, 'cv_scores': []}

    def tune_hyperparameters(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Tuple[Any, Dict]:
        """
        Tune hyperparameters using GridSearchCV.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            X_train: Training features
            y_train: Training target

        Returns:
            Tuple of (best_model, best_params)
        """
        logger.info(f"\nTuning hyperparameters for {model_type}...")

        try:
            # Get base model
            if model_type == 'random_forest':
                base_model = RandomForestClassifier(
                    random_state=config.HYPERPARAMETERS['random_forest']['random_state'],
                    n_jobs=-1
                )
            elif model_type == 'xgboost' and HAS_XGBOOST:
                base_model = xgb.XGBClassifier(
                    random_state=config.HYPERPARAMETERS['xgboost']['random_state']
                )
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                base_model = lgb.LGBMClassifier(
                    random_state=config.HYPERPARAMETERS['lightgbm']['random_state']
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Get parameter grid
            param_grid = config.PARAM_GRIDS.get(model_type, {})

            if not param_grid:
                logger.warning(f"No parameter grid found for {model_type}. Using default params.")
                base_model.set_params(**config.HYPERPARAMETERS[model_type])
                return base_model, {}

            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,  # Reduced CV for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            logger.info(f"  Testing {sum([len(v) for v in param_grid.values()])} parameter combinations...")
            grid_search.fit(X_train, y_train)

            logger.info(f"  ✓ Best parameters: {grid_search.best_params_}")
            logger.info(f"  ✓ Best CV score: {grid_search.best_score_*100:.2f}%")

            return grid_search.best_estimator_, grid_search.best_params_

        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            logger.info("  Falling back to default parameters...")

            # Return model with default params
            if model_type == 'random_forest':
                model = RandomForestClassifier(**config.HYPERPARAMETERS['random_forest'])
            elif model_type == 'xgboost' and HAS_XGBOOST:
                model = xgb.XGBClassifier(**config.HYPERPARAMETERS['xgboost'])
            elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                model = lgb.LGBMClassifier(**config.HYPERPARAMETERS['lightgbm'])
            else:
                raise

            return model, {}

    def train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        dataset_name: str,
        tune: bool = False
    ) -> Dict[str, Any]:
        """
        Train a model with optional hyperparameter tuning.

        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            dataset_name: Name of dataset
            tune: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info(f"TRAINING {model_type.upper()} - {dataset_name.upper()}")
        logger.info("="*80)

        start_time = time.time()

        try:
            # Tune or use default params
            if tune:
                model, best_params = self.tune_hyperparameters(model_type, X_train, y_train)
                self.best_params[f'{dataset_name}_{model_type}'] = best_params
            else:
                logger.info("Using default hyperparameters...")
                if model_type == 'random_forest':
                    model = RandomForestClassifier(**config.HYPERPARAMETERS['random_forest'])
                elif model_type == 'xgboost' and HAS_XGBOOST:
                    model = xgb.XGBClassifier(**config.HYPERPARAMETERS['xgboost'])
                elif model_type == 'lightgbm' and HAS_LIGHTGBM:
                    model = lgb.LGBMClassifier(**config.HYPERPARAMETERS['lightgbm'])
                else:
                    raise ValueError(f"Model type {model_type} not available")

            # Train
            logger.info("Training model...")
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            val_acc = accuracy_score(y_val, y_val_pred)

            training_time = time.time() - start_time

            logger.info(f"\n✓ Training complete!")
            logger.info(f"  Training Time: {training_time:.2f} seconds")
            logger.info(f"  Train Accuracy: {train_acc*100:.2f}%")
            logger.info(f"  Val Accuracy:   {val_acc*100:.2f}%")

            # Check for overfitting
            overfit_gap = (train_acc - val_acc) * 100
            if overfit_gap > 10:
                logger.warning(f"  ⚠️  Overfitting detected! Gap: {overfit_gap:.2f}%")
            elif overfit_gap < 0:
                logger.info(f"  ✓ Model generalizes well (val > train)")
            else:
                logger.info(f"  ✓ Good generalization (gap: {overfit_gap:.2f}%)")

            # Store model
            model_key = f'{dataset_name}_{model_type}'
            self.models[model_key] = model

            # Cross-validation
            cv_results = self.cross_validate_model(
                model,
                pd.concat([X_train, X_val]),
                np.concatenate([y_train, y_val]),
                model_type
            )

            return {
                'model_type': model_type,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'time': training_time,
                **cv_results
            }

        except Exception as e:
            logger.error(f"Training failed for {model_type}: {str(e)}")
            raise

    def evaluate_on_test(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        dataset_name: str,
        model_type: str
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target
            dataset_name: Name of dataset
            model_type: Type of model

        Returns:
            Dictionary with test results
        """
        logger.info("="*80)
        logger.info(f"TEST SET EVALUATION - {dataset_name.upper()} - {model_type.upper()}")
        logger.info("="*80)

        try:
            model_key = f'{dataset_name}_{model_type}'
            model = self.models[model_key]
            le = self.label_encoders[dataset_name]

            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            logger.info(f"\nTest Results:")
            logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
            logger.info(f"  Precision: {precision*100:.2f}%")
            logger.info(f"  Recall:    {recall*100:.2f}%")
            logger.info(f"  F1-Score:  {f1*100:.2f}%")

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                logger.info(f"\n🔍 TOP 10 IMPORTANT FEATURES:")
                feature_names = X_test.columns
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]

                for i, idx in enumerate(indices, 1):
                    logger.info(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")

            return {
                'dataset': dataset_name,
                'model_type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def train_dataset(
        self,
        dataset_name: str,
        model_types: Optional[List[str]] = None,
        tune: bool = False
    ) -> Dict[str, Any]:
        """
        Complete training pipeline for one dataset with multiple models.

        Args:
            dataset_name: Name of dataset
            model_types: List of model types to train (default: ['random_forest'])
            tune: Whether to perform hyperparameter tuning

        Returns:
            Dictionary with all results
        """
        logger.info("#"*80)
        logger.info(f"# TRAINING PIPELINE: {dataset_name.upper()}")
        logger.info("#"*80)

        if model_types is None:
            model_types = ['random_forest']

        try:
            # Load data
            df = self.load_data(dataset_name)

            # Prepare data
            X, y_encoded, y_original = self.prepare_data(df, dataset_name)

            # Create splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X, y_encoded)
            self.test_indices[dataset_name] = X_test.index.tolist()

            # Handle imbalance
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)

            # Train each model type
            dataset_results = {}
            for model_type in model_types:
                try:
                    # Train
                    train_results = self.train_model(
                        model_type,
                        X_train_balanced,
                        y_train_balanced,
                        X_val,
                        y_val,
                        dataset_name,
                        tune=tune
                    )

                    # Test
                    test_results = self.evaluate_on_test(
                        X_test, y_test, dataset_name, model_type
                    )

                    dataset_results[model_type] = {
                        'training': train_results,
                        'test': test_results
                    }

                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {str(e)}")
                    continue

            self.results[dataset_name] = dataset_results
            return dataset_results

        except Exception as e:
            logger.error(f"Training pipeline failed for {dataset_name}: {str(e)}")
            raise

    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        logger.info("\n" + "="*80)
        logger.info("FINAL TRAINING SUMMARY")
        logger.info("="*80 + "\n")

        for dataset_name, model_results in self.results.items():
            logger.info(f"\n{dataset_name.upper()}:")
            logger.info("-" * 80)
            logger.info(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'F1-Score':<12}")
            logger.info("-" * 80)

            for model_type, results in model_results.items():
                train_res = results['training']
                test_res = results['test']

                logger.info(
                    f"{model_type:<20} "
                    f"{train_res['train_acc']*100:>10.2f}% "
                    f"{train_res['val_acc']*100:>10.2f}% "
                    f"{test_res['accuracy']*100:>10.2f}% "
                    f"{test_res['f1']*100:>10.2f}%"
                )

        logger.info("\n" + "="*80)

    def save_models(self) -> None:
        """Save all trained models and encoders."""
        logger.info("\nSaving models...")

        try:
            # Ensure directory exists
            config.MODELS_FIXED_DIR.mkdir(parents=True, exist_ok=True)

            # Save each model
            for name, model in self.models.items():
                dataset_name, model_type = name.split('_', 1)
                filepath = config.get_model_file_path(dataset_name, model_type)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"  ✓ Saved: {filepath}")

            # Save label encoders
            encoder_path = config.MODELS_FIXED_DIR / 'label_encoders.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            logger.info(f"  ✓ Saved: {encoder_path}")

            # Save held-out test split so evaluation uses the same rows
            split_path = config.get_test_split_path()
            with open(split_path, 'wb') as f:
                pickle.dump(self.test_indices, f)
            logger.info(f"  ✓ Saved: {split_path}")

            # Save best params if tuning was performed
            if self.best_params:
                params_path = config.MODELS_FIXED_DIR / 'best_params.pkl'
                with open(params_path, 'wb') as f:
                    pickle.dump(self.best_params, f)
                logger.info(f"  ✓ Saved: {params_path}")

            logger.info(f"\n✅ All models saved to {config.MODELS_FIXED_DIR}!")

        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            raise


def main(tune_hyperparameters: bool = False):
    """
    Main training pipeline.

    Args:
        tune_hyperparameters: Whether to perform hyperparameter tuning
    """
    logger.info("="*80)
    logger.info("NASA EXOPLANET ML TRAINING PIPELINE - REFACTORED")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Hyperparameter tuning: {'ENABLED' if tune_hyperparameters else 'DISABLED'}")

    pipeline = ExoplanetMLPipeline()

    # Determine which models to use
    model_types = ['random_forest']
    if HAS_XGBOOST:
        model_types.append('xgboost')
    if HAS_LIGHTGBM:
        model_types.append('lightgbm')

    logger.info(f"Available models: {model_types}\n")

    # Train all datasets
    for dataset in ['cumulative', 'k2pandc', 'toi']:
        try:
            pipeline.train_dataset(
                dataset,
                model_types=model_types,
                tune=tune_hyperparameters
            )
        except Exception as e:
            logger.error(f"\n❌ Error training {dataset}: {str(e)}", exc_info=True)

    # Generate report
    pipeline.generate_summary_report()

    # Save models
    pipeline.save_models()

    logger.info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)


if __name__ == "__main__":
    # Set to True to enable hyperparameter tuning (takes longer)
    main(tune_hyperparameters=False)
