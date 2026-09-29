"""
Refactored model testing and evaluation with type hints and logging.
"""
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelTester:
    """Test and evaluate trained models with comprehensive metrics."""

    def __init__(self):
        """Initialize the model tester."""
        self.models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.missing: List[str] = []
        self.test_indices: Dict[str, List] = {}

    def load_models(
        self,
        model_types: Optional[List[str]] = None
    ) -> None:
        """
        Load trained models from disk.

        Args:
            model_types: List of model types to load (default: config.MODEL_TYPES)
        """
        logger.info("="*80)
        logger.info("LOADING TRAINED MODELS")
        logger.info("="*80)

        if model_types is None:
            model_types = config.MODEL_TYPES

        for dataset in ['cumulative', 'k2pandc', 'toi']:
            for model_type in model_types:
                try:
                    model_path = config.get_model_file_path(dataset, model_type)

                    if not Path(model_path).exists():
                        logger.error(f"Model not found: {model_path}")
                        self.missing.append(model_path)
                        continue

                    with open(model_path, 'rb') as f:
                        self.models[f'{dataset}_{model_type}'] = pickle.load(f)
                    logger.info(f"✓ Loaded: {model_path}")

                except Exception as e:
                    logger.error(f"✗ Error loading {dataset}_{model_type}: {e}")

        # Load label encoders
        try:
            encoder_path = config.MODELS_FIXED_DIR / 'label_encoders.pkl'
            with open(encoder_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"✓ Loaded: label_encoders.pkl")
        except Exception as e:
            logger.error(f"✗ Error loading label encoders: {e}")
            self.missing.append(str(encoder_path))

        # Load the held-out test split saved by training
        split_path = config.get_test_split_path()
        try:
            with open(split_path, 'rb') as f:
                self.test_indices = pickle.load(f)
            logger.info(f"✓ Loaded: {split_path}")
        except Exception as e:
            logger.error(f"✗ Error loading test split: {e}")
            self.missing.append(split_path)

    def load_and_prepare_data(
        self,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, LabelEncoder]:
        """
        Load and prepare test data.

        Args:
            dataset_name: Name of dataset

        Returns:
            Tuple of (features, encoded_target, original_target, label_encoder)
        """
        logger.info("="*80)
        logger.info(f"LOADING {dataset_name.upper()} TEST DATA")
        logger.info("="*80)

        try:
            # Load cleaned data
            file_path = config.get_cleaned_file_path(dataset_name)
            df = pd.read_csv(file_path)

            # Same feature preparation as training
            X, y = prepare_features(df, dataset_name)

            # Encode target
            le = self.label_encoders[dataset_name]
            y_encoded = le.transform(y)

            logger.info(f"  Total samples: {len(X)}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  Classes: {len(le.classes_)}")

            return X, y_encoded, y, le

        except Exception as e:
            logger.error(f"Failed to load data for {dataset_name}: {str(e)}")
            raise

    def evaluate_model(
        self,
        dataset_name: str,
        model_type: str = 'rf'
    ) -> Dict[str, float]:
        """
        Evaluate model on the held-out test split saved by training.

        Args:
            dataset_name: Name of dataset
            model_type: Type of model to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("="*80)
        logger.info(f"EVALUATING {dataset_name.upper()} - {model_type.upper()}")
        logger.info("="*80)

        try:
            # Load data
            X, y_encoded, y_original, le = self.load_and_prepare_data(dataset_name)

            # Get model
            model_key = f'{dataset_name}_{model_type}'
            if model_key not in self.models:
                logger.error(f"Model {model_key} not loaded!")
                return {}

            model = self.models[model_key]

            # Restrict to the held-out test rows (the rest were used for training)
            test_idx = self.test_indices[dataset_name]
            X = X.loc[test_idx]
            y_original = y_original.loc[test_idx]
            y_encoded = le.transform(y_original)
            logger.info(f"  Held-out test samples: {len(X)}")

            # Make predictions
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)

            # Calculate metrics
            accuracy = accuracy_score(y_encoded, y_pred)
            precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_encoded, y_pred, average='weighted', zero_division=0)

            logger.info(f"\n📊 OVERALL METRICS:")
            logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
            logger.info(f"  Precision: {precision*100:.2f}%")
            logger.info(f"  Recall:    {recall*100:.2f}%")
            logger.info(f"  F1-Score:  {f1*100:.2f}%")

            # Confusion matrix
            cm = confusion_matrix(y_encoded, y_pred)

            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                logger.info(f"\n🔍 TOP 10 IMPORTANT FEATURES:")
                feature_names = X.columns
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]

                for i, idx in enumerate(indices, 1):
                    logger.info(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")

                # Generate visualizations
                self.plot_confusion_matrix(cm, le.classes_, dataset_name, model_type)
                self.plot_feature_importance(feature_names, importances, dataset_name, model_type)

            # Store results
            self.results[model_key] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            return self.results[model_key]

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {}

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: np.ndarray,
        dataset_name: str,
        model_type: str
    ) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix
            classes: Class labels
            dataset_name: Name of dataset
            model_type: Type of model
        """
        try:
            plt.figure(figsize=(10, 8))

            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage'}
            )

            plt.title(
                f'{dataset_name.upper()} - {model_type.upper()} - Confusion Matrix\n(Normalized by True Label)',
                fontsize=14, fontweight='bold'
            )
            plt.ylabel('True Label', fontsize=12, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
            plt.tight_layout()

            # Save
            output_path = config.PROCESSED_DATA_DIR / f'{dataset_name}_{model_type}_confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"\n✓ Saved confusion matrix: {output_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}")

    def plot_feature_importance(
        self,
        feature_names: pd.Index,
        importances: np.ndarray,
        dataset_name: str,
        model_type: str
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_names: Feature names
            importances: Feature importance values
            dataset_name: Name of dataset
            model_type: Type of model
        """
        try:
            plt.figure(figsize=(12, 8))

            # Get top 15 features
            indices = np.argsort(importances)[::-1][:15]
            top_features = [feature_names[i] for i in indices]
            top_importances = [importances[i] for i in indices]

            # Create bar plot
            plt.barh(range(len(top_features)), top_importances, color='steelblue')
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
            plt.title(
                f'{dataset_name.upper()} - {model_type.upper()} - Top 15 Feature Importances',
                fontsize=14, fontweight='bold'
            )
            plt.gca().invert_yaxis()

            # Add values on bars
            for i, v in enumerate(top_importances):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center')

            plt.tight_layout()

            # Save
            output_path = config.PROCESSED_DATA_DIR / f'{dataset_name}_{model_type}_feature_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved feature importance: {output_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Failed to plot feature importance: {str(e)}")

    def generate_summary_report(self) -> None:
        """Generate final summary report."""
        logger.info("\n" + "="*80)
        logger.info("FINAL EVALUATION SUMMARY")
        logger.info("="*80 + "\n")

        if not self.results:
            logger.warning("No results to report!")
            return

        logger.info(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        logger.info("=" * 80)

        for model_key, results in self.results.items():
            logger.info(
                f"{model_key:<25} {results['accuracy']*100:>10.2f}% "
                f"{results['precision']*100:>10.2f}% {results['recall']*100:>10.2f}% "
                f"{results['f1']*100:>10.2f}%"
            )

        # Average metrics
        avg_acc = np.mean([r['accuracy'] for r in self.results.values()])
        avg_prec = np.mean([r['precision'] for r in self.results.values()])
        avg_rec = np.mean([r['recall'] for r in self.results.values()])
        avg_f1 = np.mean([r['f1'] for r in self.results.values()])

        logger.info("-" * 80)
        logger.info(
            f"{'AVERAGE':<25} {avg_acc*100:>10.2f}% {avg_prec*100:>10.2f}% "
            f"{avg_rec*100:>10.2f}% {avg_f1*100:>10.2f}%"
        )

        logger.info("\n" + "="*80)


def main(model_types: Optional[List[str]] = None) -> int:
    """
    Main testing pipeline.

    Args:
        model_types: List of model types to test (default: config.MODEL_TYPES)

    Returns:
        Process exit code: 0 if every model was found and evaluated, 1 otherwise
    """
    logger.info("="*80)
    logger.info("NASA EXOPLANET MODEL TESTING & EVALUATION")
    logger.info("="*80)

    if model_types is None:
        model_types = config.MODEL_TYPES

    tester = ModelTester()

    # Load models
    tester.load_models(model_types=model_types)

    # Evaluate each dataset and model
    for dataset in ['cumulative', 'k2pandc', 'toi']:
        for model_type in model_types:
            model_key = f'{dataset}_{model_type}'
            if model_key in tester.models:
                tester.evaluate_model(dataset, model_type)

    # Generate summary
    tester.generate_summary_report()

    expected = len(model_types) * len(['cumulative', 'k2pandc', 'toi'])
    if tester.missing:
        logger.error(
            f"\n❌ Missing {len(tester.missing)} trained model file(s). "
            f"Run scripts/train_models_refactored.py first."
        )
        return 1
    if len(tester.results) < expected:
        logger.error(f"\n❌ Only {len(tester.results)} of {expected} models were evaluated.")
        return 1

    logger.info("\n✅ TESTING COMPLETE!")
    logger.info("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
