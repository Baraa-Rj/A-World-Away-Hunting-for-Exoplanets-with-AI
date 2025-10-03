"""
FIXED ML Training Pipeline - Removing Data Leakage
Properly excludes all target-related columns
"""

import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')


class ExoplanetMLPipeline:
    """Complete ML training pipeline - FIXED for data leakage"""

    def __init__(self):
        self.results = {}
        self.models = {}
        self.label_encoders = {}

    def load_data(self, dataset_name):
        """Load cleaned dataset"""
        print(f"\n{'='*80}")
        print(f"LOADING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")

        file_path = f'data/processed/{dataset_name}_cleaned.csv'
        df = pd.read_csv(file_path)

        print(f"✓ Loaded: {df.shape}")
        return df

    def prepare_data(self, df, dataset_name):
        """Prepare features and target - FIXED to remove data leakage"""
        print(f"\nPreparing data...")

        # Define target column based on dataset
        if dataset_name == 'cumulative':
            target_col = 'koi_disposition'
            # REMOVE ALL TARGET-RELATED COLUMNS
            leakage_cols = ['koi_disposition', 'koi_disposition_encoded',
                           'koi_pdisposition', 'koi_pdisposition_encoded']
        elif dataset_name == 'k2pandc':
            target_col = 'disposition'
            leakage_cols = ['disposition', 'disposition_encoded',
                           'discoverymethod', 'discoverymethod_encoded']
        else:  # toi
            target_col = 'tfopwg_disp'
            leakage_cols = ['tfopwg_disp', 'tfopwg_disp_encoded']

        # Get target before dropping
        y = df[target_col].copy()

        # Drop ALL target and leakage columns
        X = df.drop(columns=[col for col in leakage_cols if col in df.columns])

        # Keep only numeric features
        X = X.select_dtypes(include=[np.number])

        print(f"\n🔍 DATA LEAKAGE CHECK:")
        print(f"  Removed columns: {[col for col in leakage_cols if col in df.columns]}")
        print(f"  Remaining features: {X.shape[1]}")

        # Verify no leakage
        for col in X.columns:
            if 'disposition' in col.lower() or 'encoded' in col.lower():
                print(f"  ⚠️  WARNING: Potential leakage in column: {col}")

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"\n  Imputing {X.isnull().sum().sum()} missing values...")
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[dataset_name] = le

        print(f"\n  ✓ Features: {X.shape[1]}")
        print(f"  ✓ Samples: {len(X)}")
        print(f"  ✓ Classes: {len(le.classes_)}")
        print(f"  Class distribution:")
        for cls, count in zip(*np.unique(y_encoded, return_counts=True)):
            print(f"    {le.classes_[cls]}: {count} ({count/len(y)*100:.1f}%)")

        return X, y_encoded, y

    def create_splits(self, X, y, test_size=0.15, val_size=0.15):
        """Create train/validation/test splits"""
        print(f"\nCreating train/validation/test splits...")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_random_forest(self, X_train, y_train, X_val, y_val, dataset_name):
        """Train Random Forest classifier"""
        print(f"\n{'='*80}")
        print(f"TRAINING RANDOM FOREST - {dataset_name.upper()}")
        print(f"{'='*80}")

        start_time = time.time()

        # Train model
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        print("Training...")
        rf.fit(X_train, y_train)

        # Predictions
        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        training_time = time.time() - start_time

        print(f"\n✓ Training complete!")
        print(f"  Training Time: {training_time:.2f} seconds")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Val Accuracy:   {val_acc*100:.2f}%")

        # Check for overfitting
        overfit_gap = (train_acc - val_acc) * 100
        if overfit_gap > 5:
            print(f"  ⚠️  Overfitting detected! Gap: {overfit_gap:.2f}%")
        elif val_acc > 0.95:
            print(f"  ✓ Excellent generalization!")
        else:
            print(f"  ✓ Good generalization (gap: {overfit_gap:.2f}%)")

        # Store model
        self.models[f'{dataset_name}_rf'] = rf

        return {
            'model': 'Random Forest',
            'train_acc': train_acc,
            'val_acc': val_acc,
            'time': training_time
        }

    def evaluate_on_test(self, X_test, y_test, dataset_name):
        """Evaluate model on test set"""
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION - {dataset_name.upper()}")
        print(f"{'='*80}")

        model = self.models[f'{dataset_name}_rf']
        le = self.label_encoders[dataset_name]

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\nTest Results:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        # Feature importance
        print(f"\n🔍 TOP 10 IMPORTANT FEATURES:")
        feature_names = self.models[f'{dataset_name}_rf'].feature_names_in_
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        for i, idx in enumerate(indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")

        return {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train_dataset(self, dataset_name):
        """Complete training pipeline for one dataset"""
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING PIPELINE: {dataset_name.upper()}")
        print(f"{'#'*80}\n")

        # Load data
        df = self.load_data(dataset_name)

        # Prepare data (FIXED - no leakage)
        X, y_encoded, y_original = self.prepare_data(df, dataset_name)

        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X, y_encoded)

        # Train model
        train_results = self.train_random_forest(X_train, y_train, X_val, y_val, dataset_name)

        # Test evaluation
        test_results = self.evaluate_on_test(X_test, y_test, dataset_name)

        self.results[dataset_name] = {
            'training': train_results,
            'test': test_results
        }

        return train_results, test_results

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n\n{'='*80}")
        print("FINAL TRAINING SUMMARY (NO DATA LEAKAGE)")
        print(f"{'='*80}\n")

        print(f"{'Dataset':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'F1-Score':<12}")
        print("=" * 70)

        for dataset_name in ['cumulative', 'k2pandc', 'toi']:
            if dataset_name not in self.results:
                continue

            train_res = self.results[dataset_name]['training']
            test_res = self.results[dataset_name]['test']

            print(f"{dataset_name.upper():<15} {train_res['train_acc']*100:>10.2f}% "
                  f"{train_res['val_acc']*100:>10.2f}% {test_res['accuracy']*100:>10.2f}% "
                  f"{test_res['f1']*100:>10.2f}%")

        print(f"\n{'='*80}\n")

    def save_models(self):
        """Save trained models"""
        print("Saving FIXED models...")

        import os
        os.makedirs('models_fixed', exist_ok=True)

        for name, model in self.models.items():
            filepath = f'models_fixed/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Saved: {filepath}")

        # Save label encoders
        with open('models_fixed/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"  ✓ Saved: models_fixed/label_encoders.pkl")

        print("\n✅ All FIXED models saved to models_fixed/!")


def main():
    """Main training pipeline"""
    print("="*80)
    print("NASA EXOPLANET ML TRAINING PIPELINE - FIXED")
    print("Removed data leakage - using only real features")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pipeline = ExoplanetMLPipeline()

    # Train all datasets
    for dataset in ['cumulative', 'k2pandc', 'toi']:
        try:
            pipeline.train_dataset(dataset)
        except Exception as e:
            print(f"\n❌ Error training {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Generate report
    pipeline.generate_summary_report()

    # Save models
    pipeline.save_models()

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
