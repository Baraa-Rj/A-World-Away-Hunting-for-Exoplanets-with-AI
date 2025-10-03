"""
Simplified ML Training Pipeline for NASA Exoplanet Classification
Trains Random Forest and Gradient Boosting (sklearn) on all three datasets
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

import warnings
warnings.filterwarnings('ignore')


class ExoplanetMLPipeline:
    """Complete ML training pipeline for exoplanet classification"""

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
        """Prepare features and target for training"""
        print(f"\nPreparing data...")

        # Define target column based on dataset
        if dataset_name == 'cumulative':
            target_col = 'koi_disposition'
        elif dataset_name == 'k2pandc':
            target_col = 'disposition'
        else:  # toi
            target_col = 'tfopwg_disp'

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Drop non-numeric columns
        non_numeric = ['koi_disposition', 'disposition', 'tfopwg_disp',
                       'discoverymethod', 'koi_pdisposition']
        X = X.select_dtypes(include=[np.number])

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[dataset_name] = le

        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {len(X)}")
        print(f"  Classes: {len(le.classes_)}")
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

        # Store model
        self.models[f'{dataset_name}_rf'] = rf

        return {
            'model': 'Random Forest',
            'train_acc': train_acc,
            'val_acc': val_acc,
            'time': training_time
        }

    def train_gradient_boosting(self, X_train, y_train, X_val, y_val, dataset_name):
        """Train Gradient Boosting classifier"""
        print(f"\n{'='*80}")
        print(f"TRAINING GRADIENT BOOSTING - {dataset_name.upper()}")
        print(f"{'='*80}")

        start_time = time.time()

        # Train model
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )

        print("Training...")
        gb.fit(X_train, y_train)

        # Predictions
        y_train_pred = gb.predict(X_train)
        y_val_pred = gb.predict(X_val)

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        training_time = time.time() - start_time

        print(f"\n✓ Training complete!")
        print(f"  Training Time: {training_time:.2f} seconds")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Val Accuracy:   {val_acc*100:.2f}%")

        # Store model
        self.models[f'{dataset_name}_gb'] = gb

        return {
            'model': 'Gradient Boosting',
            'train_acc': train_acc,
            'val_acc': val_acc,
            'time': training_time
        }

    def evaluate_on_test(self, X_test, y_test, y_test_original, dataset_name):
        """Evaluate all models on test set"""
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION - {dataset_name.upper()}")
        print(f"{'='*80}")

        results = []
        le = self.label_encoders[dataset_name]

        for model_name, model in self.models.items():
            if dataset_name not in model_name:
                continue

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Multi-class metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            model_type = model_name.split('_')[-1].upper()

            print(f"\n{model_type}:")
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"  {cm}")

            results.append({
                'dataset': dataset_name,
                'model': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        return results

    def train_dataset(self, dataset_name):
        """Complete training pipeline for one dataset"""
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING PIPELINE: {dataset_name.upper()}")
        print(f"{'#'*80}\n")

        # Load data
        df = self.load_data(dataset_name)

        # Prepare data
        X, y_encoded, y_original = self.prepare_data(df, dataset_name)

        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_splits(X, y_encoded)

        # Train models
        results = []

        # Random Forest
        rf_results = self.train_random_forest(X_train, y_train, X_val, y_val, dataset_name)
        results.append(rf_results)

        # Gradient Boosting
        gb_results = self.train_gradient_boosting(X_train, y_train, X_val, y_val, dataset_name)
        results.append(gb_results)

        # Test evaluation
        test_results = self.evaluate_on_test(X_test, y_test, y_original, dataset_name)

        self.results[dataset_name] = {
            'training': results,
            'test': test_results
        }

        return results, test_results

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n\n{'='*80}")
        print("TRAINING SUMMARY REPORT")
        print(f"{'='*80}\n")

        for dataset_name in ['cumulative', 'k2pandc', 'toi']:
            if dataset_name not in self.results:
                continue

            print(f"\n{'─'*80}")
            print(f"{dataset_name.upper()} DATASET")
            print(f"{'─'*80}")

            # Training results
            print(f"\nValidation Results:")
            print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12} {'Time (s)':<12}")
            print("─" * 60)

            for result in self.results[dataset_name]['training']:
                print(f"{result['model']:<20} {result['train_acc']*100:>10.2f}% "
                      f"{result['val_acc']*100:>10.2f}% {result['time']:>10.2f}s")

            # Test results
            print(f"\nTest Results:")
            print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print("─" * 80)

            for result in self.results[dataset_name]['test']:
                print(f"{result['model']:<20} {result['accuracy']*100:>10.2f}% "
                      f"{result['precision']*100:>10.2f}% {result['recall']*100:>10.2f}% "
                      f"{result['f1']*100:>10.2f}%")

        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}\n")

        # Find best models
        best_models = {}
        for dataset_name in ['cumulative', 'k2pandc', 'toi']:
            if dataset_name not in self.results:
                continue

            best_acc = 0
            best_model = None

            for result in self.results[dataset_name]['test']:
                if result['accuracy'] > best_acc:
                    best_acc = result['accuracy']
                    best_model = result['model']

            best_models[dataset_name] = (best_model, best_acc)

        print("🏆 Best Models per Dataset:")
        for dataset, (model, acc) in best_models.items():
            print(f"  {dataset.upper():<15}: {model:<20} ({acc*100:.2f}% accuracy)")

        print(f"\n{'='*80}\n")

    def save_models(self):
        """Save trained models"""
        print("Saving models...")

        import os
        os.makedirs('models', exist_ok=True)

        for name, model in self.models.items():
            filepath = f'models/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Saved: {filepath}")

        # Save label encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"  ✓ Saved: models/label_encoders.pkl")

        print("\n✅ All models saved!")


def main():
    """Main training pipeline"""
    print("="*80)
    print("NASA EXOPLANET ML TRAINING PIPELINE")
    print("Using: Random Forest & Gradient Boosting (sklearn)")
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
