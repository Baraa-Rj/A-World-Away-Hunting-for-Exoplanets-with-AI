"""
XGBoost Enhanced Training Pipeline
Advanced ML with feature engineering, ensembling, and overfitting prevention
Target: 78-80% accuracy (from current 72%)
"""

import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime

# Core ML
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

import warnings
warnings.filterwarnings('ignore')


class EnhancedExoplanetML:
    """Enhanced ML pipeline with XGBoost and feature engineering"""

    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_names = {}
        self.results = {}

    def load_data(self, dataset_name):
        """Load cleaned dataset"""
        print(f"\n{'='*80}")
        print(f"LOADING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")

        df = pd.read_csv(f'data/processed/{dataset_name}_cleaned.csv')
        print(f"✓ Loaded: {df.shape}")
        return df

    def engineer_features(self, X, dataset_name):
        """Create advanced features using domain knowledge"""
        print(f"\n🔬 FEATURE ENGINEERING...")

        X_engineered = X.copy()
        initial_features = X.shape[1]

        # Dataset-specific feature engineering
        if dataset_name == 'cumulative':
            # 1. Interaction features
            if 'koi_period' in X.columns and 'koi_prad' in X.columns:
                X_engineered['period_radius_product'] = X['koi_period'] * X['koi_prad']
                X_engineered['period_radius_ratio'] = X['koi_period'] / (X['koi_prad'] + 1e-6)

            # 2. Temperature-based features
            if 'koi_teq' in X.columns:
                X_engineered['temp_squared'] = X['koi_teq'] ** 2
                X_engineered['is_hot_planet'] = (X['koi_teq'] > 1000).astype(int)
                X_engineered['is_temperate'] = ((X['koi_teq'] >= 200) & (X['koi_teq'] <= 400)).astype(int)

            # 3. Stellar flux features
            if 'koi_insol' in X.columns and 'koi_teq' in X.columns:
                X_engineered['flux_temp_ratio'] = X['koi_insol'] / (X['koi_teq'] + 1e-6)

            # 4. Transit signal strength
            if 'koi_depth' in X.columns and 'koi_duration' in X.columns:
                X_engineered['transit_signal'] = X['koi_depth'] * X['koi_duration']

            # 5. Planet density proxy
            if 'koi_prad' in X.columns and 'planet_volume' in X.columns:
                X_engineered['density_proxy'] = X['planet_volume'] / (X['koi_prad'] ** 3 + 1e-6)

            # 6. Orbital characteristics
            if 'koi_period' in X.columns:
                X_engineered['log_period'] = np.log1p(X['koi_period'])
                X_engineered['is_short_period'] = (X['koi_period'] < 10).astype(int)
                X_engineered['is_earth_like_period'] = ((X['koi_period'] >= 200) & (X['koi_period'] <= 500)).astype(int)

            # 7. Impact parameter features
            if 'koi_impact' in X.columns:
                X_engineered['impact_squared'] = X['koi_impact'] ** 2
                X_engineered['is_central_transit'] = (np.abs(X['koi_impact']) < 0.3).astype(int)

        elif dataset_name == 'k2pandc':
            # K2-specific features
            if 'pl_orbper' in X.columns and 'pl_rade' in X.columns:
                X_engineered['period_radius_product'] = X['pl_orbper'] * X['pl_rade']
                X_engineered['period_radius_ratio'] = X['pl_orbper'] / (X['pl_rade'] + 1e-6)

            if 'st_rad' in X.columns and 'st_mass' in X.columns:
                X_engineered['stellar_density'] = X['st_mass'] / (X['st_rad'] ** 3 + 1e-6)

            if 'st_teff' in X.columns:
                X_engineered['stellar_temp_squared'] = X['st_teff'] ** 2
                X_engineered['is_sun_like_star'] = ((X['st_teff'] >= 5000) & (X['st_teff'] <= 6500)).astype(int)

            if 'pl_orbper' in X.columns:
                X_engineered['log_period'] = np.log1p(X['pl_orbper'])
                X_engineered['is_hot_jupiter'] = ((X['pl_orbper'] < 10) & (X['pl_rade'] > 8)).astype(int)

        else:  # toi
            # TESS-specific features
            if 'pl_orbper' in X.columns and 'pl_rade' in X.columns:
                X_engineered['period_radius_product'] = X['pl_orbper'] * X['pl_rade']
                X_engineered['period_radius_ratio'] = X['pl_orbper'] / (X['pl_rade'] + 1e-6)

            if 'pl_trandurh' in X.columns and 'pl_trandep' in X.columns:
                X_engineered['transit_signal'] = X['pl_trandurh'] * X['pl_trandep']

            if 'st_tmag' in X.columns:
                X_engineered['is_bright_star'] = (X['st_tmag'] < 10).astype(int)

            if 'pl_orbper' in X.columns:
                X_engineered['log_period'] = np.log1p(X['pl_orbper'])

        # Replace any inf or -inf with nan, then fill with 0
        X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)
        X_engineered = X_engineered.fillna(0)

        new_features = X_engineered.shape[1] - initial_features
        print(f"  ✓ Created {new_features} new features ({initial_features} → {X_engineered.shape[1]})")

        return X_engineered

    def prepare_data(self, df, dataset_name):
        """Prepare data with feature engineering"""
        print(f"\nPreparing data...")

        # Define target and leakage columns
        if dataset_name == 'cumulative':
            target_col = 'koi_disposition'
            leakage_cols = ['koi_disposition', 'koi_disposition_encoded',
                           'koi_pdisposition', 'koi_pdisposition_encoded']
        elif dataset_name == 'k2pandc':
            target_col = 'disposition'
            leakage_cols = ['disposition', 'disposition_encoded',
                           'discoverymethod', 'discoverymethod_encoded']
        else:
            target_col = 'tfopwg_disp'
            leakage_cols = ['tfopwg_disp', 'tfopwg_disp_encoded']

        # Get target
        y = df[target_col].copy()

        # Remove leakage columns
        X = df.drop(columns=[col for col in leakage_cols if col in df.columns])
        X = X.select_dtypes(include=[np.number])

        print(f"  Original features: {X.shape[1]}")

        # Feature engineering
        X = self.engineer_features(X, dataset_name)

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"  Imputing {X.isnull().sum().sum()} missing values...")
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[dataset_name] = le

        print(f"\n  ✓ Final features: {X.shape[1]}")
        print(f"  ✓ Samples: {len(X)}")
        print(f"  ✓ Classes: {len(le.classes_)} - {list(le.classes_)}")

        self.feature_names[dataset_name] = X.columns.tolist()

        return X, y_encoded

    def train_xgboost(self, X_train, y_train, X_val, y_val, dataset_name):
        """Train XGBoost with anti-overfitting settings"""
        print(f"\n{'='*80}")
        print(f"TRAINING XGBOOST - {dataset_name.upper()}")
        print(f"{'='*80}")

        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available, skipping...")
            return None, 0

        # Determine number of classes
        n_classes = len(np.unique(y_train))

        # XGBoost parameters optimized for small datasets
        params = {
            'n_estimators': 500,
            'max_depth': 8,              # Reduced from 20 to prevent overfitting
            'learning_rate': 0.05,        # Lower learning rate for better generalization
            'min_child_weight': 5,        # Regularization
            'gamma': 0.1,                 # Regularization
            'subsample': 0.8,             # Use 80% of data per tree
            'colsample_bytree': 0.8,      # Use 80% of features per tree
            'reg_alpha': 0.1,             # L1 regularization
            'reg_lambda': 1.0,            # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50   # Stop if no improvement for 50 rounds
        }

        if n_classes == 2:
            model = xgb.XGBClassifier(objective='binary:logistic', **params)
        else:
            model = xgb.XGBClassifier(objective='multi:softmax', num_class=n_classes, **params)

        print("Training with early stopping...")
        start = time.time()

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        train_time = time.time() - start

        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"\n✓ Training complete!")
        print(f"  Training Time: {train_time:.2f}s")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        print(f"  Val Accuracy:   {val_acc*100:.2f}%")
        print(f"  Overfitting Gap: {(train_acc - val_acc)*100:.2f}%")

        if (train_acc - val_acc) < 0.10:
            print(f"  ✅ Excellent generalization!")
        elif (train_acc - val_acc) < 0.15:
            print(f"  ✓ Good generalization")
        else:
            print(f"  ⚠️  Still some overfitting")

        return model, val_acc

    def train_ensemble(self, X_train, y_train, X_val, y_val, dataset_name, xgb_model):
        """Create ensemble of XGBoost + RF + GB"""
        print(f"\n{'='*80}")
        print(f"TRAINING ENSEMBLE - {dataset_name.upper()}")
        print(f"{'='*80}")

        # Random Forest with anti-overfitting
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,              # Reduced from 20
            min_samples_split=10,       # Increased from 2
            min_samples_leaf=4,         # Increased from 1
            max_features='log2',        # Feature diversity
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            min_samples_split=10,
            random_state=42
        )

        print("Training ensemble components...")

        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        # Create voting ensemble
        if XGBOOST_AVAILABLE and xgb_model is not None:
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('rf', rf),
                    ('gb', gb)
                ],
                voting='soft',
                weights=[2, 1, 1]  # XGBoost gets double weight
            )
        else:
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb)
                ],
                voting='soft'
            )

        ensemble.fit(X_train, y_train)

        # Evaluate
        y_val_pred = ensemble.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"\n✓ Ensemble complete!")
        print(f"  Val Accuracy: {val_acc*100:.2f}%")

        return ensemble, val_acc

    def cross_validate(self, X, y, model, dataset_name):
        """Perform 5-fold cross-validation"""
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION - {dataset_name.upper()}")
        print(f"{'='*80}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        print(f"\nCross-Validation Scores (5 folds):")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score*100:.2f}%")

        print(f"\n📊 Summary:")
        print(f"  Mean: {scores.mean()*100:.2f}%")
        print(f"  Std:  {scores.std()*100:.2f}%")
        print(f"  Min:  {scores.min()*100:.2f}%")
        print(f"  Max:  {scores.max()*100:.2f}%")

        return scores.mean(), scores.std()

    def train_dataset(self, dataset_name):
        """Complete training pipeline"""
        print(f"\n\n{'#'*80}")
        print(f"# ENHANCED TRAINING: {dataset_name.upper()}")
        print(f"{'#'*80}")

        # Load and prepare
        df = self.load_data(dataset_name)
        X, y = self.prepare_data(df, dataset_name)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nData splits:")
        print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")

        # Train XGBoost
        xgb_model, xgb_acc = self.train_xgboost(X_train, y_train, X_val, y_val, dataset_name)

        # Train ensemble
        ensemble_model, ensemble_acc = self.train_ensemble(X_train, y_train, X_val, y_val, dataset_name, xgb_model)

        # Choose best model
        if XGBOOST_AVAILABLE and xgb_acc > ensemble_acc:
            best_model = xgb_model
            best_acc = xgb_acc
            model_type = 'XGBoost'
        else:
            best_model = ensemble_model
            best_acc = ensemble_acc
            model_type = 'Ensemble'

        print(f"\n{'='*80}")
        print(f"BEST MODEL: {model_type}")
        print(f"{'='*80}")

        # Cross-validation on best model
        cv_mean, cv_std = self.cross_validate(X, y, best_model, dataset_name)

        # Save results
        self.models[dataset_name] = best_model
        self.results[dataset_name] = {
            'model_type': model_type,
            'val_accuracy': best_acc,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }

        return best_model, cv_mean

    def generate_report(self):
        """Generate comparison report"""
        print(f"\n\n{'='*80}")
        print("FINAL RESULTS - ENHANCED MODELS")
        print(f"{'='*80}\n")

        print(f"{'Dataset':<15} {'Model':<12} {'CV Accuracy':<15} {'Improvement'}")
        print("="*80)

        baseline = {'cumulative': 0.72, 'k2pandc': 0.76, 'toi': 0.69}

        for dataset, results in self.results.items():
            cv_acc = results['cv_mean']
            improvement = (cv_acc - baseline[dataset]) * 100

            print(f"{dataset.upper():<15} {results['model_type']:<12} "
                  f"{cv_acc*100:.2f}% ± {results['cv_std']*100:.2f}%  "
                  f"{improvement:+.2f}%")

        avg_cv = np.mean([r['cv_mean'] for r in self.results.values()])
        avg_baseline = np.mean(list(baseline.values()))
        total_improvement = (avg_cv - avg_baseline) * 100

        print("-"*80)
        print(f"{'AVERAGE':<15} {'Mixed':<12} {avg_cv*100:.2f}%            {total_improvement:+.2f}%")

        print(f"\n{'='*80}")
        if avg_cv >= 0.78:
            print("🎉 SUCCESS! Achieved 78%+ target accuracy!")
        elif avg_cv >= 0.75:
            print("✅ GOOD! Significant improvement achieved!")
        else:
            print("📊 Improvement made. Consider Phase 2 (more data) for 85%")
        print(f"{'='*80}\n")

    def save_models(self):
        """Save enhanced models"""
        print("Saving enhanced models...")

        import os
        os.makedirs('models_enhanced', exist_ok=True)

        for dataset, model in self.models.items():
            filepath = f'models_enhanced/{dataset}_enhanced.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  ✓ Saved: {filepath}")

        with open('models_enhanced/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        print(f"\n✅ All enhanced models saved!")


def main():
    """Main pipeline"""
    print("="*80)
    print("ENHANCED EXOPLANET ML PIPELINE")
    print("XGBoost + Feature Engineering + Ensemble")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not XGBOOST_AVAILABLE:
        print("\n⚠️  XGBoost not installed!")
        print("Install with: pip install xgboost")
        print("Continuing with Random Forest + Gradient Boosting ensemble...\n")

    pipeline = EnhancedExoplanetML()

    # Train all datasets
    for dataset in ['cumulative', 'k2pandc', 'toi']:
        try:
            pipeline.train_dataset(dataset)
        except Exception as e:
            print(f"\n❌ Error training {dataset}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Generate report
    pipeline.generate_report()

    # Save models
    pipeline.save_models()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
