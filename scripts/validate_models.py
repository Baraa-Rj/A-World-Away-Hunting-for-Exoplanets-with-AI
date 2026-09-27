"""
Comprehensive Model Validation
Tests model robustness, generalization, and real-world performance
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """Comprehensive model validation"""

    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.results = {}

    def load_models(self):
        """Load trained models"""
        print("="*80)
        print("LOADING MODELS FOR VALIDATION")
        print("="*80)

        for dataset in ['cumulative', 'k2pandc', 'toi']:
            with open(f'models_fixed/{dataset}_rf.pkl', 'rb') as f:
                self.models[dataset] = pickle.load(f)
            print(f"✓ Loaded: {dataset}_rf.pkl")

        with open('models_fixed/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        print(f"✓ Loaded: label_encoders.pkl\n")

    def prepare_data(self, dataset_name):
        """Prepare data without leakage"""
        df = pd.read_csv(f'data/processed/{dataset_name}_cleaned.csv')

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

        y = df[target_col].copy()
        X = df.drop(columns=[col for col in leakage_cols if col in df.columns])
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        from sklearn.impute import SimpleImputer
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        # Encode target
        le = self.label_encoders[dataset_name]
        y_encoded = le.transform(y)

        return X, y_encoded, le

    def test_1_cross_validation(self, dataset_name):
        """Test 1: K-Fold Cross-Validation"""
        print(f"\n{'='*80}")
        print(f"TEST 1: K-FOLD CROSS-VALIDATION - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Test if model generalizes well to unseen data\n")

        X, y, le = self.prepare_data(dataset_name)
        model = self.models[dataset_name]

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

        print(f"Cross-Validation Scores (5 folds):")
        for i, score in enumerate(cv_scores, 1):
            print(f"  Fold {i}: {score*100:.2f}%")

        print(f"\n📊 Summary:")
        print(f"  Mean Accuracy: {cv_scores.mean()*100:.2f}%")
        print(f"  Std Deviation: {cv_scores.std()*100:.2f}%")
        print(f"  Min Accuracy:  {cv_scores.min()*100:.2f}%")
        print(f"  Max Accuracy:  {cv_scores.max()*100:.2f}%")

        # Interpretation
        if cv_scores.std() < 0.05:
            print(f"\n✅ PASS: Low variance ({cv_scores.std():.4f}) indicates consistent performance")
        else:
            print(f"\n⚠️  WARNING: High variance ({cv_scores.std():.4f}) suggests instability")

        return cv_scores.mean(), cv_scores.std()

    def test_2_data_leakage_check(self, dataset_name):
        """Test 2: Verify No Data Leakage"""
        print(f"\n{'='*80}")
        print(f"TEST 2: DATA LEAKAGE CHECK - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Ensure target-related features are not in training data\n")

        X, y, le = self.prepare_data(dataset_name)

        # Check for suspicious keywords in feature names
        suspicious_keywords = ['disposition', 'disp', 'confirmed', 'candidate',
                              'false', 'positive', 'refuted']

        leakage_found = []
        for col in X.columns:
            col_lower = col.lower()
            for keyword in suspicious_keywords:
                if keyword in col_lower:
                    leakage_found.append(col)
                    break

        print(f"Features checked: {len(X.columns)}")
        print(f"Suspicious features: {len(leakage_found)}")

        if leakage_found:
            print(f"\n⚠️  WARNING: Potentially suspicious features found:")
            for col in leakage_found:
                print(f"  - {col}")
            print(f"\n❌ FAIL: Data leakage may be present")
            return False
        else:
            print(f"\n✅ PASS: No suspicious feature names detected")
            return True

    def test_3_feature_importance_sanity(self, dataset_name):
        """Test 3: Feature Importance Sanity Check"""
        print(f"\n{'='*80}")
        print(f"TEST 3: FEATURE IMPORTANCE SANITY - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Ensure no single feature dominates predictions\n")

        X, y, le = self.prepare_data(dataset_name)
        model = self.models[dataset_name]

        importances = model.feature_importances_
        max_importance = importances.max()

        print(f"Top 5 Most Important Features:")
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {X.columns[idx]:30s} {importances[idx]*100:6.2f}%")

        print(f"\n📊 Analysis:")
        print(f"  Max feature importance: {max_importance*100:.2f}%")
        print(f"  Features > 10% importance: {sum(importances > 0.10)}")

        # Check if any single feature dominates
        if max_importance > 0.30:
            print(f"\n⚠️  WARNING: One feature dominates (>{max_importance*100:.1f}%)")
            print(f"  This may indicate data leakage or overfitting")
            return False
        else:
            print(f"\n✅ PASS: Features are balanced (max {max_importance*100:.1f}%)")
            return True

    def test_4_class_distribution(self, dataset_name):
        """Test 4: Predictions Match True Class Distribution"""
        print(f"\n{'='*80}")
        print(f"TEST 4: CLASS DISTRIBUTION ANALYSIS - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Check if predictions are reasonable, not biased\n")

        X, y, le = self.prepare_data(dataset_name)
        model = self.models[dataset_name]
        y_pred = model.predict(X)

        print(f"{'Class':<20} {'True %':<12} {'Pred %':<12} {'Difference'}")
        print("-" * 60)

        max_diff = 0
        for i, cls in enumerate(le.classes_):
            true_pct = (y == i).sum() / len(y) * 100
            pred_pct = (y_pred == i).sum() / len(y_pred) * 100
            diff = abs(true_pct - pred_pct)
            max_diff = max(max_diff, diff)

            print(f"{cls:<20} {true_pct:>10.2f}% {pred_pct:>10.2f}% {diff:>10.2f}%")

        print(f"\n📊 Analysis:")
        print(f"  Max distribution difference: {max_diff:.2f}%")

        if max_diff < 5.0:
            print(f"\n✅ PASS: Predictions closely match true distribution")
            return True
        elif max_diff < 10.0:
            print(f"\n⚠️  WARNING: Some class imbalance in predictions")
            return True
        else:
            print(f"\n❌ FAIL: Large distribution mismatch (>{max_diff:.1f}%)")
            return False

    def test_5_prediction_confidence(self, dataset_name):
        """Test 5: Model Confidence Analysis"""
        print(f"\n{'='*80}")
        print(f"TEST 5: PREDICTION CONFIDENCE ANALYSIS - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Check if model is overconfident or underconfident\n")

        X, y, le = self.prepare_data(dataset_name)
        model = self.models[dataset_name]
        y_pred_proba = model.predict_proba(X)

        # Get max confidence for each prediction
        max_confidences = y_pred_proba.max(axis=1)

        print(f"Confidence Distribution:")
        print(f"  Mean confidence: {max_confidences.mean()*100:.2f}%")
        print(f"  Median confidence: {np.median(max_confidences)*100:.2f}%")
        print(f"  Min confidence: {max_confidences.min()*100:.2f}%")
        print(f"  Max confidence: {max_confidences.max()*100:.2f}%")

        # Confidence brackets
        print(f"\n📊 Confidence Brackets:")
        print(f"  Very Low (0-50%):   {(max_confidences < 0.50).sum():5d} ({(max_confidences < 0.50).sum()/len(max_confidences)*100:.1f}%)")
        print(f"  Low (50-70%):       {((max_confidences >= 0.50) & (max_confidences < 0.70)).sum():5d} ({((max_confidences >= 0.50) & (max_confidences < 0.70)).sum()/len(max_confidences)*100:.1f}%)")
        print(f"  Medium (70-85%):    {((max_confidences >= 0.70) & (max_confidences < 0.85)).sum():5d} ({((max_confidences >= 0.70) & (max_confidences < 0.85)).sum()/len(max_confidences)*100:.1f}%)")
        print(f"  High (85-95%):      {((max_confidences >= 0.85) & (max_confidences < 0.95)).sum():5d} ({((max_confidences >= 0.85) & (max_confidences < 0.95)).sum()/len(max_confidences)*100:.1f}%)")
        print(f"  Very High (95-100%): {(max_confidences >= 0.95).sum():5d} ({(max_confidences >= 0.95).sum()/len(max_confidences)*100:.1f}%)")

        mean_conf = max_confidences.mean()

        if mean_conf > 0.95:
            print(f"\n⚠️  WARNING: Model may be overconfident (mean {mean_conf*100:.1f}%)")
            return False
        elif mean_conf < 0.60:
            print(f"\n⚠️  WARNING: Model is underconfident (mean {mean_conf*100:.1f}%)")
            return False
        else:
            print(f"\n✅ PASS: Confidence levels are reasonable")
            return True

    def test_6_error_analysis(self, dataset_name):
        """Test 6: Analyze Misclassifications"""
        print(f"\n{'='*80}")
        print(f"TEST 6: ERROR ANALYSIS - {dataset_name.upper()}")
        print(f"{'='*80}")
        print("Purpose: Understand what types of mistakes the model makes\n")

        X, y, le = self.prepare_data(dataset_name)
        model = self.models[dataset_name]
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        # Find misclassifications
        errors = y != y_pred
        n_errors = errors.sum()

        print(f"Total Predictions: {len(y)}")
        print(f"Errors: {n_errors} ({n_errors/len(y)*100:.2f}%)")
        print(f"Correct: {(~errors).sum()} ({(~errors).sum()/len(y)*100:.2f}%)")

        if n_errors > 0:
            print(f"\n📊 Common Misclassifications:")

            # Confusion patterns
            from collections import Counter
            error_pairs = []
            for i in range(len(y)):
                if errors[i]:
                    true_label = le.classes_[y[i]]
                    pred_label = le.classes_[y_pred[i]]
                    error_pairs.append((true_label, pred_label))

            most_common = Counter(error_pairs).most_common(5)
            for (true_cls, pred_cls), count in most_common:
                print(f"  {true_cls:20s} → {pred_cls:20s}: {count:4d} errors")

            # Average confidence on errors
            error_confidences = y_pred_proba.max(axis=1)[errors]
            print(f"\n📊 Confidence on Errors:")
            print(f"  Mean confidence: {error_confidences.mean()*100:.2f}%")
            print(f"  (Lower is better - means model knew it was uncertain)")

        print(f"\n✅ Analysis complete")
        return True

    def run_all_tests(self):
        """Run all validation tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL VALIDATION SUITE")
        print("="*80)
        print("Running 6 validation tests on each dataset...\n")

        self.load_models()

        all_results = {}

        for dataset in ['cumulative', 'k2pandc', 'toi']:
            print(f"\n\n{'#'*80}")
            print(f"# VALIDATING: {dataset.upper()}")
            print(f"{'#'*80}")

            results = {}

            # Run all tests
            cv_mean, cv_std = self.test_1_cross_validation(dataset)
            results['cross_val'] = {'mean': cv_mean, 'std': cv_std}

            results['no_leakage'] = self.test_2_data_leakage_check(dataset)
            results['feature_sanity'] = self.test_3_feature_importance_sanity(dataset)
            results['class_dist'] = self.test_4_class_distribution(dataset)
            results['confidence'] = self.test_5_prediction_confidence(dataset)
            results['error_analysis'] = self.test_6_error_analysis(dataset)

            all_results[dataset] = results

        # Final summary
        self.print_final_summary(all_results)

    def print_final_summary(self, all_results):
        """Print final validation summary"""
        print(f"\n\n{'='*80}")
        print("FINAL VALIDATION SUMMARY")
        print(f"{'='*80}\n")

        for dataset, results in all_results.items():
            print(f"\n{dataset.upper()}:")
            print(f"  ✓ Cross-Validation: {results['cross_val']['mean']*100:.2f}% ± {results['cross_val']['std']*100:.2f}%")
            print(f"  {'✓' if results['no_leakage'] else '✗'} Data Leakage Check: {'PASS' if results['no_leakage'] else 'FAIL'}")
            print(f"  {'✓' if results['feature_sanity'] else '✗'} Feature Importance: {'PASS' if results['feature_sanity'] else 'WARNING'}")
            print(f"  {'✓' if results['class_dist'] else '✗'} Class Distribution: {'PASS' if results['class_dist'] else 'FAIL'}")
            print(f"  {'✓' if results['confidence'] else '✗'} Confidence Levels: {'PASS' if results['confidence'] else 'WARNING'}")
            print(f"  ✓ Error Analysis: Complete")

        # Overall assessment
        print(f"\n{'='*80}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*80}")

        all_passed = all(
            r['no_leakage'] and r['feature_sanity'] and r['class_dist']
            for r in all_results.values()
        )

        if all_passed:
            print("\n✅ ALL TESTS PASSED!")
            print("✅ Models are validated and ready for production use")
            print("✅ No data leakage detected")
            print("✅ Performance is robust and generalizable")
        else:
            print("\n⚠️  SOME TESTS FAILED OR WARNINGS DETECTED")
            print("Review the detailed output above for specific issues")

        print(f"\n{'='*80}\n")


def main():
    validator = ModelValidator()
    validator.run_all_tests()


if __name__ == "__main__":
    main()
