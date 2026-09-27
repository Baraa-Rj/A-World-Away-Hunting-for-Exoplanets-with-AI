"""Quick test of enhanced model vs baseline"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def prepare_data(dataset_name='cumulative'):
    df = pd.read_csv(f'data/processed/{dataset_name}_cleaned.csv')

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

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded

print("="*80)
print("QUICK MODEL COMPARISON")
print("="*80)

for dataset in ['cumulative', 'k2pandc', 'toi']:
    print(f"\n{dataset.upper()}:")

    X, y = prepare_data(dataset)

    # Current model
    current = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42,
        class_weight='balanced', n_jobs=-1
    )

    # Enhanced model (main fix: reduce overfitting)
    enhanced = RandomForestClassifier(
        n_estimators=200,          # 2x trees
        max_depth=15,              # Shallower (less overfitting)
        min_samples_split=10,      # More regularization
        min_samples_leaf=4,        # More regularization
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    # Gradient Boosting alternative
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Only 3 folds for speed

    print("  Testing baseline...")
    current_scores = cross_val_score(current, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print("  Testing enhanced RF...")
    enhanced_scores = cross_val_score(enhanced, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print("  Testing Gradient Boosting...")
    gb_scores = cross_val_score(gb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"\n  Results (3-fold CV):")
    print(f"    Baseline RF:      {current_scores.mean()*100:.2f}% ± {current_scores.std()*100:.2f}%")
    print(f"    Enhanced RF:      {enhanced_scores.mean()*100:.2f}% ± {enhanced_scores.std()*100:.2f}%")
    print(f"    GradientBoosting: {gb_scores.mean()*100:.2f}% ± {gb_scores.std()*100:.2f}%")

    best_score = max(current_scores.mean(), enhanced_scores.mean(), gb_scores.mean())
    improvement = (best_score - current_scores.mean()) * 100

    if improvement > 0:
        print(f"    ✅ Best improvement: +{improvement:.2f}%")
    else:
        print(f"    ⚠️  No improvement")

print(f"\n{'='*80}")
print("KEY INSIGHTS:")
print("  • Enhanced RF reduces overfitting with regularization")
print("  • GradientBoosting often works better on small datasets")
print("  • To get 75-80%, consider XGBoost or more data")
print("="*80)
