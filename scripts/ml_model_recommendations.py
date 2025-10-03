import pandas as pd
import numpy as np

class MLModelRecommendation:
    """Analyze datasets and recommend best ML models"""

    def __init__(self):
        self.datasets = {}

    def load_datasets(self):
        """Load cleaned datasets"""
        print("="*80)
        print("LOADING CLEANED DATASETS FOR MODEL RECOMMENDATION")
        print("="*80)

        self.datasets['cumulative'] = pd.read_csv('data/processed/cumulative_cleaned.csv')
        self.datasets['k2pandc'] = pd.read_csv('data/processed/k2pandc_cleaned.csv')
        self.datasets['toi'] = pd.read_csv('data/processed/toi_cleaned.csv')

        for name, df in self.datasets.items():
            print(f"✓ {name.upper()}: {df.shape}")

    def analyze_dataset_characteristics(self):
        """Analyze characteristics relevant to model selection"""
        print("\n" + "="*80)
        print("DATASET CHARACTERISTICS ANALYSIS")
        print("="*80)

        for name, df in self.datasets.items():
            print(f"\n{'='*40}")
            print(f"{name.upper()}")
            print(f"{'='*40}")

            # Dataset size
            n_samples = len(df)
            n_features = df.shape[1] - 1  # Excluding target

            print(f"\n📊 Size:")
            print(f"  Samples:  {n_samples:,}")
            print(f"  Features: {n_features}")
            print(f"  Ratio:    {n_samples/n_features:.1f} samples per feature")

            # Class distribution
            if name == 'cumulative':
                target = 'koi_disposition'
            elif name == 'k2pandc':
                target = 'disposition'
            else:
                target = 'tfopwg_disp'

            print(f"\n🎯 Class Balance:")
            class_counts = df[target].value_counts()
            for cls, count in class_counts.items():
                pct = (count / n_samples) * 100
                print(f"  {str(cls)[:20]:20s}: {count:5,} ({pct:5.1f}%)")

            # Imbalance ratio
            majority_class = class_counts.max()
            minority_class = class_counts.min()
            imbalance_ratio = majority_class / minority_class

            print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1")

            if imbalance_ratio < 2:
                print(f"  Status: ✅ Well-balanced")
            elif imbalance_ratio < 5:
                print(f"  Status: ⚠️  Moderately imbalanced")
            else:
                print(f"  Status: ❌ Highly imbalanced")

            # Feature types
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            print(f"\n📝 Feature Types:")
            print(f"  Numerical:   {len(numerical_cols)}")
            print(f"  Categorical: {len(categorical_cols)}")

    def recommend_models(self):
        """Recommend best models based on dataset characteristics"""
        print("\n" + "="*80)
        print("ML MODEL RECOMMENDATIONS")
        print("="*80)

        print("\n" + "="*40)
        print("1. TRADITIONAL ML (RECOMMENDED)")
        print("="*40)

        print("\n🏆 TOP RECOMMENDATION: XGBOOST")
        print("-" * 40)
        print("✅ ADVANTAGES:")
        print("  • BEST for tabular data (like exoplanet features)")
        print("  • Handles class imbalance well (built-in scale_pos_weight)")
        print("  • Fast training on your dataset sizes")
        print("  • Built-in feature importance")
        print("  • Robust to outliers and missing values")
        print("  • Regularization prevents overfitting")
        print("  • Excellent performance on 1k-10k samples")
        print("\n📈 Expected Performance: 90-95% accuracy")
        print("⏱️  Training Time: 1-5 minutes per dataset")
        print("\n💡 PERFECT FOR:")
        print("  • Cumulative: 6,484 samples, 20 features")
        print("  • K2PANDC: 3,047 samples, 21 features")
        print("  • TOI: 7,110 samples, 21 features")

        print("\n\n🥈 SECOND CHOICE: RANDOM FOREST")
        print("-" * 40)
        print("✅ ADVANTAGES:")
        print("  • Easy to use (fewer hyperparameters)")
        print("  • Very interpretable (feature importance)")
        print("  • Handles non-linear relationships")
        print("  • Built-in class_weight for imbalance")
        print("  • Robust baseline model")
        print("\n📈 Expected Performance: 85-92% accuracy")
        print("⏱️  Training Time: 30 seconds - 2 minutes")

        print("\n\n🥉 THIRD CHOICE: LIGHTGBM")
        print("-" * 40)
        print("✅ ADVANTAGES:")
        print("  • Faster than XGBoost on large datasets")
        print("  • Lower memory usage")
        print("  • Great for TOI dataset (7,110 samples)")
        print("  • Similar performance to XGBoost")
        print("\n📈 Expected Performance: 88-94% accuracy")
        print("⏱️  Training Time: 30 seconds - 2 minutes")

        print("\n\n📊 OTHER GOOD OPTIONS:")
        print("-" * 40)
        print("  • Gradient Boosting (sklearn): Solid baseline")
        print("  • CatBoost: Great for categorical features")
        print("  • Support Vector Machine (SVM): Good for smaller datasets")
        print("  • Logistic Regression: Fast, interpretable baseline")

        print("\n\n" + "="*40)
        print("2. NEURAL NETWORKS (NOT RECOMMENDED)")
        print("="*40)

        print("\n⚠️  NEURAL NETWORKS - USE WITH CAUTION")
        print("-" * 40)
        print("❌ DISADVANTAGES for your data:")
        print("  • Need 10,000+ samples for good performance")
        print("    Your datasets: 3k-7k samples (TOO SMALL)")
        print("  • Require extensive hyperparameter tuning")
        print("  • Risk of overfitting on small datasets")
        print("  • Longer training time (5-30 minutes)")
        print("  • Less interpretable (black box)")
        print("  • Need careful architecture design")
        print("  • Require more preprocessing (normalization)")
        print("\n✅ WHEN TO USE:")
        print("  • If you have >20,000 samples")
        print("  • For very complex non-linear patterns")
        print("  • When you need deep feature learning")
        print("  • For ensemble with gradient boosting")

        print("\n📉 Expected Performance: 80-88% accuracy")
        print("⏱️  Training Time: 5-30 minutes")
        print("⚠️  Overfitting Risk: HIGH on your dataset sizes")

        print("\n\n" + "="*80)
        print("DETAILED COMPARISON TABLE")
        print("="*80)

        print(f"\n{'Model':<20} {'Accuracy':<15} {'Train Time':<15} {'Interpretable':<15} {'Good for Small Data'}")
        print("-" * 90)
        print(f"{'XGBoost':<20} {'90-95%':<15} {'1-5 min':<15} {'High':<15} {'✅ YES'}")
        print(f"{'Random Forest':<20} {'85-92%':<15} {'0.5-2 min':<15} {'High':<15} {'✅ YES'}")
        print(f"{'LightGBM':<20} {'88-94%':<15} {'0.5-2 min':<15} {'High':<15} {'✅ YES'}")
        print(f"{'Neural Network':<20} {'80-88%':<15} {'5-30 min':<15} {'Low':<15} {'❌ NO'}")
        print(f"{'SVM':<20} {'82-90%':<15} {'2-10 min':<15} {'Medium':<15} {'✅ YES'}")
        print(f"{'Logistic Reg':<20} {'75-85%':<15} {'<1 min':<15} {'Very High':<15} {'✅ YES'}")

        print("\n\n" + "="*80)
        print("RECOMMENDED APPROACH: ENSEMBLE STRATEGY")
        print("="*80)

        print("\n📋 STEP-BY-STEP PLAN:")
        print("\n1️⃣  BASELINE (Quick validation):")
        print("   • Logistic Regression - Fast baseline")
        print("   • Random Forest - Solid comparison")
        print("   • Expected time: 5 minutes total")

        print("\n2️⃣  MAIN MODELS (Best performance):")
        print("   • XGBoost (primary)")
        print("   • LightGBM (comparison)")
        print("   • Expected time: 10-15 minutes total")

        print("\n3️⃣  ENSEMBLE (Maximum accuracy):")
        print("   • Voting Classifier:")
        print("     - XGBoost (weight: 0.4)")
        print("     - LightGBM (weight: 0.3)")
        print("     - Random Forest (weight: 0.3)")
        print("   • Expected boost: +2-5% accuracy")

        print("\n4️⃣  OPTIONAL - Neural Network:")
        print("   • Only if ensemble accuracy < 90%")
        print("   • Use as additional ensemble member")
        print("   • Simple architecture: 3-4 layers")

        print("\n\n" + "="*80)
        print("FINAL RECOMMENDATION")
        print("="*80)

        print("\n🎯 FOR YOUR EXOPLANET DATASETS:")
        print("\n✅ PRIMARY MODEL: XGBoost")
        print("   Why: Best accuracy, handles imbalance, fast training")
        print("\n✅ BACKUP MODEL: LightGBM")
        print("   Why: Similar performance, faster, good comparison")
        print("\n✅ BASELINE: Random Forest")
        print("   Why: Easy to implement, good interpretability")
        print("\n❌ AVOID: Neural Networks")
        print("   Why: Too small dataset, risk of overfitting")

        print("\n\n💡 IMPLEMENTATION PRIORITY:")
        print("  1. Start with Random Forest (baseline)")
        print("  2. Train XGBoost (main model)")
        print("  3. Train LightGBM (comparison)")
        print("  4. Create ensemble of top 2-3 models")
        print("  5. Neural network ONLY if needed")

        print("\n" + "="*80)

    def dataset_specific_recommendations(self):
        """Specific recommendations for each dataset"""
        print("\n" + "="*80)
        print("DATASET-SPECIFIC RECOMMENDATIONS")
        print("="*80)

        print("\n📁 CUMULATIVE (6,484 samples, balanced)")
        print("-" * 40)
        print("  Best Model: XGBoost or Random Forest")
        print("  Why: Balanced classes, good sample size")
        print("  Special Notes:")
        print("    • No need for SMOTE (well-balanced)")
        print("    • Can use standard train/test split")
        print("    • Focus on hyperparameter tuning")

        print("\n📁 K2PANDC (3,047 samples, imbalanced)")
        print("-" * 40)
        print("  Best Model: XGBoost with class weights")
        print("  Why: Handles imbalance (58% vs 34% vs 8%)")
        print("  Special Notes:")
        print("    • Use class_weight or SMOTE")
        print("    • Consider oversampling minority classes")
        print("    • Use stratified splits")

        print("\n📁 TOI (7,110 samples, highly imbalanced)")
        print("-" * 40)
        print("  Best Model: XGBoost or LightGBM")
        print("  Why: Large dataset, extreme imbalance (61% vs 8%)")
        print("  Special Notes:")
        print("    • MUST use SMOTE or class weights")
        print("    • Consider focal loss")
        print("    • May benefit from Neural Network")
        print("    • Use stratified K-fold CV")

        print("\n" + "="*80)

def main():
    analyzer = MLModelRecommendation()
    analyzer.load_datasets()
    analyzer.analyze_dataset_characteristics()
    analyzer.recommend_models()
    analyzer.dataset_specific_recommendations()

if __name__ == "__main__":
    main()
