import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Analyze and rank feature importance for ML models"""

    def __init__(self):
        self.datasets = {}
        self.feature_importance = {}

    def load_datasets(self):
        """Load preprocessed datasets"""
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)

        self.datasets['cumulative'] = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        self.datasets['k2pandc'] = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        self.datasets['toi'] = pd.read_csv('data/processed/toi_preprocessed.csv')

        for name, df in self.datasets.items():
            print(f"✓ Loaded {name}: {df.shape}")

    def identify_feature_types(self, df):
        """Categorize features by type"""
        # Metadata columns (not useful for training)
        metadata_keywords = ['rowid', 'name', 'id', 'str', 'link', 'url', 'date',
                            'comment', 'prov', 'refname', 'flag', 'stat']

        # Error columns (typically not used directly)
        error_keywords = ['err1', 'err2', 'symerr', 'lim']

        # Encoded/derived columns
        derived_keywords = ['encoded', 'ratio', 'density', 'gravity', 'velocity',
                           'volume', 'habitable', 'snr', 'modulus']

        metadata_cols = []
        error_cols = []
        derived_cols = []
        numerical_cols = []
        categorical_cols = []

        for col in df.columns:
            col_lower = col.lower()

            # Check if metadata
            if any(keyword in col_lower for keyword in metadata_keywords):
                metadata_cols.append(col)
            # Check if error column
            elif any(keyword in col_lower for keyword in error_keywords):
                error_cols.append(col)
            # Check if derived
            elif any(keyword in col_lower for keyword in derived_keywords):
                derived_cols.append(col)
            # Check data type
            elif df[col].dtype in ['float64', 'int64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        return {
            'metadata': metadata_cols,
            'error': error_cols,
            'derived': derived_cols,
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }

    def analyze_cumulative(self):
        """Analyze Cumulative dataset features"""
        print("\n" + "="*80)
        print("CUMULATIVE (KEPLER) DATASET - FEATURE ANALYSIS")
        print("="*80)

        df = self.datasets['cumulative'].copy()

        # Categorize features
        feature_types = self.identify_feature_types(df)

        print(f"\nFeature Categories:")
        print(f"  Metadata columns:    {len(feature_types['metadata'])}")
        print(f"  Error columns:       {len(feature_types['error'])}")
        print(f"  Derived features:    {len(feature_types['derived'])}")
        print(f"  Numerical features:  {len(feature_types['numerical'])}")
        print(f"  Categorical features: {len(feature_types['categorical'])}")

        # Target variable
        target = 'koi_disposition'

        if target not in df.columns:
            print(f"\n✗ Target '{target}' not found!")
            return

        # Key planet parameters (most important for classification)
        key_planet_features = [
            'koi_period',      # Orbital period
            'koi_prad',        # Planet radius
            'koi_depth',       # Transit depth
            'koi_duration',    # Transit duration
            'koi_impact',      # Impact parameter
            'koi_teq',         # Equilibrium temperature
            'koi_insol',       # Insolation flux
        ]

        # Key stellar parameters
        key_stellar_features = [
            'koi_steff',       # Stellar effective temperature
            'koi_slogg',       # Stellar surface gravity
            'koi_srad',        # Stellar radius
            'koi_smass',       # Stellar mass
            'koi_smet',        # Stellar metallicity
        ]

        # Derived features (exclude error columns)
        derived_features = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['ratio', 'density', 'volume', 'habitable']
        ) and not any(err in col.lower() for err in ['err1', 'err2', 'symerr'])]

        # Filter available features
        key_planet = [f for f in key_planet_features if f in df.columns]
        key_stellar = [f for f in key_stellar_features if f in df.columns]
        derived_features = [f for f in derived_features if f in df.columns]

        print(f"\n\nKEY PLANET FEATURES ({len(key_planet)}):")
        for feat in key_planet:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nKEY STELLAR FEATURES ({len(key_stellar)}):")
        for feat in key_stellar:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nDERIVED FEATURES ({len(derived_features)}):")
        for feat in derived_features:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:35s} - Missing: {missing_pct:5.1f}%")

        # Calculate feature importance using Random Forest
        print(f"\n\nCALCULATING FEATURE IMPORTANCE (Random Forest)...")

        # Prepare data - remove duplicates
        features_to_test = list(set(key_planet + key_stellar + derived_features))
        df_clean = df[features_to_test + [target]].dropna()

        if len(df_clean) < 100:
            print("Not enough clean data for feature importance")
            return

        X = df_clean[features_to_test]
        y = LabelEncoder().fit_transform(df_clean[target])

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)

        # Get feature importance
        importance = pd.DataFrame({
            'feature': features_to_test,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 60)
        for i, row in importance.head(15).iterrows():
            print(f"  {row['feature']:35s} {row['importance']:.4f}")

        self.feature_importance['cumulative'] = importance

        # Recommendations
        print(f"\n\n{'='*80}")
        print("RECOMMENDATIONS FOR CUMULATIVE DATASET")
        print("="*80)

        print("\n✅ KEEP THESE FEATURES (High Importance):")
        top_features = importance.head(15)['feature'].tolist()
        for feat in top_features:
            print(f"  • {feat}")

        print("\n❌ CAN REMOVE (Low Importance):")
        low_importance = importance.tail(10)['feature'].tolist()
        for feat in low_importance:
            print(f"  • {feat}")

        print("\n⚠️  ALWAYS REMOVE:")
        print("  • Metadata: rowid, kepid, kepoi_name, kepler_name")
        print("  • Links: koi_datalink_dvr, koi_datalink_dvs")
        print("  • Dates: koi_vet_date")
        print("  • Comments: koi_comment")
        print("  • Error columns: *_err1, *_err2 (keep main values only)")
        print("  • Provenance: koi_parm_prov, koi_sparprov")

        return importance

    def analyze_k2pandc(self):
        """Analyze K2PANDC dataset features"""
        print("\n" + "="*80)
        print("K2PANDC DATASET - FEATURE ANALYSIS")
        print("="*80)

        df = self.datasets['k2pandc'].copy()

        # Categorize features
        feature_types = self.identify_feature_types(df)

        print(f"\nFeature Categories:")
        print(f"  Metadata columns:    {len(feature_types['metadata'])}")
        print(f"  Error columns:       {len(feature_types['error'])}")
        print(f"  Derived features:    {len(feature_types['derived'])}")
        print(f"  Numerical features:  {len(feature_types['numerical'])}")
        print(f"  Categorical features: {len(feature_types['categorical'])}")

        # Target variable
        target = 'disposition'

        # Key features
        key_planet_features = [
            'pl_orbper',       # Orbital period
            'pl_rade',         # Planet radius (Earth radii)
            'pl_masse',        # Planet mass (Earth masses)
            'pl_orbeccen',     # Orbital eccentricity
            'pl_insol',        # Insolation flux
            'pl_eqt',          # Equilibrium temperature
            'pl_orbincl',      # Orbital inclination
        ]

        key_stellar_features = [
            'st_teff',         # Stellar temperature
            'st_rad',          # Stellar radius
            'st_mass',         # Stellar mass
            'st_logg',         # Stellar surface gravity
            'st_met',          # Stellar metallicity
        ]

        # Derived features (exclude error columns)
        derived_features = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['density', 'gravity', 'velocity']
        ) and not any(err in col.lower() for err in ['err1', 'err2', 'symerr'])]

        # Filter available
        key_planet = [f for f in key_planet_features if f in df.columns]
        key_stellar = [f for f in key_stellar_features if f in df.columns]
        derived_features = [f for f in derived_features if f in df.columns]

        print(f"\n\nKEY PLANET FEATURES ({len(key_planet)}):")
        for feat in key_planet:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nKEY STELLAR FEATURES ({len(key_stellar)}):")
        for feat in key_stellar:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nDERIVED FEATURES ({len(derived_features)}):")
        for feat in derived_features:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:35s} - Missing: {missing_pct:5.1f}%")

        # Feature importance
        print(f"\n\nCALCULATING FEATURE IMPORTANCE...")

        features_to_test = list(set(key_planet + key_stellar + derived_features))
        df_clean = df[features_to_test + [target]].dropna()

        if len(df_clean) < 100:
            print("Not enough clean data")
            return

        X = df_clean[features_to_test]
        y = LabelEncoder().fit_transform(df_clean[target])

        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)

        importance = pd.DataFrame({
            'feature': features_to_test,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 60)
        for i, row in importance.head(15).iterrows():
            print(f"  {row['feature']:35s} {row['importance']:.4f}")

        self.feature_importance['k2pandc'] = importance

        # Recommendations
        print(f"\n\n{'='*80}")
        print("RECOMMENDATIONS FOR K2PANDC DATASET")
        print("="*80)

        print("\n✅ KEEP THESE FEATURES:")
        for feat in importance.head(15)['feature'].tolist():
            print(f"  • {feat}")

        print("\n⚠️  ALWAYS REMOVE:")
        print("  • Metadata: rowid, pl_name, hostname, pl_letter")
        print("  • Alternative names: k2_name, epic_*, hd_name, hip_name")
        print("  • IDs: tic_id, gaia_id")
        print("  • References: disc_refname, pl_refname, st_refname")
        print("  • Links and publication info")
        print("  • Flags: cb_flag, rv_flag, tran_flag, etc.")
        print("  • Error columns: *_err1, *_err2")

        return importance

    def analyze_toi(self):
        """Analyze TOI dataset features"""
        print("\n" + "="*80)
        print("TOI (TESS) DATASET - FEATURE ANALYSIS")
        print("="*80)

        df = self.datasets['toi'].copy()

        # Categorize features
        feature_types = self.identify_feature_types(df)

        print(f"\nFeature Categories:")
        print(f"  Metadata columns:    {len(feature_types['metadata'])}")
        print(f"  Error columns:       {len(feature_types['error'])}")
        print(f"  Derived features:    {len(feature_types['derived'])}")
        print(f"  Numerical features:  {len(feature_types['numerical'])}")
        print(f"  Categorical features: {len(feature_types['categorical'])}")

        # Target variable
        target = 'tfopwg_disp'

        # Key features
        key_planet_features = [
            'pl_orbper',       # Orbital period
            'pl_rade',         # Planet radius
            'pl_trandurh',     # Transit duration
            'pl_trandep',      # Transit depth
            'pl_insol',        # Insolation flux
            'pl_eqt',          # Equilibrium temperature
        ]

        key_stellar_features = [
            'st_teff',         # Stellar temperature
            'st_rad',          # Stellar radius
            'st_logg',         # Stellar surface gravity
            'st_tmag',         # TESS magnitude
            'st_dist',         # Distance
        ]

        # Derived features (exclude error columns)
        derived_features = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['habitable', 'snr', 'modulus']
        ) and not any(err in col.lower() for err in ['err1', 'err2', 'symerr'])]

        # Filter available
        key_planet = [f for f in key_planet_features if f in df.columns]
        key_stellar = [f for f in key_stellar_features if f in df.columns]
        derived_features = [f for f in derived_features if f in df.columns]

        print(f"\n\nKEY PLANET FEATURES ({len(key_planet)}):")
        for feat in key_planet:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nKEY STELLAR FEATURES ({len(key_stellar)}):")
        for feat in key_stellar:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:20s} - Missing: {missing_pct:5.1f}%")

        print(f"\nDERIVED FEATURES ({len(derived_features)}):")
        for feat in derived_features:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            print(f"  • {feat:35s} - Missing: {missing_pct:5.1f}%")

        # Feature importance
        print(f"\n\nCALCULATING FEATURE IMPORTANCE...")

        features_to_test = list(set(key_planet + key_stellar + derived_features))
        df_clean = df[features_to_test + [target]].dropna()

        if len(df_clean) < 100:
            print("Not enough clean data")
            return

        X = df_clean[features_to_test]
        y = LabelEncoder().fit_transform(df_clean[target])

        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)

        importance = pd.DataFrame({
            'feature': features_to_test,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
        print("-" * 60)
        for i, row in importance.head(15).iterrows():
            print(f"  {row['feature']:35s} {row['importance']:.4f}")

        self.feature_importance['toi'] = importance

        # Recommendations
        print(f"\n\n{'='*80}")
        print("RECOMMENDATIONS FOR TOI DATASET")
        print("="*80)

        print("\n✅ KEEP THESE FEATURES:")
        for feat in importance.head(15)['feature'].tolist():
            print(f"  • {feat}")

        print("\n⚠️  ALWAYS REMOVE:")
        print("  • Metadata: rowid, toi, toipfx, tid, ctoi_alias")
        print("  • String columns: rastr, decstr")
        print("  • Error/limit columns: *err1, *err2, *lim, *symerr")

        return importance

    def generate_final_recommendations(self):
        """Generate final feature selection recommendations"""
        print("\n" + "="*80)
        print("FINAL FEATURE SELECTION RECOMMENDATIONS")
        print("="*80)

        print("\n" + "="*40)
        print("GENERAL PRINCIPLES")
        print("="*40)

        print("\n1. ALWAYS REMOVE:")
        print("   • Metadata: IDs, names, row numbers")
        print("   • References: URLs, publication links, reference names")
        print("   • Dates: Discovery dates, update dates")
        print("   • Comments and text fields")
        print("   • Error columns: *_err1, *_err2, *_symerr, *_lim")
        print("   • Provenance/source columns")
        print("   • Flags (unless they provide useful information)")

        print("\n2. KEEP HIGH-VALUE FEATURES:")
        print("   ✓ Planet physical properties (radius, mass, period)")
        print("   ✓ Orbital characteristics (eccentricity, inclination)")
        print("   ✓ Transit measurements (depth, duration)")
        print("   ✓ Stellar properties (temperature, radius, mass)")
        print("   ✓ Derived features (density, gravity, habitability)")

        print("\n3. HANDLE MISSING VALUES:")
        print("   • Features with >70% missing → Consider removing")
        print("   • Features with <30% missing → Keep and impute")
        print("   • 30-70% missing → Evaluate importance first")

        print("\n\n" + "="*80)
        print("NEXT STEPS: RECOMMENDED WORKFLOW")
        print("="*80)

        print("\n📋 STEP 1: FEATURE SELECTION")
        print("   Create cleaned datasets with only selected features")
        print("   Remove metadata, errors, and low-importance features")

        print("\n📋 STEP 2: PREPARE FOR ML")
        print("   • Define target variables clearly")
        print("   • Split data: 70% train, 15% validation, 15% test")
        print("   • Handle class imbalance (SMOTE/class weights)")

        print("\n📋 STEP 3: TRAIN MODELS")
        print("   Start with these algorithms:")
        print("   1. Random Forest (baseline)")
        print("   2. XGBoost (likely best performance)")
        print("   3. Neural Network (if enough data)")
        print("   4. SVM (for comparison)")

        print("\n📋 STEP 4: EVALUATE")
        print("   • Accuracy, Precision, Recall, F1-score")
        print("   • Confusion matrices")
        print("   • Feature importance plots")
        print("   • Cross-validation scores")

        print("\n📋 STEP 5: OPTIMIZE")
        print("   • Hyperparameter tuning")
        print("   • Feature engineering iterations")
        print("   • Ensemble methods")

        print("\n" + "="*80)

def main():
    analyzer = FeatureImportanceAnalyzer()

    # Load datasets
    analyzer.load_datasets()

    # Analyze each dataset
    analyzer.analyze_cumulative()
    analyzer.analyze_k2pandc()
    analyzer.analyze_toi()

    # Final recommendations
    analyzer.generate_final_recommendations()

if __name__ == "__main__":
    main()
