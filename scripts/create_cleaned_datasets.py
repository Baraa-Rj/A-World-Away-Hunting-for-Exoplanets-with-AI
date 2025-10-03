import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DatasetCleaner:
    """Create cleaned datasets with only important features for ML training"""

    def __init__(self):
        self.cleaned_datasets = {}

    def clean_cumulative(self):
        """Clean Cumulative (Kepler) dataset"""
        print("="*80)
        print("CLEANING CUMULATIVE (KEPLER) DATASET")
        print("="*80)

        # Load data
        df = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        print(f"\nOriginal shape: {df.shape}")

        # Define important features based on feature importance analysis
        important_features = [
            # Target variable
            'koi_disposition',

            # Top planet features (from importance analysis)
            'koi_period',           # 0.1242 - Orbital period
            'koi_teq',              # 0.1133 - Equilibrium temperature
            'koi_insol',            # 0.0925 - Insolation flux
            'koi_impact',           # 0.0772 - Impact parameter
            'koi_duration',         # 0.0752 - Transit duration
            'koi_prad',             # 0.0728 - Planet radius
            'koi_depth',            # 0.0633 - Transit depth

            # Top stellar features
            'koi_steff',            # 0.0431 - Stellar temperature
            'koi_slogg',            # 0.0373 - Stellar surface gravity
            'koi_srad',             # 0.0363 - Stellar radius
            'koi_smass',            # 0.0322 - Stellar mass
            'koi_smet',             # 0.0644 - Stellar metallicity

            # Derived features
            'planet_star_radius_ratio',  # 0.0842
            'planet_volume',             # 0.0832
            'in_habitable_zone',         # Habitability indicator

            # Encoded target (useful for some models)
            'koi_disposition_encoded',
            'koi_pdisposition_encoded',

            # Coordinates (for potential spatial analysis)
            'ra',
            'dec',
        ]

        # Filter available columns
        available_features = [col for col in important_features if col in df.columns]
        print(f"\nSelected features: {len(available_features)}")

        # Create cleaned dataset
        df_clean = df[available_features].copy()

        # Remove rows with missing values in critical features
        critical_features = ['koi_disposition', 'koi_period', 'koi_prad', 'koi_depth']
        df_clean = df_clean.dropna(subset=critical_features)

        print(f"Shape after removing missing critical values: {df_clean.shape}")
        print(f"Features removed: {df.shape[1] - df_clean.shape[1]}")
        print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

        # Display selected features
        print(f"\n✅ SELECTED FEATURES ({len(df_clean.columns)}):")
        for i, col in enumerate(df_clean.columns, 1):
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
            print(f"  {i:2d}. {col:35s} - Missing: {missing_pct:5.1f}%")

        # Display target distribution
        if 'koi_disposition' in df_clean.columns:
            print(f"\n📊 TARGET DISTRIBUTION:")
            print(df_clean['koi_disposition'].value_counts())
            print(f"\nPercentages:")
            print(df_clean['koi_disposition'].value_counts(normalize=True) * 100)

        # Save cleaned dataset
        output_path = 'data/processed/cumulative_cleaned.csv'
        df_clean.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")

        self.cleaned_datasets['cumulative'] = df_clean
        return df_clean

    def clean_k2pandc(self):
        """Clean K2PANDC dataset"""
        print("\n" + "="*80)
        print("CLEANING K2PANDC DATASET")
        print("="*80)

        # Load data
        df = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        print(f"\nOriginal shape: {df.shape}")

        # Define important features
        important_features = [
            # Target variable
            'disposition',

            # Top planet features (from importance analysis)
            'pl_orbper',            # 0.1415 - Orbital period
            'pl_rade',              # 0.0610 - Planet radius (Earth radii)
            'pl_masse',             # 0.0000 - Planet mass (Earth masses)
            'pl_orbeccen',          # 0.0043 - Orbital eccentricity
            'pl_insol',             # 0.0115 - Insolation flux
            'pl_eqt',               # 0.0694 - Equilibrium temperature
            'pl_orbincl',           # 0.0599 - Orbital inclination

            # Top stellar features
            'st_teff',              # 0.0855 - Stellar temperature
            'st_rad',               # 0.1126 - Stellar radius
            'st_mass',              # 0.0916 - Stellar mass
            'st_logg',              # 0.0921 - Stellar surface gravity
            'st_met',               # 0.0947 - Stellar metallicity

            # Derived features
            'planet_density',       # 0.0597 - Planet density
            'surface_gravity',      # 0.0563 - Surface gravity
            'escape_velocity',      # 0.0600 - Escape velocity

            # Discovery method (categorical)
            'discoverymethod',
            'discoverymethod_encoded',

            # Encoded target
            'disposition_encoded',

            # Coordinates
            'ra',
            'dec',
        ]

        # Filter available columns
        available_features = [col for col in important_features if col in df.columns]
        print(f"\nSelected features: {len(available_features)}")

        # Create cleaned dataset
        df_clean = df[available_features].copy()

        # Remove rows with missing values in critical features
        critical_features = ['disposition', 'pl_orbper', 'pl_rade']
        df_clean = df_clean.dropna(subset=critical_features)

        print(f"Shape after removing missing critical values: {df_clean.shape}")
        print(f"Features removed: {df.shape[1] - df_clean.shape[1]}")
        print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

        # Display selected features
        print(f"\n✅ SELECTED FEATURES ({len(df_clean.columns)}):")
        for i, col in enumerate(df_clean.columns, 1):
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
            print(f"  {i:2d}. {col:35s} - Missing: {missing_pct:5.1f}%")

        # Display target distribution
        if 'disposition' in df_clean.columns:
            print(f"\n📊 TARGET DISTRIBUTION:")
            print(df_clean['disposition'].value_counts())
            print(f"\nPercentages:")
            print(df_clean['disposition'].value_counts(normalize=True) * 100)

        # Save cleaned dataset
        output_path = 'data/processed/k2pandc_cleaned.csv'
        df_clean.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")

        self.cleaned_datasets['k2pandc'] = df_clean
        return df_clean

    def clean_toi(self):
        """Clean TOI (TESS) dataset"""
        print("\n" + "="*80)
        print("CLEANING TOI (TESS) DATASET")
        print("="*80)

        # Load data
        df = pd.read_csv('data/processed/toi_preprocessed.csv')
        print(f"\nOriginal shape: {df.shape}")

        # Define important features
        important_features = [
            # Target variable
            'tfopwg_disp',

            # Top planet features (from importance analysis)
            'pl_orbper',            # 0.0719 - Orbital period
            'pl_rade',              # 0.0834 - Planet radius
            'pl_trandurh',          # 0.0647 - Transit duration (hours)
            'pl_trandep',           # 0.0740 - Transit depth
            'pl_insol',             # 0.0769 - Insolation flux
            'pl_eqt',               # 0.0796 - Equilibrium temperature

            # Top stellar features
            'st_teff',              # 0.0535 - Stellar temperature
            'st_rad',               # 0.0513 - Stellar radius
            'st_logg',              # 0.0487 - Stellar surface gravity
            'st_tmag',              # 0.1429 - TESS magnitude (MOST IMPORTANT)
            'st_dist',              # 0.0940 - Distance to star

            # Derived features
            'transit_snr_approx',   # 0.0719 - Transit SNR approximation
            'distance_modulus',     # 0.0868 - Distance modulus
            'in_habitable_zone',    # Habitability indicator

            # Encoded target
            'tfopwg_disp_encoded',

            # Coordinates
            'ra',
            'dec',

            # Additional useful features
            'pl_tranmid',           # Transit midpoint (for timing analysis)
            'st_pmra',              # Proper motion RA
            'st_pmdec',             # Proper motion DEC
        ]

        # Filter available columns
        available_features = [col for col in important_features if col in df.columns]
        print(f"\nSelected features: {len(available_features)}")

        # Create cleaned dataset
        df_clean = df[available_features].copy()

        # Remove rows with missing values in critical features
        critical_features = ['tfopwg_disp', 'pl_orbper', 'pl_rade', 'st_tmag']
        df_clean = df_clean.dropna(subset=critical_features)

        print(f"Shape after removing missing critical values: {df_clean.shape}")
        print(f"Features removed: {df.shape[1] - df_clean.shape[1]}")
        print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

        # Display selected features
        print(f"\n✅ SELECTED FEATURES ({len(df_clean.columns)}):")
        for i, col in enumerate(df_clean.columns, 1):
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
            print(f"  {i:2d}. {col:35s} - Missing: {missing_pct:5.1f}%")

        # Display target distribution
        if 'tfopwg_disp' in df_clean.columns:
            print(f"\n📊 TARGET DISTRIBUTION:")
            print(df_clean['tfopwg_disp'].value_counts())
            print(f"\nPercentages:")
            print(df_clean['tfopwg_disp'].value_counts(normalize=True) * 100)

        # Save cleaned dataset
        output_path = 'data/processed/toi_cleaned.csv'
        df_clean.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")

        self.cleaned_datasets['toi'] = df_clean
        return df_clean

    def generate_summary_report(self):
        """Generate summary report of cleaning process"""
        print("\n" + "="*80)
        print("CLEANING SUMMARY REPORT")
        print("="*80)

        # Load original preprocessed datasets for comparison
        cum_orig = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        k2_orig = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        toi_orig = pd.read_csv('data/processed/toi_preprocessed.csv')

        summary_data = []

        for name in ['cumulative', 'k2pandc', 'toi']:
            if name == 'cumulative':
                orig = cum_orig
            elif name == 'k2pandc':
                orig = k2_orig
            else:
                orig = toi_orig

            cleaned = self.cleaned_datasets[name]

            summary_data.append({
                'Dataset': name.upper(),
                'Original Rows': orig.shape[0],
                'Original Cols': orig.shape[1],
                'Cleaned Rows': cleaned.shape[0],
                'Cleaned Cols': cleaned.shape[1],
                'Rows Removed': orig.shape[0] - cleaned.shape[0],
                'Cols Removed': orig.shape[1] - cleaned.shape[1],
                'Row Retention %': (cleaned.shape[0] / orig.shape[0]) * 100,
                'Col Retention %': (cleaned.shape[1] / orig.shape[1]) * 100
            })

        # Create summary table
        print(f"\n{'Dataset':<15} {'Orig Rows':>12} {'Clean Rows':>12} {'Rows Removed':>15} {'Retention':>12}")
        print("-" * 80)
        for item in summary_data:
            print(f"{item['Dataset']:<15} {item['Original Rows']:>12,} {item['Cleaned Rows']:>12,} "
                  f"{item['Rows Removed']:>15,} {item['Row Retention %']:>11.1f}%")

        print(f"\n{'Dataset':<15} {'Orig Cols':>12} {'Clean Cols':>12} {'Cols Removed':>15} {'Retention':>12}")
        print("-" * 80)
        for item in summary_data:
            print(f"{item['Dataset']:<15} {item['Original Cols']:>12,} {item['Cleaned Cols']:>12,} "
                  f"{item['Cols Removed']:>15,} {item['Col Retention %']:>11.1f}%")

        # Total statistics
        print("\n" + "="*80)
        print("TOTAL STATISTICS")
        print("="*80)

        total_orig_rows = sum(item['Original Rows'] for item in summary_data)
        total_clean_rows = sum(item['Cleaned Rows'] for item in summary_data)
        total_orig_cols = sum(item['Original Cols'] for item in summary_data)
        total_clean_cols = sum(item['Cleaned Cols'] for item in summary_data)

        print(f"\nTotal Records:")
        print(f"  Original:  {total_orig_rows:,}")
        print(f"  Cleaned:   {total_clean_rows:,}")
        print(f"  Removed:   {total_orig_rows - total_clean_rows:,}")
        print(f"  Retention: {(total_clean_rows / total_orig_rows) * 100:.1f}%")

        print(f"\nTotal Features:")
        print(f"  Original:  {total_orig_cols:,}")
        print(f"  Cleaned:   {total_clean_cols:,}")
        print(f"  Removed:   {total_orig_cols - total_clean_cols:,}")
        print(f"  Retention: {(total_clean_cols / total_orig_cols) * 100:.1f}%")

        print("\n" + "="*80)
        print("✅ CLEANED DATASETS READY FOR ML TRAINING!")
        print("="*80)

        print("\nOutput Files:")
        print("  📁 data/processed/cumulative_cleaned.csv")
        print("  📁 data/processed/k2pandc_cleaned.csv")
        print("  📁 data/processed/toi_cleaned.csv")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. ✅ Feature selection - COMPLETE")
        print("2. 📋 Create train/validation/test splits")
        print("3. 📋 Handle class imbalance (SMOTE/class weights)")
        print("4. 📋 Train baseline models (Random Forest, XGBoost)")
        print("5. 📋 Evaluate and optimize models")

        print("\n" + "="*80)

def main():
    cleaner = DatasetCleaner()

    # Clean all datasets
    cleaner.clean_cumulative()
    cleaner.clean_k2pandc()
    cleaner.clean_toi()

    # Generate summary report
    cleaner.generate_summary_report()

if __name__ == "__main__":
    main()
