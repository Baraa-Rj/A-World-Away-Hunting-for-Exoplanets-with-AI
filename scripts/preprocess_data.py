import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ExoplanetPreprocessor:
    """Comprehensive preprocessing for NASA Exoplanet datasets"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def load_data(self):
        """Load all three datasets"""
        print("Loading datasets...")
        self.cumulative = pd.read_csv('cumulative_2025.10.03_00.50.03.csv', comment='#')
        self.k2pandc = pd.read_csv('k2pandc_2025.10.03_00.53.03.csv', comment='#')
        self.toi = pd.read_csv('TOI_2025.10.03_00.51.26.csv', comment='#')
        print(f"✓ Loaded cumulative: {self.cumulative.shape}")
        print(f"✓ Loaded k2pandc: {self.k2pandc.shape}")
        print(f"✓ Loaded TOI: {self.toi.shape}")

    def preprocess_cumulative(self):
        """Preprocess Kepler cumulative dataset"""
        print("\n" + "="*80)
        print("PREPROCESSING CUMULATIVE DATASET")
        print("="*80)

        df = self.cumulative.copy()

        # 1. Handle missing values in key columns
        print("\n1. Handling missing values...")

        # Key planet parameters
        planet_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
                      'koi_teq', 'koi_insol', 'koi_impact']

        # Key stellar parameters
        stellar_cols = ['koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass']

        # Fill missing values with median for numerical columns
        for col in planet_cols + stellar_cols:
            if col in df.columns:
                missing_before = df[col].isnull().sum()
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  {col}: filled {missing_before} missing values with median")

        # 2. Remove outliers using IQR method
        print("\n2. Removing outliers...")
        initial_rows = len(df)

        for col in ['koi_period', 'koi_prad', 'koi_depth']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before - len(df)
                if removed > 0:
                    print(f"  {col}: removed {removed} outliers")

        print(f"  Total rows removed: {initial_rows - len(df)}")

        # 3. Encode categorical variables
        print("\n3. Encoding categorical variables...")

        if 'koi_disposition' in df.columns:
            le = LabelEncoder()
            df['koi_disposition_encoded'] = le.fit_transform(df['koi_disposition'].fillna('UNKNOWN'))
            self.encoders['koi_disposition'] = le
            print(f"  koi_disposition: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        if 'koi_pdisposition' in df.columns:
            le = LabelEncoder()
            df['koi_pdisposition_encoded'] = le.fit_transform(df['koi_pdisposition'].fillna('UNKNOWN'))
            self.encoders['koi_pdisposition'] = le
            print(f"  koi_pdisposition: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # 4. Feature engineering
        print("\n4. Creating derived features...")

        # Planet-star radius ratio
        if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
            df['planet_star_radius_ratio'] = df['koi_prad'] / (df['koi_srad'] * 109.2)  # Sun radius in Earth radii
            print("  ✓ Created planet_star_radius_ratio")

        # Density estimate (if mass available)
        if 'koi_prad' in df.columns:
            # Volume in Earth volumes (4/3 * π * r³)
            df['planet_volume'] = (4/3) * np.pi * (df['koi_prad'] ** 3)
            print("  ✓ Created planet_volume")

        # Habitable zone indicator (simplified)
        if 'koi_teq' in df.columns:
            df['in_habitable_zone'] = ((df['koi_teq'] >= 200) & (df['koi_teq'] <= 350)).astype(int)
            print(f"  ✓ Created in_habitable_zone ({df['in_habitable_zone'].sum()} planets)")

        # 5. Normalize numerical features
        print("\n5. Normalizing numerical features...")

        numerical_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
                         'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass']

        scaler = StandardScaler()
        available_cols = [col for col in numerical_cols if col in df.columns]

        if available_cols:
            df[available_cols] = scaler.fit_transform(df[available_cols])
            self.scalers['cumulative'] = scaler
            print(f"  ✓ Normalized {len(available_cols)} columns")

        self.cumulative_processed = df
        print(f"\n✓ Final shape: {df.shape}")

        return df

    def preprocess_k2pandc(self):
        """Preprocess K2 and confirmed planets dataset"""
        print("\n" + "="*80)
        print("PREPROCESSING K2PANDC DATASET")
        print("="*80)

        df = self.k2pandc.copy()

        # 1. Handle missing values
        print("\n1. Handling missing values...")

        planet_cols = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_orbeccen',
                      'pl_insol', 'pl_eqt', 'pl_orbincl']
        stellar_cols = ['st_teff', 'st_rad', 'st_mass', 'st_logg']

        for col in planet_cols + stellar_cols:
            if col in df.columns:
                missing_before = df[col].isnull().sum()
                df[col].fillna(df[col].median(), inplace=True)
                if missing_before > 0:
                    print(f"  {col}: filled {missing_before} missing values")

        # 2. Remove outliers
        print("\n2. Removing outliers...")
        initial_rows = len(df)

        for col in ['pl_orbper', 'pl_rade', 'pl_masse']:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before - len(df)
                if removed > 0:
                    print(f"  {col}: removed {removed} outliers")

        print(f"  Total rows removed: {initial_rows - len(df)}")

        # 3. Encode categorical variables
        print("\n3. Encoding categorical variables...")

        if 'discoverymethod' in df.columns:
            le = LabelEncoder()
            df['discoverymethod_encoded'] = le.fit_transform(df['discoverymethod'].fillna('Unknown'))
            self.encoders['discoverymethod'] = le
            print(f"  discoverymethod: {len(le.classes_)} unique methods")

        if 'disposition' in df.columns:
            le = LabelEncoder()
            df['disposition_encoded'] = le.fit_transform(df['disposition'].fillna('Unknown'))
            self.encoders['disposition'] = le

        # 4. Feature engineering
        print("\n4. Creating derived features...")

        # Planet density
        if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
            # Density = mass / volume (Earth units)
            df['planet_density'] = df['pl_masse'] / ((4/3) * np.pi * (df['pl_rade'] ** 3))
            print("  ✓ Created planet_density")

        # Surface gravity
        if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
            # g = GM/r² (in Earth g units)
            df['surface_gravity'] = df['pl_masse'] / (df['pl_rade'] ** 2)
            print("  ✓ Created surface_gravity")

        # Escape velocity
        if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
            # v_esc ∝ sqrt(M/r)
            df['escape_velocity'] = np.sqrt(df['pl_masse'] / df['pl_rade'])
            print("  ✓ Created escape_velocity")

        # 5. Normalize numerical features
        print("\n5. Normalizing numerical features...")

        numerical_cols = ['pl_orbper', 'pl_rade', 'pl_masse', 'pl_orbeccen',
                         'st_teff', 'st_rad', 'st_mass', 'st_logg']

        scaler = StandardScaler()
        available_cols = [col for col in numerical_cols if col in df.columns]

        if available_cols:
            df[available_cols] = scaler.fit_transform(df[available_cols])
            self.scalers['k2pandc'] = scaler
            print(f"  ✓ Normalized {len(available_cols)} columns")

        self.k2pandc_processed = df
        print(f"\n✓ Final shape: {df.shape}")

        return df

    def preprocess_toi(self):
        """Preprocess TESS TOI dataset"""
        print("\n" + "="*80)
        print("PREPROCESSING TOI DATASET")
        print("="*80)

        df = self.toi.copy()

        # 1. Handle missing values
        print("\n1. Handling missing values...")

        planet_cols = ['pl_orbper', 'pl_rade', 'pl_trandurh', 'pl_trandep',
                      'pl_insol', 'pl_eqt']
        stellar_cols = ['st_teff', 'st_rad', 'st_logg', 'st_tmag', 'st_dist']

        for col in planet_cols + stellar_cols:
            if col in df.columns:
                missing_before = df[col].isnull().sum()
                df[col].fillna(df[col].median(), inplace=True)
                if missing_before > 0:
                    print(f"  {col}: filled {missing_before} missing values")

        # 2. Remove outliers
        print("\n2. Removing outliers...")
        initial_rows = len(df)

        for col in ['pl_orbper', 'pl_rade', 'pl_trandep']:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before - len(df)
                if removed > 0:
                    print(f"  {col}: removed {removed} outliers")

        print(f"  Total rows removed: {initial_rows - len(df)}")

        # 3. Encode categorical variables
        print("\n3. Encoding categorical variables...")

        if 'tfopwg_disp' in df.columns:
            le = LabelEncoder()
            df['tfopwg_disp_encoded'] = le.fit_transform(df['tfopwg_disp'].fillna('Unknown'))
            self.encoders['tfopwg_disp'] = le
            print(f"  tfopwg_disp: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # 4. Feature engineering
        print("\n4. Creating derived features...")

        # Transit signal-to-noise ratio approximation
        if 'pl_trandep' in df.columns and 'pl_trandurh' in df.columns:
            df['transit_snr_approx'] = df['pl_trandep'] * np.sqrt(df['pl_trandurh'])
            print("  ✓ Created transit_snr_approx")

        # Distance modulus
        if 'st_dist' in df.columns:
            df['distance_modulus'] = 5 * np.log10(df['st_dist']) - 5
            print("  ✓ Created distance_modulus")

        # Habitable zone indicator
        if 'pl_eqt' in df.columns:
            df['in_habitable_zone'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350)).astype(int)
            print(f"  ✓ Created in_habitable_zone ({df['in_habitable_zone'].sum()} candidates)")

        # 5. Normalize numerical features
        print("\n5. Normalizing numerical features...")

        numerical_cols = ['pl_orbper', 'pl_rade', 'pl_trandurh', 'pl_trandep',
                         'st_teff', 'st_rad', 'st_logg', 'st_tmag']

        scaler = StandardScaler()
        available_cols = [col for col in numerical_cols if col in df.columns]

        if available_cols:
            df[available_cols] = scaler.fit_transform(df[available_cols])
            self.scalers['toi'] = scaler
            print(f"  ✓ Normalized {len(available_cols)} columns")

        self.toi_processed = df
        print(f"\n✓ Final shape: {df.shape}")

        return df

    def save_processed_data(self):
        """Save preprocessed datasets"""
        print("\n" + "="*80)
        print("SAVING PREPROCESSED DATA")
        print("="*80)

        self.cumulative_processed.to_csv('cumulative_preprocessed.csv', index=False)
        print(f"✓ Saved cumulative_preprocessed.csv ({self.cumulative_processed.shape})")

        self.k2pandc_processed.to_csv('k2pandc_preprocessed.csv', index=False)
        print(f"✓ Saved k2pandc_preprocessed.csv ({self.k2pandc_processed.shape})")

        self.toi_processed.to_csv('toi_preprocessed.csv', index=False)
        print(f"✓ Saved toi_preprocessed.csv ({self.toi_processed.shape})")

    def generate_summary(self):
        """Generate preprocessing summary"""
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)

        summary = {
            'Cumulative': {
                'Original': self.cumulative.shape,
                'Processed': self.cumulative_processed.shape,
                'Rows removed': self.cumulative.shape[0] - self.cumulative_processed.shape[0],
                'Features added': self.cumulative_processed.shape[1] - self.cumulative.shape[1]
            },
            'K2PANDC': {
                'Original': self.k2pandc.shape,
                'Processed': self.k2pandc_processed.shape,
                'Rows removed': self.k2pandc.shape[0] - self.k2pandc_processed.shape[0],
                'Features added': self.k2pandc_processed.shape[1] - self.k2pandc.shape[1]
            },
            'TOI': {
                'Original': self.toi.shape,
                'Processed': self.toi_processed.shape,
                'Rows removed': self.toi.shape[0] - self.toi_processed.shape[0],
                'Features added': self.toi_processed.shape[1] - self.toi.shape[1]
            }
        }

        for dataset, stats in summary.items():
            print(f"\n{dataset}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        return summary

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ExoplanetPreprocessor()

    # Load data
    preprocessor.load_data()

    # Preprocess each dataset
    preprocessor.preprocess_cumulative()
    preprocessor.preprocess_k2pandc()
    preprocessor.preprocess_toi()

    # Save processed data
    preprocessor.save_processed_data()

    # Generate summary
    preprocessor.generate_summary()

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
