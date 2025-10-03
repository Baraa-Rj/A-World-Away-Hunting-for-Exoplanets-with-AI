import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class DatasetAnalyzer:
    """Comprehensive analysis of preprocessed exoplanet datasets"""

    def __init__(self):
        self.datasets = {}

    def load_datasets(self):
        """Load all preprocessed datasets"""
        print("="*80)
        print("LOADING PREPROCESSED DATASETS")
        print("="*80)

        self.datasets['cumulative'] = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        self.datasets['k2pandc'] = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        self.datasets['toi'] = pd.read_csv('data/processed/toi_preprocessed.csv')

        for name, df in self.datasets.items():
            print(f"\n✓ {name.upper()}: {df.shape[0]:,} rows × {df.shape[1]} columns")

    def basic_statistics(self):
        """Display basic statistics for each dataset"""
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)

        for name, df in self.datasets.items():
            print(f"\n{'='*40}")
            print(f"{name.upper()} DATASET")
            print(f"{'='*40}")

            # Data types
            print(f"\nData Types:")
            print(df.dtypes.value_counts())

            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing Values (Top 10):")
                print(missing[missing > 0].sort_values(ascending=False).head(10))
            else:
                print(f"\n✓ No missing values!")

            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(f"\nMemory Usage: {memory_mb:.2f} MB")

    def analyze_cumulative(self):
        """Detailed analysis of Cumulative (Kepler) dataset"""
        print("\n" + "="*80)
        print("CUMULATIVE (KEPLER) DATASET ANALYSIS")
        print("="*80)

        df = self.datasets['cumulative']

        # Disposition analysis
        if 'koi_disposition' in df.columns:
            print("\n1. PLANET DISPOSITION")
            print("-" * 40)
            disp_counts = df['koi_disposition'].value_counts()
            disp_pct = df['koi_disposition'].value_counts(normalize=True) * 100

            for disp, count in disp_counts.items():
                pct = disp_pct[disp]
                print(f"  {disp:20s}: {count:5,} ({pct:5.2f}%)")

        # Planet radius distribution
        if 'koi_prad' in df.columns:
            print("\n2. PLANET RADIUS (Earth Radii)")
            print("-" * 40)
            # Note: values are normalized, so we show percentiles
            print(f"  Mean (normalized):   {df['koi_prad'].mean():.4f}")
            print(f"  Std Dev:             {df['koi_prad'].std():.4f}")
            print(f"  Min:                 {df['koi_prad'].min():.4f}")
            print(f"  25th percentile:     {df['koi_prad'].quantile(0.25):.4f}")
            print(f"  Median:              {df['koi_prad'].median():.4f}")
            print(f"  75th percentile:     {df['koi_prad'].quantile(0.75):.4f}")
            print(f"  Max:                 {df['koi_prad'].max():.4f}")

        # Orbital period distribution
        if 'koi_period' in df.columns:
            print("\n3. ORBITAL PERIOD (Days) - Normalized")
            print("-" * 40)
            print(f"  Mean (normalized):   {df['koi_period'].mean():.4f}")
            print(f"  Std Dev:             {df['koi_period'].std():.4f}")
            print(f"  Median:              {df['koi_period'].median():.4f}")

        # Habitable zone analysis
        if 'in_habitable_zone' in df.columns:
            hz_count = df['in_habitable_zone'].sum()
            hz_pct = (hz_count / len(df)) * 100
            print("\n4. HABITABLE ZONE CANDIDATES")
            print("-" * 40)
            print(f"  In Habitable Zone:   {hz_count:,} ({hz_pct:.2f}%)")
            print(f"  Outside HZ:          {len(df) - hz_count:,} ({100-hz_pct:.2f}%)")

        # Stellar properties
        if 'koi_steff' in df.columns:
            print("\n5. STELLAR EFFECTIVE TEMPERATURE (Normalized)")
            print("-" * 40)
            print(f"  Mean:                {df['koi_steff'].mean():.4f}")
            print(f"  Std Dev:             {df['koi_steff'].std():.4f}")

    def analyze_k2pandc(self):
        """Detailed analysis of K2PANDC dataset"""
        print("\n" + "="*80)
        print("K2PANDC (CONFIRMED PLANETS) DATASET ANALYSIS")
        print("="*80)

        df = self.datasets['k2pandc']

        # Discovery method
        if 'discoverymethod' in df.columns:
            print("\n1. DISCOVERY METHODS")
            print("-" * 40)
            methods = df['discoverymethod'].value_counts()
            methods_pct = df['discoverymethod'].value_counts(normalize=True) * 100

            for method, count in methods.head(10).items():
                pct = methods_pct[method]
                print(f"  {str(method)[:25]:25s}: {count:5,} ({pct:5.2f}%)")

        # Disposition
        if 'disposition' in df.columns:
            print("\n2. DISPOSITION")
            print("-" * 40)
            disp_counts = df['disposition'].value_counts()
            for disp, count in disp_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {str(disp)[:25]:25s}: {count:5,} ({pct:5.2f}%)")

        # Planet properties (normalized values)
        print("\n3. PLANET PROPERTIES (Normalized)")
        print("-" * 40)

        if 'pl_rade' in df.columns:
            print(f"  Radius - Mean: {df['pl_rade'].mean():.4f}, Std: {df['pl_rade'].std():.4f}")

        if 'pl_masse' in df.columns:
            print(f"  Mass - Mean:   {df['pl_masse'].mean():.4f}, Std: {df['pl_masse'].std():.4f}")

        if 'pl_orbper' in df.columns:
            print(f"  Period - Mean: {df['pl_orbper'].mean():.4f}, Std: {df['pl_orbper'].std():.4f}")

        # Derived features
        if 'planet_density' in df.columns:
            print("\n4. DERIVED FEATURES")
            print("-" * 40)
            print(f"  Density - Mean:       {df['planet_density'].mean():.4f}")
            print(f"  Surface Gravity - Mean: {df['surface_gravity'].mean():.4f}")
            print(f"  Escape Velocity - Mean: {df['escape_velocity'].mean():.4f}")

    def analyze_toi(self):
        """Detailed analysis of TOI dataset"""
        print("\n" + "="*80)
        print("TOI (TESS) DATASET ANALYSIS")
        print("="*80)

        df = self.datasets['toi']

        # Disposition
        if 'tfopwg_disp' in df.columns:
            print("\n1. TFOPWG DISPOSITION")
            print("-" * 40)
            disp_counts = df['tfopwg_disp'].value_counts()
            disp_pct = df['tfopwg_disp'].value_counts(normalize=True) * 100

            disposition_map = {
                'PC': 'Planet Candidate',
                'CP': 'Confirmed Planet',
                'FP': 'False Positive',
                'KP': 'Known Planet',
                'FA': 'False Alarm',
                'APC': 'Ambiguous PC'
            }

            for disp, count in disp_counts.items():
                pct = disp_pct[disp]
                name = disposition_map.get(disp, disp)
                print(f"  {disp} ({name:20s}): {count:5,} ({pct:5.2f}%)")

        # TOI properties
        print("\n2. PLANET PROPERTIES (Normalized)")
        print("-" * 40)

        if 'pl_rade' in df.columns:
            print(f"  Radius - Mean: {df['pl_rade'].mean():.4f}, Std: {df['pl_rade'].std():.4f}")

        if 'pl_orbper' in df.columns:
            print(f"  Period - Mean: {df['pl_orbper'].mean():.4f}, Std: {df['pl_orbper'].std():.4f}")

        if 'pl_trandep' in df.columns:
            print(f"  Transit Depth - Mean: {df['pl_trandep'].mean():.4f}")

        # Habitable zone
        if 'in_habitable_zone' in df.columns:
            hz_count = df['in_habitable_zone'].sum()
            hz_pct = (hz_count / len(df)) * 100
            print("\n3. HABITABLE ZONE CANDIDATES")
            print("-" * 40)
            print(f"  In Habitable Zone:   {hz_count:,} ({hz_pct:.2f}%)")
            print(f"  Outside HZ:          {len(df) - hz_count:,} ({100-hz_pct:.2f}%)")

        # Stellar magnitude
        if 'st_tmag' in df.columns:
            print("\n4. STELLAR BRIGHTNESS (TESS Magnitude - Normalized)")
            print("-" * 40)
            print(f"  Mean:   {df['st_tmag'].mean():.4f}")
            print(f"  Median: {df['st_tmag'].median():.4f}")

    def correlation_analysis(self):
        """Analyze correlations in datasets"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        # Cumulative correlations
        df = self.datasets['cumulative']
        numeric_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
                       'koi_steff', 'koi_srad']
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) >= 2:
            print("\nCUMULATIVE DATASET - Top Correlations:")
            print("-" * 40)
            corr_matrix = df[available_cols].corr()

            # Get top correlations (excluding diagonal)
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'corr': corr_matrix.iloc[i, j]
                    })

            correlations = sorted(correlations, key=lambda x: abs(x['corr']), reverse=True)

            for item in correlations[:5]:
                print(f"  {item['var1']:15s} <-> {item['var2']:15s}: {item['corr']:6.3f}")

    def dataset_comparison(self):
        """Compare datasets"""
        print("\n" + "="*80)
        print("DATASET COMPARISON")
        print("="*80)

        print(f"\n{'Dataset':<15} {'Rows':>10} {'Columns':>10} {'Confirmed':>12} {'Candidates':>12}")
        print("-" * 65)

        # Cumulative
        df_cum = self.datasets['cumulative']
        confirmed_cum = (df_cum['koi_disposition'] == 'CONFIRMED').sum() if 'koi_disposition' in df_cum.columns else 'N/A'
        candidate_cum = (df_cum['koi_disposition'] == 'CANDIDATE').sum() if 'koi_disposition' in df_cum.columns else 'N/A'
        print(f"{'Cumulative':<15} {len(df_cum):>10,} {df_cum.shape[1]:>10} {str(confirmed_cum):>12} {str(candidate_cum):>12}")

        # K2PANDC
        df_k2 = self.datasets['k2pandc']
        confirmed_k2 = (df_k2['disposition'] == 'CONFIRMED').sum() if 'disposition' in df_k2.columns else 'N/A'
        print(f"{'K2PANDC':<15} {len(df_k2):>10,} {df_k2.shape[1]:>10} {str(confirmed_k2):>12} {'N/A':>12}")

        # TOI
        df_toi = self.datasets['toi']
        confirmed_toi = (df_toi['tfopwg_disp'] == 'CP').sum() if 'tfopwg_disp' in df_toi.columns else 'N/A'
        candidate_toi = (df_toi['tfopwg_disp'] == 'PC').sum() if 'tfopwg_disp' in df_toi.columns else 'N/A'
        print(f"{'TOI':<15} {len(df_toi):>10,} {df_toi.shape[1]:>10} {str(confirmed_toi):>12} {str(candidate_toi):>12}")

    def generate_summary_report(self):
        """Generate comprehensive summary"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)

        total_rows = sum(len(df) for df in self.datasets.values())
        total_cols = sum(df.shape[1] for df in self.datasets.values())

        print(f"\nTotal Records Across All Datasets: {total_rows:,}")
        print(f"Total Features: {total_cols}")

        # Calculate confirmed planets
        cum_confirmed = (self.datasets['cumulative']['koi_disposition'] == 'CONFIRMED').sum() if 'koi_disposition' in self.datasets['cumulative'].columns else 0
        k2_confirmed = (self.datasets['k2pandc']['disposition'] == 'CONFIRMED').sum() if 'disposition' in self.datasets['k2pandc'].columns else 0
        toi_confirmed = (self.datasets['toi']['tfopwg_disp'] == 'CP').sum() if 'tfopwg_disp' in self.datasets['toi'].columns else 0

        print(f"\nConfirmed Planets:")
        print(f"  Cumulative: {cum_confirmed:,}")
        print(f"  K2PANDC:    {k2_confirmed:,}")
        print(f"  TOI:        {toi_confirmed:,}")

        # Habitable zone candidates
        cum_hz = self.datasets['cumulative']['in_habitable_zone'].sum() if 'in_habitable_zone' in self.datasets['cumulative'].columns else 0
        toi_hz = self.datasets['toi']['in_habitable_zone'].sum() if 'in_habitable_zone' in self.datasets['toi'].columns else 0

        print(f"\nHabitable Zone Candidates:")
        print(f"  Cumulative: {cum_hz:,}")
        print(f"  TOI:        {toi_hz:,}")
        print(f"  Total:      {cum_hz + toi_hz:,}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)

if __name__ == "__main__":
    analyzer = DatasetAnalyzer()

    # Load datasets
    analyzer.load_datasets()

    # Basic statistics
    analyzer.basic_statistics()

    # Detailed analysis for each dataset
    analyzer.analyze_cumulative()
    analyzer.analyze_k2pandc()
    analyzer.analyze_toi()

    # Correlation analysis
    analyzer.correlation_analysis()

    # Dataset comparison
    analyzer.dataset_comparison()

    # Summary report
    analyzer.generate_summary_report()
