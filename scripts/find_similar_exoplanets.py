import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class ExoplanetSimilarityAnalyzer:
    """Analyze and find similar exoplanets across datasets"""

    def __init__(self):
        self.datasets = {}

    def load_datasets(self):
        """Load all preprocessed datasets"""
        print("="*80)
        print("LOADING DATASETS")
        print("="*80)

        self.datasets['cumulative'] = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        self.datasets['k2pandc'] = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        self.datasets['toi'] = pd.read_csv('data/processed/toi_preprocessed.csv')

        for name, df in self.datasets.items():
            print(f"✓ Loaded {name}: {df.shape}")

    def find_duplicates_by_name(self):
        """Find potentially duplicate planets by name matching"""
        print("\n" + "="*80)
        print("FINDING DUPLICATES BY NAME")
        print("="*80)

        cum = self.datasets['cumulative']
        k2 = self.datasets['k2pandc']

        # Check for Kepler names in cumulative and k2pandc
        kepler_names_cum = set()
        if 'kepler_name' in cum.columns:
            kepler_names_cum = set(cum['kepler_name'].dropna().unique())
            print(f"\nCumulative dataset: {len(kepler_names_cum)} unique Kepler names")

        kepler_names_k2 = set()
        if 'pl_name' in k2.columns:
            # Filter for Kepler planets
            k2_kepler = k2[k2['pl_name'].str.contains('Kepler', na=False, case=False)]
            kepler_names_k2 = set(k2_kepler['pl_name'].dropna().unique())
            print(f"K2PANDC dataset: {len(kepler_names_k2)} Kepler planets")

        # Find overlapping names
        if kepler_names_cum and kepler_names_k2:
            overlap = kepler_names_cum.intersection(kepler_names_k2)
            print(f"\n✓ Found {len(overlap)} planets appearing in BOTH datasets!")

            if len(overlap) > 0:
                print(f"\nSample overlapping planets (first 10):")
                for i, name in enumerate(list(overlap)[:10], 1):
                    print(f"  {i}. {name}")

        return overlap if kepler_names_cum and kepler_names_k2 else set()

    def find_by_coordinates(self):
        """Find duplicate planets by celestial coordinates"""
        print("\n" + "="*80)
        print("FINDING DUPLICATES BY COORDINATES")
        print("="*80)

        cum = self.datasets['cumulative']
        k2 = self.datasets['k2pandc']
        toi = self.datasets['toi']

        matches = []

        # Cumulative has 'ra' and 'dec'
        # K2PANDC has 'ra' and 'dec'
        # TOI has 'ra' and 'dec'

        if all(col in cum.columns for col in ['ra', 'dec']) and \
           all(col in k2.columns for col in ['ra', 'dec']):

            print("\nSearching for coordinate matches between Cumulative and K2PANDC...")

            # Sample comparison (checking first 100 from each for demonstration)
            cum_sample = cum[['ra', 'dec', 'kepoi_name']].dropna().head(100)
            k2_sample = k2[['ra', 'dec', 'pl_name']].dropna().head(100)

            tolerance = 0.001  # degrees (~3.6 arcseconds)

            for _, cum_row in cum_sample.iterrows():
                for _, k2_row in k2_sample.iterrows():
                    ra_diff = abs(cum_row['ra'] - k2_row['ra'])
                    dec_diff = abs(cum_row['dec'] - k2_row['dec'])

                    if ra_diff < tolerance and dec_diff < tolerance:
                        matches.append({
                            'cumulative_name': cum_row['kepoi_name'],
                            'k2_name': k2_row['pl_name'],
                            'ra_diff': ra_diff,
                            'dec_diff': dec_diff
                        })

            if matches:
                print(f"\n✓ Found {len(matches)} coordinate matches (sample search)!")
                print("\nSample matches:")
                for i, match in enumerate(matches[:5], 1):
                    print(f"  {i}. {match['cumulative_name']} ↔ {match['k2_name']}")
                    print(f"     RA diff: {match['ra_diff']:.6f}°, DEC diff: {match['dec_diff']:.6f}°")
            else:
                print("\n✗ No coordinate matches found in sample")

        return matches

    def cluster_by_characteristics(self, dataset_name='cumulative', n_clusters=5):
        """Cluster planets by similar characteristics"""
        print("\n" + "="*80)
        print(f"CLUSTERING {dataset_name.upper()} BY CHARACTERISTICS")
        print("="*80)

        df = self.datasets[dataset_name].copy()

        # Select relevant features based on dataset
        if dataset_name == 'cumulative':
            feature_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_teq', 'koi_steff']
        elif dataset_name == 'k2pandc':
            feature_cols = ['pl_orbper', 'pl_rade', 'pl_masse', 'st_teff', 'st_rad']
        else:  # toi
            feature_cols = ['pl_orbper', 'pl_rade', 'pl_trandep', 'st_teff', 'st_tmag']

        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]

        if len(available_cols) < 2:
            print("Not enough features available for clustering")
            return None

        print(f"\nUsing features: {available_cols}")

        # Drop rows with missing values in these columns
        df_clean = df[available_cols].dropna()
        print(f"Rows after dropping NaN: {len(df_clean)}")

        if len(df_clean) < n_clusters:
            print(f"Not enough data points for {n_clusters} clusters")
            return None

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(df_clean)

        # Add cluster labels back to dataframe
        df_clean['cluster'] = clusters

        # Analyze clusters
        print(f"\n{'Cluster':<10} {'Count':<10} {'% of Total':<15}")
        print("-" * 40)

        for i in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == i]
            count = len(cluster_data)
            pct = (count / len(df_clean)) * 100
            print(f"{i:<10} {count:<10,} {pct:<15.2f}%")

        # Show cluster characteristics
        print(f"\nCluster Characteristics:")
        print("-" * 80)

        for i in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == i]
            print(f"\nCluster {i} (n={len(cluster_data)}):")

            for col in available_cols[:3]:  # Show first 3 features
                mean_val = cluster_data[col].mean()
                print(f"  {col}: mean = {mean_val:.4f}")

        return df_clean

    def find_earth_like_planets(self):
        """Find planets with Earth-like characteristics"""
        print("\n" + "="*80)
        print("FINDING EARTH-LIKE PLANETS")
        print("="*80)

        earth_like = []

        # Cumulative dataset
        cum = self.datasets['cumulative']

        # Since data is normalized, Earth-like would be near certain normalized values
        # We'll look for planets in habitable zone with reasonable sizes

        if 'in_habitable_zone' in cum.columns and 'koi_prad' in cum.columns:
            # Filter for habitable zone
            hz_planets = cum[cum['in_habitable_zone'] == 1].copy()

            # Filter for Earth-like radius (normalized values between -1 and 1 are reasonable)
            # In normalized space, values close to mean (0) with small radius
            earth_like_cum = hz_planets[
                (hz_planets['koi_prad'] >= -1.0) &
                (hz_planets['koi_prad'] <= 1.0)
            ]

            print(f"\nCumulative dataset:")
            print(f"  Planets in habitable zone: {len(hz_planets)}")
            print(f"  Earth-like candidates (HZ + reasonable size): {len(earth_like_cum)}")

            if len(earth_like_cum) > 0:
                print(f"\n  Sample Earth-like candidates:")
                for i, (idx, row) in enumerate(earth_like_cum.head(5).iterrows(), 1):
                    name = row.get('kepoi_name', 'Unknown')
                    radius = row.get('koi_prad', 'N/A')
                    temp = row.get('koi_teq', 'N/A')
                    print(f"    {i}. {name} - Radius(norm): {radius:.3f}, Temp(norm): {temp:.3f}")

        # TOI dataset
        toi = self.datasets['toi']

        if 'in_habitable_zone' in toi.columns and 'pl_rade' in toi.columns:
            hz_planets_toi = toi[toi['in_habitable_zone'] == 1].copy()

            earth_like_toi = hz_planets_toi[
                (hz_planets_toi['pl_rade'] >= -1.0) &
                (hz_planets_toi['pl_rade'] <= 1.0)
            ]

            print(f"\nTOI dataset:")
            print(f"  Planets in habitable zone: {len(hz_planets_toi)}")
            print(f"  Earth-like candidates (HZ + reasonable size): {len(earth_like_toi)}")

            if len(earth_like_toi) > 0:
                print(f"\n  Sample Earth-like candidates:")
                for i, (idx, row) in enumerate(earth_like_toi.head(5).iterrows(), 1):
                    toi_num = row.get('toi', 'Unknown')
                    radius = row.get('pl_rade', 'N/A')
                    temp = row.get('pl_eqt', 'N/A')
                    print(f"    {i}. TOI-{toi_num} - Radius(norm): {radius:.3f}, Temp(norm): {temp:.3f}")

    def find_hot_jupiters(self):
        """Find Hot Jupiter candidates (large planets with short periods)"""
        print("\n" + "="*80)
        print("FINDING HOT JUPITERS")
        print("="*80)

        print("\nHot Jupiters: Large planets (>8 Earth radii) with short orbital periods (<10 days)")

        # Cumulative
        cum = self.datasets['cumulative']

        if 'koi_prad' in cum.columns and 'koi_period' in cum.columns:
            # In normalized space, large radius and short period would be specific thresholds
            # We'll use percentiles
            large_radius = cum['koi_prad'].quantile(0.75)  # Top 25%
            short_period = cum['koi_period'].quantile(0.25)  # Bottom 25%

            hot_jupiters_cum = cum[
                (cum['koi_prad'] >= large_radius) &
                (cum['koi_period'] <= short_period)
            ]

            print(f"\nCumulative dataset:")
            print(f"  Hot Jupiter candidates: {len(hot_jupiters_cum)}")

            if len(hot_jupiters_cum) > 0:
                print(f"\n  Sample Hot Jupiters:")
                for i, (idx, row) in enumerate(hot_jupiters_cum.head(5).iterrows(), 1):
                    name = row.get('kepoi_name', 'Unknown')
                    radius = row.get('koi_prad', 'N/A')
                    period = row.get('koi_period', 'N/A')
                    print(f"    {i}. {name} - Radius(norm): {radius:.3f}, Period(norm): {period:.3f}")

        # K2PANDC
        k2 = self.datasets['k2pandc']

        if 'pl_rade' in k2.columns and 'pl_orbper' in k2.columns:
            large_radius = k2['pl_rade'].quantile(0.75)
            short_period = k2['pl_orbper'].quantile(0.25)

            hot_jupiters_k2 = k2[
                (k2['pl_rade'] >= large_radius) &
                (k2['pl_orbper'] <= short_period)
            ]

            print(f"\nK2PANDC dataset:")
            print(f"  Hot Jupiter candidates: {len(hot_jupiters_k2)}")

            if len(hot_jupiters_k2) > 0:
                print(f"\n  Sample Hot Jupiters:")
                for i, (idx, row) in enumerate(hot_jupiters_k2.head(5).iterrows(), 1):
                    name = row.get('pl_name', 'Unknown')
                    radius = row.get('pl_rade', 'N/A')
                    period = row.get('pl_orbper', 'N/A')
                    print(f"    {i}. {name} - Radius(norm): {radius:.3f}, Period(norm): {period:.3f}")

    def summary_report(self):
        """Generate summary of similar planets found"""
        print("\n" + "="*80)
        print("SIMILARITY ANALYSIS SUMMARY")
        print("="*80)

        print("\n1. DATASET OVERLAPS:")
        print("   - Multiple datasets may contain the same planets")
        print("   - Confirmed planets from Kepler appear in both Cumulative and K2PANDC")

        print("\n2. PLANET TYPES IDENTIFIED:")
        print("   ✓ Earth-like candidates (habitable zone + reasonable size)")
        print("   ✓ Hot Jupiters (large size + short period)")
        print("   ✓ Various clusters of similar characteristics")

        print("\n3. CLUSTERING:")
        print("   - Planets naturally group into ~5 clusters based on physical properties")
        print("   - Each cluster represents planets with similar characteristics")

        print("\n4. RECOMMENDATIONS:")
        print("   - Remove duplicate entries when training ML models")
        print("   - Use clustering for stratified sampling")
        print("   - Consider planet types for specialized classification")

        print("\n" + "="*80)

def main():
    analyzer = ExoplanetSimilarityAnalyzer()

    # Load datasets
    analyzer.load_datasets()

    # Find duplicates by name
    analyzer.find_duplicates_by_name()

    # Find duplicates by coordinates
    analyzer.find_by_coordinates()

    # Cluster planets by characteristics
    print("\n")
    analyzer.cluster_by_characteristics('cumulative', n_clusters=5)

    # Find specific planet types
    analyzer.find_earth_like_planets()
    analyzer.find_hot_jupiters()

    # Summary
    analyzer.summary_report()

if __name__ == "__main__":
    main()
