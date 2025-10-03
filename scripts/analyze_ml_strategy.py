import pandas as pd
import numpy as np

class MLStrategyAnalyzer:
    """Analyze whether to combine datasets or train separately"""

    def __init__(self):
        self.cumulative = pd.read_csv('data/processed/cumulative_preprocessed.csv')
        self.k2pandc = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
        self.toi = pd.read_csv('data/processed/toi_preprocessed.csv')

    def analyze_common_features(self):
        """Analyze feature overlap between datasets"""
        print("="*80)
        print("FEATURE OVERLAP ANALYSIS")
        print("="*80)

        cum_cols = set(self.cumulative.columns)
        k2_cols = set(self.k2pandc.columns)
        toi_cols = set(self.toi.columns)

        print(f"\nDataset Sizes:")
        print(f"  Cumulative: {self.cumulative.shape}")
        print(f"  K2PANDC:    {self.k2pandc.shape}")
        print(f"  TOI:        {self.toi.shape}")

        print(f"\n\nColumn Counts:")
        print(f"  Cumulative: {len(cum_cols)} columns")
        print(f"  K2PANDC:    {len(k2_cols)} columns")
        print(f"  TOI:        {len(toi_cols)} columns")

        # Common columns across all three
        common_all = cum_cols & k2_cols & toi_cols
        print(f"\n\n{'='*80}")
        print(f"COMMON COLUMNS ACROSS ALL 3 DATASETS: {len(common_all)}")
        print(f"{'='*80}")
        if common_all:
            for col in sorted(common_all):
                print(f"  - {col}")

        # Common between pairs
        cum_k2 = cum_cols & k2_cols
        cum_toi = cum_cols & toi_cols
        k2_toi = k2_cols & toi_cols

        print(f"\n\nPairwise Overlaps:")
        print(f"  Cumulative ∩ K2PANDC: {len(cum_k2)} columns")
        print(f"  Cumulative ∩ TOI:     {len(cum_toi)} columns")
        print(f"  K2PANDC ∩ TOI:        {len(k2_toi)} columns")

        return common_all, cum_k2, cum_toi, k2_toi

    def analyze_target_variables(self):
        """Analyze target variable compatibility"""
        print("\n" + "="*80)
        print("TARGET VARIABLE ANALYSIS")
        print("="*80)

        # Cumulative: koi_disposition
        if 'koi_disposition' in self.cumulative.columns:
            print("\nCumulative - koi_disposition:")
            print(self.cumulative['koi_disposition'].value_counts())

        # K2PANDC: disposition
        if 'disposition' in self.k2pandc.columns:
            print("\nK2PANDC - disposition:")
            print(self.k2pandc['disposition'].value_counts())

        # TOI: tfopwg_disp
        if 'tfopwg_disp' in self.toi.columns:
            print("\nTOI - tfopwg_disp:")
            print(self.toi['tfopwg_disp'].value_counts())

        print("\n" + "="*80)
        print("TARGET COMPATIBILITY:")
        print("="*80)
        print("\n✗ INCOMPATIBLE - Different target variables:")
        print("  • Cumulative: koi_disposition (CONFIRMED/CANDIDATE/FALSE POSITIVE)")
        print("  • K2PANDC:    disposition (CONFIRMED/CANDIDATE/FALSE POSITIVE/REFUTED)")
        print("  • TOI:        tfopwg_disp (PC/CP/FP/KP/APC/FA)")
        print("\n  Different naming and categories make direct combination difficult!")

    def identify_key_features(self):
        """Identify key features for ML"""
        print("\n" + "="*80)
        print("KEY FEATURES FOR ML")
        print("="*80)

        # Map similar features across datasets
        feature_mapping = {
            'Orbital Period': {
                'cumulative': 'koi_period',
                'k2pandc': 'pl_orbper',
                'toi': 'pl_orbper'
            },
            'Planet Radius': {
                'cumulative': 'koi_prad',
                'k2pandc': 'pl_rade',
                'toi': 'pl_rade'
            },
            'Transit Depth': {
                'cumulative': 'koi_depth',
                'k2pandc': None,
                'toi': 'pl_trandep'
            },
            'Equilibrium Temperature': {
                'cumulative': 'koi_teq',
                'k2pandc': 'pl_eqt',
                'toi': 'pl_eqt'
            },
            'Stellar Temperature': {
                'cumulative': 'koi_steff',
                'k2pandc': 'st_teff',
                'toi': 'st_teff'
            },
            'Stellar Radius': {
                'cumulative': 'koi_srad',
                'k2pandc': 'st_rad',
                'toi': 'st_rad'
            },
            'Stellar Mass': {
                'cumulative': 'koi_smass',
                'k2pandc': 'st_mass',
                'toi': None
            }
        }

        print("\nFeature Name Mapping:")
        print("-" * 80)
        print(f"{'Feature':<30} {'Cumulative':<20} {'K2PANDC':<20} {'TOI':<20}")
        print("-" * 80)

        for feature, datasets in feature_mapping.items():
            cum = datasets['cumulative'] if datasets['cumulative'] else '-'
            k2 = datasets['k2pandc'] if datasets['k2pandc'] else '-'
            toi_val = datasets['toi'] if datasets['toi'] else '-'
            print(f"{feature:<30} {cum:<20} {k2:<20} {toi_val:<20}")

        return feature_mapping

    def provide_recommendations(self):
        """Provide ML strategy recommendations"""
        print("\n" + "="*80)
        print("ML STRATEGY RECOMMENDATIONS")
        print("="*80)

        print("\n" + "="*40)
        print("OPTION 1: TRAIN SEPARATELY (RECOMMENDED)")
        print("="*40)

        print("\n✅ PROS:")
        print("  1. Preserve dataset-specific features")
        print("     • Cumulative: 146 features (Kepler-specific)")
        print("     • K2PANDC: 300 features (multi-mission data)")
        print("     • TOI: 91 features (TESS-specific)")
        print("\n  2. Different missions, different instruments")
        print("     • Each has unique measurement characteristics")
        print("     • Kepler: Long-term monitoring")
        print("     • K2: Ecliptic plane survey")
        print("     • TESS: All-sky survey")
        print("\n  3. Different target variables")
        print("     • No need to force compatibility")
        print("     • Each model optimized for its task")
        print("\n  4. Better model interpretability")
        print("     • Features match original mission parameters")
        print("     • Easier to explain to astronomers")

        print("\n❌ CONS:")
        print("  1. Cannot leverage combined data size (16,641 total)")
        print("  2. Need to maintain 3 separate models")
        print("  3. Cannot transfer learning across datasets")

        print("\n\n" + "="*40)
        print("OPTION 2: COMBINE WITH FEATURE MAPPING")
        print("="*40)

        print("\n✅ PROS:")
        print("  1. Larger training set (16,641 samples)")
        print("  2. Single unified model")
        print("  3. Can learn general exoplanet patterns")
        print("  4. Better generalization across missions")

        print("\n❌ CONS:")
        print("  1. Feature loss - only ~5-10 common features")
        print("     • Lose 90%+ of dataset-specific features")
        print("  2. Target variable mismatch")
        print("     • Need to harmonize labels (CONFIRMED vs CP vs PC)")
        print("  3. Different measurement scales/errors")
        print("     • Combining normalized data from different sources")
        print("  4. Mission-specific biases mixed together")

        print("\n\n" + "="*80)
        print("FINAL RECOMMENDATION")
        print("="*80)

        print("\n🎯 RECOMMENDED APPROACH: HYBRID STRATEGY")
        print("\n1. TRAIN 3 SEPARATE MODELS (Primary)")
        print("   • One for each dataset")
        print("   • Use all available features")
        print("   • Optimized for each mission's characteristics")
        print("\n   Models:")
        print("   ├─ Model 1: Cumulative (Kepler) → Binary: CONFIRMED vs FALSE POSITIVE")
        print("   ├─ Model 2: K2PANDC → Multi-class: CONFIRMED/CANDIDATE/FALSE POSITIVE")
        print("   └─ Model 3: TOI (TESS) → Multi-class: CP/PC/FP/KP/etc.")

        print("\n2. CREATE COMBINED MODEL (Secondary/Experimental)")
        print("   • Use only common features:")
        print("     - Orbital period")
        print("     - Planet radius")
        print("     - Stellar temperature")
        print("     - Stellar radius")
        print("     - Equilibrium temperature")
        print("   • Unified target: Binary (CONFIRMED vs NOT CONFIRMED)")
        print("   • Good for: General exoplanet classification")

        print("\n3. ENSEMBLE APPROACH (Advanced)")
        print("   • Train all 4 models (3 separate + 1 combined)")
        print("   • Use voting/averaging for final prediction")
        print("   • Best of both worlds")

        print("\n\n" + "="*80)
        print("SUGGESTED WORKFLOW")
        print("="*80)

        print("\n📋 Step-by-step:")
        print("\n1. Start with SEPARATE MODELS:")
        print("   ✓ Easiest to implement")
        print("   ✓ Best use of available features")
        print("   ✓ Can compare performance across missions")

        print("\n2. Evaluate performance:")
        print("   • Cumulative: Likely highest accuracy (most confirmed)")
        print("   • K2PANDC: Good for confirmed planets")
        print("   • TOI: Most candidates, harder task")

        print("\n3. If needed, create COMBINED model:")
        print("   • Map common features")
        print("   • Harmonize target labels")
        print("   • Compare against separate models")

        print("\n4. Production deployment:")
        print("   • Use SEPARATE models for mission-specific data")
        print("   • Use COMBINED model for cross-mission validation")

        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n🏆 WINNER: Train 3 Separate Models")
        print("\nReasons:")
        print("  • Different feature sets (146 vs 300 vs 91 columns)")
        print("  • Different target variables")
        print("  • Different mission characteristics")
        print("  • Better performance with full feature sets")
        print("  • Easier to interpret and validate")

        print("\n💡 Bonus: Create a simple combined model as baseline")
        print("   for comparison and cross-validation")

        print("\n" + "="*80)

def main():
    analyzer = MLStrategyAnalyzer()

    # Analyze feature overlap
    analyzer.analyze_common_features()

    # Analyze target variables
    analyzer.analyze_target_variables()

    # Identify key features
    analyzer.identify_key_features()

    # Provide recommendations
    analyzer.provide_recommendations()

if __name__ == "__main__":
    main()
