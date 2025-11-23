"""
Refactored dataset cleaning with type hints and logging.
Creates cleaned datasets with only important features for ML training.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING["LEVEL"]),
    format=config.LOGGING["FORMAT"],
    datefmt=config.LOGGING["DATE_FORMAT"],
    handlers=[
        logging.FileHandler(config.LOGGING["LOG_FILE"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatasetCleaner:
    """Create cleaned datasets with only important features for ML training."""

    # Define important features for each dataset
    FEATURE_SETS = {
        'cumulative': {
            'target': 'koi_disposition',
            'planetary': [
                'koi_period', 'koi_teq', 'koi_insol', 'koi_impact',
                'koi_duration', 'koi_prad', 'koi_depth'
            ],
            'stellar': [
                'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass', 'koi_smet'
            ],
            'derived': [
                'planet_star_radius_ratio', 'planet_volume', 'in_habitable_zone'
            ],
            'encoded': ['koi_disposition_encoded', 'koi_pdisposition_encoded'],
            'coordinates': ['ra', 'dec'],
            'critical': ['koi_disposition', 'koi_period', 'koi_prad', 'koi_depth']
        },
        'k2pandc': {
            'target': 'disposition',
            'planetary': [
                'pl_orbper', 'pl_rade', 'pl_masse', 'pl_orbeccen',
                'pl_insol', 'pl_eqt', 'pl_orbincl'
            ],
            'stellar': [
                'st_teff', 'st_rad', 'st_mass', 'st_logg', 'st_met'
            ],
            'derived': [
                'planet_density', 'surface_gravity', 'escape_velocity'
            ],
            'encoded': ['disposition_encoded', 'discoverymethod_encoded'],
            'categorical': ['discoverymethod'],
            'coordinates': ['ra', 'dec'],
            'critical': ['disposition', 'pl_orbper', 'pl_rade']
        },
        'toi': {
            'target': 'tfopwg_disp',
            'planetary': [
                'pl_orbper', 'pl_rade', 'pl_trandurh', 'pl_trandep',
                'pl_insol', 'pl_eqt'
            ],
            'stellar': [
                'st_teff', 'st_rad', 'st_logg', 'st_tmag', 'st_dist'
            ],
            'derived': [
                'transit_snr_approx', 'distance_modulus', 'in_habitable_zone'
            ],
            'encoded': ['tfopwg_disp_encoded'],
            'coordinates': ['ra', 'dec'],
            'additional': ['pl_tranmid', 'st_pmra', 'st_pmdec'],
            'critical': ['tfopwg_disp', 'pl_orbper', 'pl_rade', 'st_tmag']
        }
    }

    def __init__(self):
        """Initialize the dataset cleaner."""
        self.cleaned_datasets: Dict[str, pd.DataFrame] = {}

    def clean_dataset(
        self,
        dataset_name: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Clean a specific dataset by selecting important features.

        Args:
            dataset_name: Name of dataset ('cumulative', 'k2pandc', or 'toi')
            save: Whether to save the cleaned dataset

        Returns:
            Cleaned DataFrame

        Raises:
            ValueError: If dataset_name is invalid
            FileNotFoundError: If preprocessed file doesn't exist
        """
        logger.info("="*80)
        logger.info(f"CLEANING {dataset_name.upper()} DATASET")
        logger.info("="*80)

        if dataset_name not in self.FEATURE_SETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        try:
            # Load preprocessed data
            file_path = config.get_processed_file_path(dataset_name)

            if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"Preprocessed file not found: {file_path}\n"
                    f"Please run preprocessing first."
                )

            df = pd.read_csv(file_path)
            logger.info(f"\nOriginal shape: {df.shape[0]} rows, {df.shape[1]} columns")

            # Get feature set for this dataset
            feature_set = self.FEATURE_SETS[dataset_name]

            # Collect all important features
            important_features = []

            # Add features from all categories
            for category in ['target', 'planetary', 'stellar', 'derived',
                           'encoded', 'categorical', 'coordinates', 'additional']:
                if category == 'target':
                    if feature_set.get('target'):
                        important_features.append(feature_set['target'])
                else:
                    important_features.extend(feature_set.get(category, []))

            # Filter only available columns
            available_features = [col for col in important_features if col in df.columns]
            logger.info(f"Selected features: {len(available_features)}")

            # Create cleaned dataset
            df_clean = df[available_features].copy()

            # Remove rows with missing values in critical features
            critical_features = feature_set.get('critical', [])
            critical_available = [col for col in critical_features if col in df_clean.columns]

            if critical_available:
                rows_before = len(df_clean)
                df_clean = df_clean.dropna(subset=critical_available)
                rows_removed = rows_before - len(df_clean)
                logger.info(f"Rows removed due to missing critical values: {rows_removed}")

            logger.info(f"Final shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
            logger.info(f"Features removed: {df.shape[1] - df_clean.shape[1]}")
            logger.info(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

            # Display selected features
            logger.info(f"\n✅ SELECTED FEATURES ({len(df_clean.columns)}):")
            for i, col in enumerate(df_clean.columns, 1):
                missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
                logger.info(f"  {i:2d}. {col:35s} - Missing: {missing_pct:5.1f}%")

            # Display target distribution
            target_col = feature_set.get('target')
            if target_col and target_col in df_clean.columns:
                logger.info(f"\n📊 TARGET DISTRIBUTION:")
                value_counts = df_clean[target_col].value_counts()
                for label, count in value_counts.items():
                    pct = (count / len(df_clean)) * 100
                    logger.info(f"  {label}: {count} ({pct:.1f}%)")

            # Save if requested
            if save:
                output_path = config.get_cleaned_file_path(dataset_name)
                df_clean.to_csv(output_path, index=False)
                logger.info(f"\n💾 Saved to: {output_path}")

            # Store cleaned dataset
            self.cleaned_datasets[dataset_name] = df_clean

            return df_clean

        except Exception as e:
            logger.error(f"Failed to clean {dataset_name}: {str(e)}")
            raise

    def clean_all_datasets(self, save: bool = True) -> None:
        """
        Clean all datasets.

        Args:
            save: Whether to save cleaned datasets
        """
        for dataset_name in ['cumulative', 'k2pandc', 'toi']:
            try:
                self.clean_dataset(dataset_name, save=save)
            except Exception as e:
                logger.error(f"Failed to clean {dataset_name}: {str(e)}")

    def generate_summary_report(self) -> None:
        """Generate summary report of cleaning process."""
        logger.info("\n" + "="*80)
        logger.info("CLEANING SUMMARY REPORT")
        logger.info("="*80)

        if not self.cleaned_datasets:
            logger.warning("No datasets have been cleaned yet!")
            return

        # Load original preprocessed datasets for comparison
        summary_data = []

        for dataset_name in self.cleaned_datasets.keys():
            try:
                # Load original
                orig_path = config.get_processed_file_path(dataset_name)
                orig = pd.read_csv(orig_path)

                # Get cleaned
                cleaned = self.cleaned_datasets[dataset_name]

                summary_data.append({
                    'Dataset': dataset_name.upper(),
                    'Original Rows': orig.shape[0],
                    'Original Cols': orig.shape[1],
                    'Cleaned Rows': cleaned.shape[0],
                    'Cleaned Cols': cleaned.shape[1],
                    'Rows Removed': orig.shape[0] - cleaned.shape[0],
                    'Cols Removed': orig.shape[1] - cleaned.shape[1],
                    'Row Retention %': (cleaned.shape[0] / orig.shape[0]) * 100,
                    'Col Retention %': (cleaned.shape[1] / orig.shape[1]) * 100
                })

            except Exception as e:
                logger.error(f"Error generating summary for {dataset_name}: {str(e)}")

        if not summary_data:
            return

        # Display summary tables
        logger.info(f"\n{'Dataset':<15} {'Orig Rows':>12} {'Clean Rows':>12} {'Rows Removed':>15} {'Retention':>12}")
        logger.info("-" * 80)
        for item in summary_data:
            logger.info(
                f"{item['Dataset']:<15} {item['Original Rows']:>12,} {item['Cleaned Rows']:>12,} "
                f"{item['Rows Removed']:>15,} {item['Row Retention %']:>11.1f}%"
            )

        logger.info(f"\n{'Dataset':<15} {'Orig Cols':>12} {'Clean Cols':>12} {'Cols Removed':>15} {'Retention':>12}")
        logger.info("-" * 80)
        for item in summary_data:
            logger.info(
                f"{item['Dataset']:<15} {item['Original Cols']:>12,} {item['Cleaned Cols']:>12,} "
                f"{item['Cols Removed']:>15,} {item['Col Retention %']:>11.1f}%"
            )

        # Total statistics
        logger.info("\n" + "="*80)
        logger.info("TOTAL STATISTICS")
        logger.info("="*80)

        total_orig_rows = sum(item['Original Rows'] for item in summary_data)
        total_clean_rows = sum(item['Cleaned Rows'] for item in summary_data)
        total_orig_cols = sum(item['Original Cols'] for item in summary_data)
        total_clean_cols = sum(item['Cleaned Cols'] for item in summary_data)

        logger.info(f"\nTotal Records:")
        logger.info(f"  Original:  {total_orig_rows:,}")
        logger.info(f"  Cleaned:   {total_clean_rows:,}")
        logger.info(f"  Removed:   {total_orig_rows - total_clean_rows:,}")
        logger.info(f"  Retention: {(total_clean_rows / total_orig_rows) * 100:.1f}%")

        logger.info(f"\nTotal Features:")
        logger.info(f"  Original:  {total_orig_cols:,}")
        logger.info(f"  Cleaned:   {total_clean_cols:,}")
        logger.info(f"  Removed:   {total_orig_cols - total_clean_cols:,}")
        logger.info(f"  Retention: {(total_clean_cols / total_orig_cols) * 100:.1f}%")

        logger.info("\n" + "="*80)
        logger.info("✅ CLEANED DATASETS READY FOR ML TRAINING!")
        logger.info("="*80)

        logger.info("\nOutput Files:")
        for dataset_name in self.cleaned_datasets.keys():
            output_path = config.get_cleaned_file_path(dataset_name)
            logger.info(f"  📁 {output_path}")

        logger.info("\n" + "="*80)


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("NASA EXOPLANET DATASET CLEANING PIPELINE")
    logger.info("="*80)

    cleaner = DatasetCleaner()

    # Clean all datasets
    cleaner.clean_all_datasets(save=True)

    # Generate summary report
    cleaner.generate_summary_report()

    logger.info("\n✅ CLEANING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
