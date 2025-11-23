"""
Refactored preprocessing module for NASA Exoplanet datasets.
Includes type hints, logging, error handling, and consolidated code.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

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


class ExoplanetPreprocessor:
    """
    Comprehensive preprocessing for NASA Exoplanet datasets.

    Handles missing values, outlier removal, feature engineering,
    categorical encoding, and normalization for multiple datasets.
    """

    def __init__(self):
        """Initialize preprocessor with empty containers for scalers and encoders."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.processed_datasets: Dict[str, pd.DataFrame] = {}

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a specific dataset from CSV file.

        Args:
            dataset_name: Name of dataset ('cumulative', 'k2pandc', or 'toi')

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the dataset file doesn't exist
            ValueError: If dataset_name is invalid
        """
        logger.info(f"Loading {dataset_name} dataset...")

        try:
            file_path = config.get_raw_file_path(dataset_name)

            if not Path(file_path).exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {file_path}\n"
                    f"Please ensure the file exists or update config.py"
                )

            df = pd.read_csv(file_path, comment='#')

            # Validation
            if df.empty:
                raise ValueError(f"Dataset {dataset_name} is empty")

            logger.info(f"✓ Loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            self.datasets[dataset_name] = df
            return df

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {str(e)}")
            raise

    def load_all_datasets(self) -> None:
        """Load all three datasets."""
        logger.info("="*80)
        logger.info("LOADING ALL DATASETS")
        logger.info("="*80)

        for dataset_name in ['cumulative', 'k2pandc', 'toi']:
            try:
                self.load_data(dataset_name)
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {str(e)}")

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values in specified columns.

        Args:
            df: Input DataFrame
            columns: List of column names to process
            strategy: Imputation strategy ('median', 'mean', or 'mode')

        Returns:
            DataFrame with missing values filled
        """
        logger.info("Handling missing values...")

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.debug(f"  Column {col} not found, skipping")
                continue

            missing_before = df[col].isnull().sum()

            if missing_before == 0:
                continue

            try:
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else df[col].median()
                else:
                    logger.warning(f"Unknown strategy {strategy}, using median")
                    fill_value = df[col].median()

                df[col].fillna(fill_value, inplace=True)
                logger.debug(f"  {col}: filled {missing_before} missing values with {strategy}")

            except Exception as e:
                logger.warning(f"  Failed to fill {col}: {str(e)}")

        return df

    def _remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        threshold: float = None
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            df: Input DataFrame
            columns: List of column names to check for outliers
            threshold: IQR multiplier (default from config)

        Returns:
            DataFrame with outliers removed
        """
        if threshold is None:
            threshold = config.PREPROCESSING["IQR_THRESHOLD"]

        logger.info(f"Removing outliers (IQR threshold: {threshold})...")

        df = df.copy()
        initial_rows = len(df)

        for col in columns:
            if col not in df.columns:
                logger.debug(f"  Column {col} not found, skipping")
                continue

            if df[col].notna().sum() == 0:
                logger.debug(f"  Column {col} has no non-null values, skipping")
                continue

            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                before = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before - len(df)

                if removed > 0:
                    logger.debug(f"  {col}: removed {removed} outliers")

            except Exception as e:
                logger.warning(f"  Failed to remove outliers from {col}: {str(e)}")

        total_removed = initial_rows - len(df)
        logger.info(f"  Total rows removed: {total_removed} ({total_removed/initial_rows*100:.1f}%)")

        return df

    def _encode_categorical(
        self,
        df: pd.DataFrame,
        column: str,
        encoder_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Encode a categorical column using LabelEncoder.

        Args:
            df: Input DataFrame
            column: Column name to encode
            encoder_name: Name to store encoder (default: same as column)

        Returns:
            DataFrame with encoded column added
        """
        if column not in df.columns:
            logger.debug(f"  Column {column} not found, skipping encoding")
            return df

        df = df.copy()
        encoder_name = encoder_name or column

        try:
            le = LabelEncoder()
            df[f'{column}_encoded'] = le.fit_transform(df[column].fillna('UNKNOWN'))
            self.encoders[encoder_name] = le

            class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.debug(f"  {column}: {len(le.classes_)} classes - {class_mapping}")

        except Exception as e:
            logger.warning(f"  Failed to encode {column}: {str(e)}")

        return df

    def _create_derived_features(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Create dataset-specific derived features.

        Args:
            df: Input DataFrame
            dataset_name: Name of dataset to determine which features to create

        Returns:
            DataFrame with derived features added
        """
        logger.info("Creating derived features...")

        df = df.copy()
        dataset_config = config.get_dataset_config(dataset_name)

        try:
            if dataset_name == 'cumulative':
                # Planet-star radius ratio
                if 'koi_prad' in df.columns and 'koi_srad' in df.columns:
                    df['planet_star_radius_ratio'] = df['koi_prad'] / (df['koi_srad'] * 109.2)
                    logger.debug("  ✓ Created planet_star_radius_ratio")

                # Planet volume
                if 'koi_prad' in df.columns:
                    df['planet_volume'] = (4/3) * np.pi * (df['koi_prad'] ** 3)
                    logger.debug("  ✓ Created planet_volume")

                # Habitable zone indicator
                if 'koi_teq' in df.columns:
                    temp_range = config.PREPROCESSING["HABITABLE_ZONE_TEMP"]
                    df['in_habitable_zone'] = (
                        (df['koi_teq'] >= temp_range[0]) &
                        (df['koi_teq'] <= temp_range[1])
                    ).astype(int)
                    logger.debug(f"  ✓ Created in_habitable_zone ({df['in_habitable_zone'].sum()} planets)")

            elif dataset_name == 'k2pandc':
                # Planet density
                if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
                    df['planet_density'] = df['pl_masse'] / ((4/3) * np.pi * (df['pl_rade'] ** 3))
                    logger.debug("  ✓ Created planet_density")

                # Surface gravity
                if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
                    df['surface_gravity'] = df['pl_masse'] / (df['pl_rade'] ** 2)
                    logger.debug("  ✓ Created surface_gravity")

                # Escape velocity
                if 'pl_masse' in df.columns and 'pl_rade' in df.columns:
                    df['escape_velocity'] = np.sqrt(df['pl_masse'] / df['pl_rade'])
                    logger.debug("  ✓ Created escape_velocity")

            elif dataset_name == 'toi':
                # Transit SNR approximation
                if 'pl_trandep' in df.columns and 'pl_trandurh' in df.columns:
                    df['transit_snr_approx'] = df['pl_trandep'] * np.sqrt(df['pl_trandurh'])
                    logger.debug("  ✓ Created transit_snr_approx")

                # Distance modulus
                if 'st_dist' in df.columns:
                    # Avoid log of non-positive values
                    df['distance_modulus'] = np.where(
                        df['st_dist'] > 0,
                        5 * np.log10(df['st_dist']) - 5,
                        np.nan
                    )
                    logger.debug("  ✓ Created distance_modulus")

                # Habitable zone indicator
                if 'pl_eqt' in df.columns:
                    temp_range = config.PREPROCESSING["HABITABLE_ZONE_TEMP"]
                    df['in_habitable_zone'] = (
                        (df['pl_eqt'] >= temp_range[0]) &
                        (df['pl_eqt'] <= temp_range[1])
                    ).astype(int)
                    logger.debug(f"  ✓ Created in_habitable_zone ({df['in_habitable_zone'].sum()} candidates)")

        except Exception as e:
            logger.error(f"  Failed to create some derived features: {str(e)}")

        return df

    def _normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        scaler_name: str
    ) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            df: Input DataFrame
            columns: List of columns to normalize
            scaler_name: Name to store the scaler

        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing numerical features...")

        df = df.copy()
        available_cols = [col for col in columns if col in df.columns]

        if not available_cols:
            logger.warning("  No columns available for normalization")
            return df

        try:
            scaler = StandardScaler()
            df[available_cols] = scaler.fit_transform(df[available_cols])
            self.scalers[scaler_name] = scaler
            logger.info(f"  ✓ Normalized {len(available_cols)} columns")

        except Exception as e:
            logger.error(f"  Failed to normalize features: {str(e)}")

        return df

    def preprocess_dataset(
        self,
        dataset_name: str,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generic preprocessing pipeline for any dataset.

        Args:
            dataset_name: Name of dataset ('cumulative', 'k2pandc', or 'toi')
            save: Whether to save the processed dataset

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If dataset hasn't been loaded or dataset_name is invalid
        """
        logger.info("="*80)
        logger.info(f"PREPROCESSING {dataset_name.upper()} DATASET")
        logger.info("="*80)

        if dataset_name not in self.datasets:
            raise ValueError(
                f"Dataset {dataset_name} not loaded. Call load_data() first."
            )

        df = self.datasets[dataset_name].copy()
        dataset_config = config.get_dataset_config(dataset_name)

        # 1. Handle missing values
        all_features = (
            dataset_config["planetary_features"] +
            dataset_config["stellar_features"]
        )
        df = self._handle_missing_values(df, all_features)

        # 2. Remove outliers (only from planetary features)
        outlier_cols = dataset_config["planetary_features"][:3]  # First 3 features
        df = self._remove_outliers(df, outlier_cols)

        # 3. Encode categorical variables
        logger.info("Encoding categorical variables...")

        # Encode target variable
        target_col = dataset_config["target_column"]
        df = self._encode_categorical(df, target_col)

        # Encode other categorical variables
        if dataset_name == 'cumulative':
            if 'koi_pdisposition' in df.columns:
                df = self._encode_categorical(df, 'koi_pdisposition')
        elif dataset_name == 'k2pandc':
            if 'discoverymethod' in df.columns:
                df = self._encode_categorical(df, 'discoverymethod')

        # 4. Create derived features
        df = self._create_derived_features(df, dataset_name)

        # 5. Normalize numerical features
        norm_cols = (
            dataset_config["planetary_features"] +
            dataset_config["stellar_features"]
        )
        df = self._normalize_features(df, norm_cols, dataset_name)

        # Store processed dataset
        self.processed_datasets[dataset_name] = df
        logger.info(f"✓ Final shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Save if requested
        if save:
            self._save_dataset(df, dataset_name)

        return df

    def _save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> None:
        """
        Save processed dataset to CSV.

        Args:
            df: DataFrame to save
            dataset_name: Name of dataset
        """
        try:
            output_path = config.get_processed_file_path(dataset_name)
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Saved {output_path}")

        except Exception as e:
            logger.error(f"Failed to save {dataset_name}: {str(e)}")

    def preprocess_all_datasets(self, save: bool = True) -> None:
        """
        Preprocess all loaded datasets.

        Args:
            save: Whether to save processed datasets
        """
        for dataset_name in self.datasets.keys():
            try:
                self.preprocess_dataset(dataset_name, save=save)
            except Exception as e:
                logger.error(f"Failed to preprocess {dataset_name}: {str(e)}")

    def generate_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate preprocessing summary statistics.

        Returns:
            Dictionary with summary statistics for each dataset
        """
        logger.info("="*80)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*80)

        summary = {}

        for dataset_name in self.datasets.keys():
            if dataset_name not in self.processed_datasets:
                logger.warning(f"{dataset_name} was not processed")
                continue

            original = self.datasets[dataset_name]
            processed = self.processed_datasets[dataset_name]

            stats = {
                'Original shape': original.shape,
                'Processed shape': processed.shape,
                'Rows removed': original.shape[0] - processed.shape[0],
                'Rows removed (%)': (original.shape[0] - processed.shape[0]) / original.shape[0] * 100,
                'Features added': processed.shape[1] - original.shape[1]
            }

            summary[dataset_name] = stats

            logger.info(f"\n{dataset_name.upper()}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")

        return summary


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("NASA EXOPLANET PREPROCESSING PIPELINE")
    logger.info("="*80)

    try:
        # Initialize preprocessor
        preprocessor = ExoplanetPreprocessor()

        # Load all datasets
        preprocessor.load_all_datasets()

        # Preprocess all datasets
        preprocessor.preprocess_all_datasets(save=True)

        # Generate summary
        summary = preprocessor.generate_summary()

        logger.info("="*80)
        logger.info("PREPROCESSING COMPLETE!")
        logger.info("="*80)

        return summary

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
