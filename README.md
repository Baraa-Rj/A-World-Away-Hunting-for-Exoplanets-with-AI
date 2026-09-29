# NASA Exoplanet Machine Learning Analysis

A comprehensive machine learning project for analyzing and classifying exoplanet data from NASA's Exoplanet Archive, including data from Kepler, K2, and TESS missions.

## Overview

This project preprocesses and analyzes exoplanet data from three major NASA missions:
- **Kepler Mission**: Cumulative KOI (Kepler Objects of Interest) dataset
- **K2 Mission**: Confirmed planets and candidates
- **TESS Mission**: TOI (TESS Objects of Interest) dataset

## Project Structure

```
ML for nasa/
│
├── data/
│   ├── raw/                    # Raw CSV files from NASA Exoplanet Archive
│   │   ├── cumulative_*.csv
│   │   ├── k2pandc_*.csv
│   │   └── TOI_*.csv
│   └── processed/              # Preprocessed datasets
│       ├── cumulative_preprocessed.csv
│       ├── k2pandc_preprocessed.csv
│       └── toi_preprocessed.csv
│
├── scripts/
│   ├── analyze_data.py         # Data exploration and analysis
│   └── preprocess_data.py      # Data preprocessing pipeline
│
├── notebooks/                  # Jupyter notebooks for analysis
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset Information

### 1. Cumulative (Kepler KOI)
- **Original**: 9,564 rows × 141 columns
- **Processed**: 6,484 rows × 146 columns
- **Key Features**: Orbital period, planet radius, transit depth, stellar parameters

### 2. K2PANDC (Confirmed Planets)
- **Original**: 4,004 rows × 295 columns
- **Processed**: 3,047 rows × 300 columns
- **Key Features**: Planet mass, radius, orbital parameters, discovery method

### 3. TOI (TESS)
- **Original**: 7,703 rows × 87 columns
- **Processed**: 7,110 rows × 91 columns
- **Key Features**: Transit parameters, stellar magnitudes, candidate disposition

## Preprocessing Pipeline

The preprocessing pipeline includes:

### 1. Missing Value Imputation
- Median imputation for numerical columns
- Critical planet and stellar parameters filled

### 2. Outlier Removal
- IQR method (3× IQR threshold)
- Applied to orbital period, planet radius, and transit depth

### 3. Feature Engineering
Created derived features:
- **Planet-star radius ratio**: Relative size comparison
- **Planet density**: Mass-to-volume ratio
- **Surface gravity**: Gravitational acceleration
- **Escape velocity**: Minimum velocity to escape planet
- **Habitable zone indicator**: Temperature-based habitability
- **Transit SNR**: Signal-to-noise approximation

### 4. Categorical Encoding
- Label encoding for dispositions (CONFIRMED, CANDIDATE, FALSE POSITIVE)
- Discovery method encoding

### 5. Normalization
- StandardScaler applied to all numerical features
- Ensures consistent scale across features

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML-for-nasa.git
cd ML-for-nasa
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Data
Download the latest datasets from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):
- Cumulative KOI table
- K2 Planets and Candidates
- TESS TOI table

Place the CSV files in `data/raw/`

### 2. Run Data Analysis
```bash
python scripts/analyze_data.py
```

### 3. Preprocess Data
```bash
python scripts/preprocess_data.py
```

This will:
- Load raw datasets
- Apply preprocessing pipeline
- Save processed data to `data/processed/`
- Generate summary statistics

### 4. Train and Evaluate Models
```bash
python scripts/train_models_refactored.py   # trains RF, XGBoost, LightGBM per dataset
python scripts/test_models_refactored.py    # evaluates them on the held-out test split
```

Training saves the models, label encoders and the held-out test split to
`models/models_fixed/`. The evaluation script exits with a non-zero status if
any of these are missing.

### 5. Run the Tests
```bash
pip install pytest
python -m pytest tests
```

## Model Results

Held-out test split (15% of each cleaned dataset, stratified, `random_state=42`),
weighted precision/recall/F1. SMOTE is applied only to training folds, and
label-derived columns (`*_disposition_encoded`, `tfopwg_disp_encoded`) are
excluded from the features.

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---------|-------|----------|-----------|--------|----|
| Cumulative (Kepler) | Random Forest | 72.35% | 72.91% | 72.35% | 72.58% |
| Cumulative (Kepler) | XGBoost | 73.48% | 72.02% | 73.48% | 72.39% |
| Cumulative (Kepler) | LightGBM | 72.15% | 70.67% | 72.15% | 71.02% |
| K2 | Random Forest | 73.58% | 73.43% | 73.58% | 73.06% |
| K2 | XGBoost | 81.88% | 81.69% | 81.88% | 81.52% |
| K2 | LightGBM | 78.60% | 78.05% | 78.60% | 78.13% |
| TOI (TESS) | Random Forest | 61.11% | 66.46% | 61.11% | 63.08% |
| TOI (TESS) | XGBoost | 63.82% | 66.85% | 63.82% | 64.90% |
| TOI (TESS) | LightGBM | 64.76% | 65.56% | 64.76% | 64.93% |

For reference, always predicting the most common class gives about 39%
(Cumulative), 58% (K2) and 61% (TOI) accuracy.

## Key Features

### Preprocessing Results

| Dataset | Rows Removed | Features Added | Missing Values Filled |
|---------|--------------|----------------|----------------------|
| Cumulative | 3,080 (32%) | 5 | 3,993 |
| K2PANDC | 957 (24%) | 5 | 20,511 |
| TOI | 593 (8%) | 4 | 2,839 |

### Engineered Features

1. **Geometric Features**
   - Planet-star radius ratio
   - Planet volume
   - Transit signal-to-noise ratio

2. **Physical Features**
   - Planet density
   - Surface gravity
   - Escape velocity

3. **Habitability Features**
   - Habitable zone indicator (200K - 350K equilibrium temperature)
   - Stellar distance modulus

## Data Sources

All data is sourced from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/):
- Kepler Mission: https://www.nasa.gov/mission_pages/kepler/main/index.html
- K2 Mission: https://www.nasa.gov/mission_pages/kepler/main/index.html
- TESS Mission: https://www.nasa.gov/tess-transiting-exoplanet-survey-satellite

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NASA Exoplanet Archive for providing comprehensive exoplanet data
- Kepler, K2, and TESS mission teams
- The exoplanet research community

## Citation

If you use this code or data in your research, please cite:
```
NASA Exoplanet Archive
https://exoplanetarchive.ipac.caltech.edu/
```

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Raw data files are not included in this repository due to size constraints. Please download them directly from the NASA Exoplanet Archive.
