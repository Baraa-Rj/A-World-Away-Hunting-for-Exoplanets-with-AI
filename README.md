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
