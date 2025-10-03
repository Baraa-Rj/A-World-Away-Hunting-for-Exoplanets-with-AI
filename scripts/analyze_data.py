import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the three datasets
print("Loading datasets...")
cumulative = pd.read_csv('cumulative_2025.10.03_00.50.03.csv', comment='#')
k2pandc = pd.read_csv('k2pandc_2025.10.03_00.53.03.csv', comment='#')
toi = pd.read_csv('TOI_2025.10.03_00.51.26.csv', comment='#')

print("\n" + "="*80)
print("CUMULATIVE DATASET (Kepler KOI)")
print("="*80)
print(f"Shape: {cumulative.shape}")
print(f"\nColumns ({len(cumulative.columns)}): {list(cumulative.columns[:10])}...")
print(f"\nMissing values per column (top 20):")
missing_cum = cumulative.isnull().sum().sort_values(ascending=False).head(20)
print(missing_cum)
print(f"\nData types:")
print(cumulative.dtypes.value_counts())

print("\n" + "="*80)
print("K2PANDC DATASET (K2 & Confirmed Planets)")
print("="*80)
print(f"Shape: {k2pandc.shape}")
print(f"\nColumns ({len(k2pandc.columns)}): {list(k2pandc.columns[:10])}...")
print(f"\nMissing values per column (top 20):")
missing_k2 = k2pandc.isnull().sum().sort_values(ascending=False).head(20)
print(missing_k2)
print(f"\nData types:")
print(k2pandc.dtypes.value_counts())

print("\n" + "="*80)
print("TOI DATASET (TESS)")
print("="*80)
print(f"Shape: {toi.shape}")
print(f"\nColumns ({len(toi.columns)}): {list(toi.columns[:10])}...")
print(f"\nMissing values per column (top 20):")
missing_toi = toi.isnull().sum().sort_values(ascending=False).head(20)
print(missing_toi)
print(f"\nData types:")
print(toi.dtypes.value_counts())

# Check for duplicates
print("\n" + "="*80)
print("DUPLICATE CHECK")
print("="*80)
print(f"Cumulative duplicates: {cumulative.duplicated().sum()}")
print(f"K2PANDC duplicates: {k2pandc.duplicated().sum()}")
print(f"TOI duplicates: {toi.duplicated().sum()}")

# Summary statistics for key numerical columns
print("\n" + "="*80)
print("KEY NUMERICAL STATISTICS")
print("="*80)

# Cumulative key columns
if 'koi_period' in cumulative.columns:
    print(f"\nCumulative - Orbital Period (koi_period):")
    print(cumulative['koi_period'].describe())

if 'koi_prad' in cumulative.columns:
    print(f"\nCumulative - Planet Radius (koi_prad):")
    print(cumulative['koi_prad'].describe())

# K2PANDC key columns
if 'pl_orbper' in k2pandc.columns:
    print(f"\nK2PANDC - Orbital Period (pl_orbper):")
    print(k2pandc['pl_orbper'].describe())

if 'pl_rade' in k2pandc.columns:
    print(f"\nK2PANDC - Planet Radius (pl_rade):")
    print(k2pandc['pl_rade'].describe())

# TOI key columns
if 'pl_orbper' in toi.columns:
    print(f"\nTOI - Orbital Period (pl_orbper):")
    print(toi['pl_orbper'].describe())

if 'pl_rade' in toi.columns:
    print(f"\nTOI - Planet Radius (pl_rade):")
    print(toi['pl_rade'].describe())
