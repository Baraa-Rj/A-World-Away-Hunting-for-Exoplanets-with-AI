import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_datasets():
    """Load preprocessed datasets"""
    cumulative = pd.read_csv('data/processed/cumulative_preprocessed.csv')
    k2pandc = pd.read_csv('data/processed/k2pandc_preprocessed.csv')
    toi = pd.read_csv('data/processed/toi_preprocessed.csv')
    return cumulative, k2pandc, toi

def plot_disposition_distributions(cumulative, k2pandc, toi):
    """Plot disposition distributions across datasets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cumulative
    if 'koi_disposition' in cumulative.columns:
        disp_counts = cumulative['koi_disposition'].value_counts()
        axes[0].bar(range(len(disp_counts)), disp_counts.values, color=['#2ecc71', '#e74c3c', '#3498db'])
        axes[0].set_xticks(range(len(disp_counts)))
        axes[0].set_xticklabels(disp_counts.index, rotation=45, ha='right')
        axes[0].set_title('Cumulative (Kepler) - Disposition', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(disp_counts.values):
            axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

    # K2PANDC
    if 'disposition' in k2pandc.columns:
        disp_counts = k2pandc['disposition'].value_counts()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        axes[1].bar(range(len(disp_counts)), disp_counts.values, color=colors[:len(disp_counts)])
        axes[1].set_xticks(range(len(disp_counts)))
        axes[1].set_xticklabels(disp_counts.index, rotation=45, ha='right')
        axes[1].set_title('K2PANDC - Disposition', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Count')
        for i, v in enumerate(disp_counts.values):
            axes[1].text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')

    # TOI
    if 'tfopwg_disp' in toi.columns:
        disp_counts = toi['tfopwg_disp'].value_counts()
        axes[2].bar(range(len(disp_counts)), disp_counts.values)
        axes[2].set_xticks(range(len(disp_counts)))
        axes[2].set_xticklabels(disp_counts.index, rotation=45, ha='right')
        axes[2].set_title('TOI (TESS) - Disposition', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Count')
        for i, v in enumerate(disp_counts.values):
            axes[2].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/processed/disposition_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: disposition_distributions.png")
    plt.close()

def plot_planet_properties(cumulative, k2pandc, toi):
    """Plot planet property distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Cumulative - Radius and Period
    if 'koi_prad' in cumulative.columns:
        axes[0, 0].hist(cumulative['koi_prad'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Cumulative - Planet Radius (Normalized)', fontweight='bold')
        axes[0, 0].set_xlabel('Normalized Radius')
        axes[0, 0].set_ylabel('Frequency')

    if 'koi_period' in cumulative.columns:
        axes[0, 1].hist(cumulative['koi_period'], bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Cumulative - Orbital Period (Normalized)', fontweight='bold')
        axes[0, 1].set_xlabel('Normalized Period')
        axes[0, 1].set_ylabel('Frequency')

    if 'koi_depth' in cumulative.columns:
        axes[0, 2].hist(cumulative['koi_depth'], bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Cumulative - Transit Depth (Normalized)', fontweight='bold')
        axes[0, 2].set_xlabel('Normalized Depth')
        axes[0, 2].set_ylabel('Frequency')

    # K2PANDC - Radius and Period
    if 'pl_rade' in k2pandc.columns:
        axes[1, 0].hist(k2pandc['pl_rade'].dropna(), bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('K2PANDC - Planet Radius (Normalized)', fontweight='bold')
        axes[1, 0].set_xlabel('Normalized Radius')
        axes[1, 0].set_ylabel('Frequency')

    if 'pl_orbper' in k2pandc.columns:
        axes[1, 1].hist(k2pandc['pl_orbper'].dropna(), bins=50, color='#e67e22', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('K2PANDC - Orbital Period (Normalized)', fontweight='bold')
        axes[1, 1].set_xlabel('Normalized Period')
        axes[1, 1].set_ylabel('Frequency')

    # TOI - Radius
    if 'pl_rade' in toi.columns:
        axes[1, 2].hist(toi['pl_rade'].dropna(), bins=50, color='#1abc9c', edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('TOI - Planet Radius (Normalized)', fontweight='bold')
        axes[1, 2].set_xlabel('Normalized Radius')
        axes[1, 2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('data/processed/planet_properties.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: planet_properties.png")
    plt.close()

def plot_habitable_zone(cumulative, toi):
    """Plot habitable zone analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative
    if 'in_habitable_zone' in cumulative.columns:
        hz_counts = cumulative['in_habitable_zone'].value_counts()
        labels = ['Outside HZ', 'In HZ']
        colors = ['#95a5a6', '#27ae60']
        explode = (0, 0.1)

        axes[0].pie(hz_counts.values, labels=labels, autopct='%1.1f%%',
                   colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[0].set_title('Cumulative - Habitable Zone Distribution', fontsize=14, fontweight='bold')

    # TOI
    if 'in_habitable_zone' in toi.columns:
        hz_counts = toi['in_habitable_zone'].value_counts()
        labels = ['Outside HZ', 'In HZ']
        colors = ['#95a5a6', '#27ae60']
        explode = (0, 0.1)

        axes[1].pie(hz_counts.values, labels=labels, autopct='%1.1f%%',
                   colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title('TOI - Habitable Zone Distribution', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('data/processed/habitable_zone_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: habitable_zone_distribution.png")
    plt.close()

def plot_correlation_heatmap(cumulative):
    """Plot correlation heatmap for key features"""
    numeric_cols = ['koi_period', 'koi_prad', 'koi_depth', 'koi_duration',
                   'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass']
    available_cols = [col for col in numeric_cols if col in cumulative.columns]

    if len(available_cols) >= 3:
        plt.figure(figsize=(12, 10))
        corr_matrix = cumulative[available_cols].corr()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Cumulative Dataset - Feature Correlations', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('data/processed/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: correlation_heatmap.png")
        plt.close()

def plot_discovery_methods(k2pandc):
    """Plot discovery methods"""
    if 'discoverymethod' in k2pandc.columns:
        plt.figure(figsize=(12, 6))
        methods = k2pandc['discoverymethod'].value_counts().head(10)

        plt.barh(range(len(methods)), methods.values, color=sns.color_palette("viridis", len(methods)))
        plt.yticks(range(len(methods)), methods.index)
        plt.xlabel('Count', fontsize=12, fontweight='bold')
        plt.title('K2PANDC - Discovery Methods', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, v in enumerate(methods.values):
            plt.text(v + 20, i, str(v), va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('data/processed/discovery_methods.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: discovery_methods.png")
        plt.close()

def plot_dataset_comparison(cumulative, k2pandc, toi):
    """Compare dataset sizes and compositions"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Dataset sizes
    datasets = ['Cumulative', 'K2PANDC', 'TOI']
    sizes = [len(cumulative), len(k2pandc), len(toi)]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    axes[0].bar(datasets, sizes, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Number of Records', fontsize=12, fontweight='bold')
    axes[0].set_title('Dataset Sizes', fontsize=14, fontweight='bold')
    for i, v in enumerate(sizes):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Confirmed planets
    cum_confirmed = (cumulative['koi_disposition'] == 'CONFIRMED').sum() if 'koi_disposition' in cumulative.columns else 0
    k2_confirmed = (k2pandc['disposition'] == 'CONFIRMED').sum() if 'disposition' in k2pandc.columns else 0
    toi_confirmed = (toi['tfopwg_disp'] == 'CP').sum() if 'tfopwg_disp' in toi.columns else 0

    confirmed = [cum_confirmed, k2_confirmed, toi_confirmed]

    axes[1].bar(datasets, confirmed, color=colors, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Number of Confirmed Planets', fontsize=12, fontweight='bold')
    axes[1].set_title('Confirmed Planets by Dataset', fontsize=14, fontweight='bold')
    for i, v in enumerate(confirmed):
        axes[1].text(i, v + 20, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('data/processed/dataset_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: dataset_comparison.png")
    plt.close()

def main():
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Load datasets
    print("\nLoading datasets...")
    cumulative, k2pandc, toi = load_datasets()
    print("✓ Datasets loaded")

    # Generate visualizations
    print("\nGenerating plots...")
    plot_disposition_distributions(cumulative, k2pandc, toi)
    plot_planet_properties(cumulative, k2pandc, toi)
    plot_habitable_zone(cumulative, toi)
    plot_correlation_heatmap(cumulative)
    plot_discovery_methods(k2pandc)
    plot_dataset_comparison(cumulative, k2pandc, toi)

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS SAVED TO: data/processed/")
    print("="*80)

if __name__ == "__main__":
    main()
