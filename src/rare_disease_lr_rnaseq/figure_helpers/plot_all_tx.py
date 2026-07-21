"""
Plot all transcripts per sample by their SQANTI3 structural category.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rare_disease_lr_rnaseq.utils import (
    DATA_DIR,
    read_sqanti3_annotated,
    get_long_read_sample_ids,
    get_unique_tx
)

logger = logging.getLogger(__name__)

MENDELIAN_GENE_DISEASE_TABLE_FILEPATH = f"{DATA_DIR}/mendelian_gene_disease_table_1_16_2026.tsv"
CLINGEN_EVIDENCE = {"Definitive", "Strong", "Moderate", "Limited"}

EVIDENCE_COLORS = {
    'Definitive': '#1a5276',
    'Strong': '#2874a6',
    'Moderate': '#5dade2',
    'Limited': '#aed6f1'
}

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
})

CATEGORY_COLORS = {
    'full-splice_match': '#4477AA',
    'incomplete-splice_match': '#66CCEE',
    'novel_in_catalog': '#228833',
    'novel_not_in_catalog': '#CCBB44',
    'genic': '#EE6677',
    'antisense': '#AA3377',
    'fusion': '#BBBBBB',
    'intergenic': '#999999',
}


def get_disease_associated_genes() -> pd.DataFrame:
    """
    Load disease-associated genes from mendelian gene disease table.

    :return: DataFrame with disease-associated genes and their CLINGEN classifications.
    """
    mendelian_da_genes_df = pd.read_csv(
        MENDELIAN_GENE_DISEASE_TABLE_FILEPATH,
        sep='\t',
        usecols=["gene_id", "gene_symbol", "CLINGEN_classification"]
    )

    mendelian_da_genes = mendelian_da_genes_df[
        mendelian_da_genes_df["CLINGEN_classification"].apply(
            lambda x: all(val.strip() in CLINGEN_EVIDENCE for val in str(x).split(';'))
        )
    ]

    return mendelian_da_genes


def get_expressed_da_genes(all_tx: dict[str, pd.DataFrame], da_genes_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calculate the number of disease-associated genes that have transcripts
    mapping to them across all samples.

    :param all_tx: Dictionary mapping sample_id to DataFrame of all transcripts.
    :param da_genes_df: DataFrame with disease-associated genes.
    :return: A tuple of (expressed_da_genes_df, summary_dict) where summary_dict
        contains counts and per-sample breakdowns.
    """
    da_gene_symbols = set(da_genes_df['gene_symbol'].dropna().unique())

    all_expressed_genes = set()
    genes_per_sample = {}

    for sample_id, df in all_tx.items():
        if 'gene_name' not in df.columns:
            logger.info(f"Warning: 'gene_name' column not found in {sample_id}")
            continue
        sample_genes = set(df['gene_name'].dropna().unique())
        genes_per_sample[sample_id] = sample_genes
        all_expressed_genes.update(sample_genes)

    expressed_da_genes = all_expressed_genes & da_gene_symbols

    expressed_da_genes_df = da_genes_df[
        da_genes_df['gene_symbol'].isin(expressed_da_genes)
    ].copy()

    evidence_counts = {}
    for evidence in CLINGEN_EVIDENCE:
        count = expressed_da_genes_df[
            expressed_da_genes_df['CLINGEN_classification'].str.contains(evidence, na=False)
        ].shape[0]
        evidence_counts[evidence] = count

    summary = {
        'total_da_genes': len(da_gene_symbols),
        'total_expressed_genes': len(all_expressed_genes),
        'expressed_da_genes': len(expressed_da_genes),
        'not_expressed_da_genes': len(da_gene_symbols) - len(expressed_da_genes),
        'pct_da_genes_expressed': len(expressed_da_genes) / len(da_gene_symbols) * 100 if len(da_gene_symbols) > 0 else 0,
        'evidence_counts': evidence_counts,
        'genes_per_sample': {sid: len(genes) for sid, genes in genes_per_sample.items()},
        'da_genes_per_sample': {
            sid: len(genes & da_gene_symbols) for sid, genes in genes_per_sample.items()
        }
    }

    return expressed_da_genes_df, summary


def plot_expressed_da_genes(summary: dict, output_suffix: str = "") -> None:
    """
    Create visualizations for expressed disease-associated genes.

    :param summary: Summary dictionary from get_expressed_da_genes.
    :param output_suffix: Suffix for output filename.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    categories = ['Expressed', 'Not Expressed']
    counts = [summary['expressed_da_genes'], summary['not_expressed_da_genes']]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax1.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                 f'{count:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Number of Disease-Associated Genes')
    ax1.set_title(f'Disease-Associated Genes with Transcripts Detected\n'
                  f'({summary["pct_da_genes_expressed"]:.1f}% of {summary["total_da_genes"]:,} DA genes)')

    ax2 = axes[1]
    evidence_order = ['Definitive', 'Strong', 'Moderate', 'Limited']
    evidence_counts = [summary['evidence_counts'].get(e, 0) for e in evidence_order]
    evidence_colors = [EVIDENCE_COLORS[e] for e in evidence_order]

    bars2 = ax2.bar(evidence_order, evidence_counts, color=evidence_colors,
                    edgecolor='black', linewidth=1.2)

    for bar, count in zip(bars2, evidence_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(evidence_counts) * 0.02,
                 f'{count:,}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax2.set_xlabel('CLINGEN Evidence Level')
    ax2.set_ylabel('Number of Expressed DA Genes')
    ax2.set_title('Expressed Disease-Associated Genes by Evidence Level')

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/expressed_da_genes_summary{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/expressed_da_genes_summary{output_suffix}.pdf")

    fig, ax = plt.subplots(figsize=(max(12, len(summary['da_genes_per_sample']) * 0.5), 6))

    sorted_samples = sorted(summary['da_genes_per_sample'].items(),
                           key=lambda x: x[1], reverse=True)
    sample_ids = [s[0] for s in sorted_samples]
    da_counts = [s[1] for s in sorted_samples]

    bars = ax.bar(range(len(sample_ids)), da_counts, color='#3498db',
                  edgecolor='black', linewidth=0.5)

    mean_da = np.mean(da_counts)
    ax.axhline(y=mean_da, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean: {mean_da:.0f}')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Number of Disease-Associated Genes Expressed')
    ax.set_title('Disease-Associated Genes Expressed per Sample')
    ax.set_xticks(range(len(sample_ids)))
    ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/da_genes_per_sample{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/da_genes_per_sample{output_suffix}.pdf")


def get_sample_group(sample_id: str) -> str:
    """
    Return the group label for a sample based on its suffix.

    :param sample_id: Sample identifier string.
    :return: One of 'Proband', 'Parents', or 'Unknown'.
    """
    if sample_id.endswith("_3_R1"):
        return "Proband"
    elif sample_id.endswith("_1_R1") or sample_id.endswith("_2_R1"):
        return "Parents"
    else:
        return "Unknown"


def get_family_id(sample_id: str) -> str:
    """
    Extract family ID from sample ID by removing the suffix.

    :param sample_id: Sample identifier string.
    :return: Family ID portion of the sample identifier.
    """
    for suffix in ['_1_R1', '_2_R1', '_3_R1']:
        if sample_id.endswith(suffix):
            return sample_id[:-len(suffix)]
    return sample_id


GROUP_COLORS = {
    'Proband': '#4477AA',  # Blue
    'Parents': '#EE6677',  # Red
    'Unknown': '#BBBBBB',  # Grey
}


def plot_da_genes_violin_by_group(summary: dict, output_suffix: str = "") -> None:
    """
    Create a violin plot comparing DA genes expressed between Probands and Parents,
    with paired t-test annotation.

    :param summary: Summary dictionary from get_expressed_da_genes.
    :param output_suffix: Suffix for output filename.
    """
    da_per_sample = summary['da_genes_per_sample']

    records = []
    for sample_id, count in da_per_sample.items():
        group = get_sample_group(sample_id)
        family_id = get_family_id(sample_id)
        records.append({
            'sample_id': sample_id,
            'family_id': family_id,
            'group': group,
            'da_gene_count': count
        })
    df = pd.DataFrame(records)

    df = df[df['group'].isin(['Proband', 'Parents'])]

    # Group by family: proband value vs mean of parents
    families = {}
    for _, row in df.iterrows():
        fid = row['family_id']
        if fid not in families:
            families[fid] = {'proband': None, 'parents': []}
        if row['group'] == 'Proband':
            families[fid]['proband'] = row['da_gene_count']
        else:
            families[fid]['parents'].append(row['da_gene_count'])

    complete_families = {
        fid: data for fid, data in families.items()
        if data['proband'] is not None and len(data['parents']) == 2
    }

    proband_values = []
    parent_avg_values = []
    for fid, data in complete_families.items():
        proband_values.append(data['proband'])
        parent_avg_values.append(np.mean(data['parents']))

    proband_values = np.array(proband_values)
    parent_avg_values = np.array(parent_avg_values)

    t_stat, p_value = stats.ttest_rel(proband_values, parent_avg_values)

    differences = proband_values - parent_avg_values
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=df, x='group', y='da_gene_count', order=['Proband', 'Parents'],
                palette=GROUP_COLORS, ax=ax, width=0.5, saturation=1.0, linewidth=1.2,
                fliersize=0)
    sns.stripplot(data=df, x='group', y='da_gene_count', order=['Proband', 'Parents'],
                  palette=GROUP_COLORS, ax=ax, size=6, alpha=0.7, jitter=0.15, legend=False)

    y_max = df['da_gene_count'].max()
    y_annotation = y_max * 1.05

    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'

    ax.plot([0, 0, 1, 1], [y_annotation, y_annotation * 1.02, y_annotation * 1.02, y_annotation],
            color='black', linewidth=1.5)
    ax.text(0.5, y_annotation * 1.04, f'{stars}\np = {p_value:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Sample Group')
    ax.set_ylabel('Number of Disease-Associated Genes Expressed')
    ax.set_title('DA Genes Expressed: Probands vs Parents\n(Paired t-test)')
    ax.set_ylim(top=y_annotation * 1.15)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/da_genes_per_sample_violin{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/da_genes_per_sample_violin{output_suffix}.pdf")

    logger.info("\n--- Paired t-test: DA Genes (Proband vs Parents) ---")
    logger.info(f"  Number of complete families: {len(complete_families)}")
    logger.info(f"  Proband mean: {np.mean(proband_values):.1f} ± {np.std(proband_values, ddof=1):.1f}")
    logger.info(f"  Parents mean: {np.mean(parent_avg_values):.1f} ± {np.std(parent_avg_values, ddof=1):.1f}")
    logger.info(f"  Mean difference: {np.mean(differences):.1f}")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f} {stars}")
    logger.info(f"  Cohen's d: {cohens_d:.3f}")


def plot_all_tx_violin_by_group(all_tx: dict[str, pd.DataFrame], output_suffix: str = "") -> None:
    """
    Create a boxplot with stripplot showing all transcript counts by structural category,
    split by Proband vs Parents, with paired t-test annotations.

    :param all_tx: Dictionary mapping sample_id to DataFrame of all transcripts.
    :param output_suffix: Suffix for output filename.
    """
    records = []
    for sample_id, df in all_tx.items():
        group = get_sample_group(sample_id)
        if group == 'Unknown':
            continue
        family_id = get_family_id(sample_id)
        category_counts = df.groupby('structural_category').size().to_dict()
        for cat, count in category_counts.items():
            records.append({
                'sample_id': sample_id,
                'family_id': family_id,
                'group': group,
                'category': cat,
                'count': count
            })

    df_long = pd.DataFrame(records)

    category_medians = df_long.groupby('category')['count'].median().sort_values(ascending=False)
    category_order = category_medians.index.tolist()
    all_categories = category_order

    sample_ids = df_long['sample_id'].unique()
    sample_groups = {sid: get_sample_group(sid) for sid in sample_ids}

    families = {}
    for sample_id in sample_ids:
        family_id = get_family_id(sample_id)
        if family_id not in families:
            families[family_id] = {'proband': None, 'parents': []}

        sample_data = df_long[df_long['sample_id'] == sample_id]
        counts_dict = dict(zip(sample_data['category'], sample_data['count']))

        if sample_groups[sample_id] == 'Proband':
            families[family_id]['proband'] = counts_dict
        else:
            families[family_id]['parents'].append(counts_dict)

    complete_families = {fid: data for fid, data in families.items()
                         if data['proband'] is not None and len(data['parents']) == 2}

    pvalues = {}
    for cat in all_categories:
        proband_vals = []
        parent_avg_vals = []
        for fid, data in complete_families.items():
            proband_vals.append(data['proband'].get(cat, 0))
            parent_avg_vals.append(np.mean([p.get(cat, 0) for p in data['parents']]))
        if len(proband_vals) > 1:
            _, p = stats.ttest_rel(proband_vals, parent_avg_vals)
            pvalues[cat] = p
        else:
            pvalues[cat] = 1.0

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df_long, x='category', y='count', hue='group', order=category_order,
                hue_order=['Proband', 'Parents'], palette=GROUP_COLORS,
                ax=ax, saturation=1.0, linewidth=1, fliersize=0)
    sns.stripplot(data=df_long, x='category', y='count', hue='group', order=category_order,
                  hue_order=['Proband', 'Parents'], palette=GROUP_COLORS,
                  ax=ax, dodge=True, size=4, alpha=0.6, legend=False)

    y_max = ax.get_ylim()[1]
    for i, cat in enumerate(category_order):
        p = pvalues.get(cat, 1.0)
        if p < 0.05:
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            else:
                stars = '*'
            cat_data = df_long[df_long['category'] == cat]['count']
            cat_max = cat_data.max() if len(cat_data) > 0 else 0
            ax.annotate(f'{stars}\np={p:.3f}', xy=(i, cat_max + y_max * 0.02),
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Structural Category')
    ax.set_ylabel('Number of Transcripts')
    ax.set_title('Distribution of All Transcripts by Category - Probands vs Parents\n(Paired t-test: * p<0.05, ** p<0.01, *** p<0.001)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Sample Group')
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/all_transcripts_violin_all_samples{output_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{DATA_DIR}/all_transcripts_violin_all_samples{output_suffix}.pdf")

    logger.info("\n--- Paired t-test: All Transcripts by Category (Proband vs Parents) ---")
    logger.info(f"Number of complete families: {len(complete_families)}")
    for cat in category_order:
        p = pvalues.get(cat, 1.0)
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        logger.info(f"  {cat}: p = {p:.4f} {sig}")


def analyze_novel_not_in_catalog_decomposition(all_tx: dict[str, pd.DataFrame], sample_ids: list[str]) -> dict[str, dict]:
    """
    Decompose the 'novel_not_in_catalog' transcripts into unique vs shared,
    and run paired t-tests on each subset to determine what drives the
    statistical significance.

    :param all_tx: Dictionary mapping sample_id to DataFrame of all transcripts.
    :param sample_ids: List of sample IDs.
    :return: Dictionary mapping subset names to statistical results.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DECOMPOSITION ANALYSIS: novel_not_in_catalog")
    logger.info("Unique vs Shared Transcripts")
    logger.info("=" * 70)

    logger.info("\nLoading unique transcripts...")
    unique_tx = get_unique_tx(sample_ids, rule_filter=False)

    category = 'novel_not_in_catalog'

    records = []
    for sample_id in sample_ids:
        group = get_sample_group(sample_id)
        if group == 'Unknown':
            continue
        family_id = get_family_id(sample_id)

        all_df = all_tx.get(sample_id)
        if all_df is not None:
            all_count = (all_df['structural_category'] == category).sum()
        else:
            all_count = 0

        unique_df = unique_tx.get(sample_id)
        if unique_df is not None:
            unique_count = (unique_df['structural_category'] == category).sum()
        else:
            unique_count = 0

        shared_count = all_count - unique_count

        records.append({
            'sample_id': sample_id,
            'family_id': family_id,
            'group': group,
            'all_count': all_count,
            'unique_count': unique_count,
            'shared_count': shared_count
        })

    df = pd.DataFrame(records)

    families = {}
    for _, row in df.iterrows():
        fid = row['family_id']
        if fid not in families:
            families[fid] = {'proband': None, 'parents': []}
        if row['group'] == 'Proband':
            families[fid]['proband'] = row
        else:
            families[fid]['parents'].append(row)

    complete_families = {fid: data for fid, data in families.items()
                         if data['proband'] is not None and len(data['parents']) == 2}

    logger.info(f"\nNumber of complete families: {len(complete_families)}")

    results = {}
    for subset in ['all_count', 'unique_count', 'shared_count']:
        proband_vals = []
        parent_avg_vals = []

        for fid, data in complete_families.items():
            proband_vals.append(data['proband'][subset])
            parent_avg_vals.append(np.mean([p[subset] for p in data['parents']]))

        proband_vals = np.array(proband_vals)
        parent_avg_vals = np.array(parent_avg_vals)
        differences = proband_vals - parent_avg_vals

        t_stat, p_value = stats.ttest_rel(proband_vals, parent_avg_vals)

        cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0

        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        results[subset] = {
            'proband_mean': np.mean(proband_vals),
            'proband_std': np.std(proband_vals, ddof=1),
            'parent_mean': np.mean(parent_avg_vals),
            'parent_std': np.std(parent_avg_vals, ddof=1),
            'mean_diff': np.mean(differences),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'stars': stars
        }

    subset_labels = {
        'all_count': 'All Transcripts',
        'unique_count': 'Unique Transcripts',
        'shared_count': 'Shared Transcripts (All - Unique)'
    }

    for subset, label in subset_labels.items():
        r = results[subset]
        logger.info(f"\n--- {label} ---")
        logger.info(f"  Proband mean:  {r['proband_mean']:.1f} ± {r['proband_std']:.1f}")
        logger.info(f"  Parents mean:  {r['parent_mean']:.1f} ± {r['parent_std']:.1f}")
        logger.info(f"  Difference:    {r['mean_diff']:.1f}")
        logger.info(f"  t-statistic:   {r['t_stat']:.3f}")
        logger.info(f"  p-value:       {r['p_value']:.4f} {r['stars']}")
        logger.info(f"  Cohen's d:     {r['cohens_d']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    subset_list = ['all_count', 'unique_count', 'shared_count']
    titles = ['All Transcripts', 'Unique Transcripts', 'Shared Transcripts']

    for idx, (subset, title) in enumerate(zip(subset_list, titles)):
        ax = axes[idx]

        plot_data = df[['group', subset]].copy()
        plot_data.columns = ['group', 'count']

        sns.boxplot(data=plot_data, x='group', y='count', order=['Proband', 'Parents'],
                    palette=GROUP_COLORS, ax=ax, width=0.5, saturation=1.0, linewidth=1.2,
                    fliersize=0)
        sns.stripplot(data=plot_data, x='group', y='count', order=['Proband', 'Parents'],
                      palette=GROUP_COLORS, ax=ax, size=6, alpha=0.7, jitter=0.15, legend=False)

        r = results[subset]
        y_max = plot_data['count'].max()
        y_annotation = y_max * 1.05

        ax.plot([0, 0, 1, 1], [y_annotation, y_annotation * 1.02, y_annotation * 1.02, y_annotation],
                color='black', linewidth=1.5)
        ax.text(0.5, y_annotation * 1.04, f"{r['stars']}\np = {r['p_value']:.4f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Sample Group')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(novel_not_in_catalog)')
        ax.set_ylim(top=y_annotation * 1.18)

    plt.suptitle('Decomposition: Is the difference driven by Unique or Shared transcripts?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/novel_not_in_catalog_decomposition.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{DATA_DIR}/novel_not_in_catalog_decomposition.pdf")

    logger.info("\n" + "-" * 70)
    logger.info("INTERPRETATION:")
    logger.info("-" * 70)

    unique_sig = results['unique_count']['p_value'] < 0.05
    shared_sig = results['shared_count']['p_value'] < 0.05

    if unique_sig and not shared_sig:
        logger.info("The significance is driven by UNIQUE transcripts.")
        logger.info("Shared transcripts do not show a significant difference.")
    elif shared_sig and not unique_sig:
        logger.info("The significance is driven by SHARED transcripts.")
        logger.info("Unique transcripts do not show a significant difference.")
    elif unique_sig and shared_sig:
        logger.info("BOTH unique and shared transcripts contribute to the significance.")
        # Compare effect sizes
        if abs(results['unique_count']['cohens_d']) > abs(results['shared_count']['cohens_d']):
            logger.info(f"Unique transcripts have a larger effect size (d={results['unique_count']['cohens_d']:.3f} vs d={results['shared_count']['cohens_d']:.3f}).")
        else:
            logger.info(f"Shared transcripts have a larger effect size (d={results['shared_count']['cohens_d']:.3f} vs d={results['unique_count']['cohens_d']:.3f}).")
    else:
        logger.info("Neither unique nor shared transcripts show significant differences individually.")
        logger.info("The overall significance may be due to combined effects.")

    logger.info("=" * 70)

    return results


def get_highest_evidence_level(classification_str: str) -> Optional[str]:
    """
    Given a CLINGEN classification string (possibly with multiple levels separated by ';'),
    return the highest evidence level.

    Hierarchy: Definitive > Strong > Moderate > Limited

    :param classification_str: Semicolon-separated CLINGEN classification string.
    :return: Highest evidence level found, or None if no valid level present.
    """
    if pd.isna(classification_str):
        return None

    levels = [c.strip() for c in str(classification_str).split(';')]
    hierarchy = ['Definitive', 'Strong', 'Moderate', 'Limited']

    for level in hierarchy:
        if level in levels:
            return level
    return None


def plot_novel_not_in_catalog_by_clingen(all_tx: dict[str, pd.DataFrame], output_suffix: str = "") -> tuple[dict[str, float], dict[str, float]]:
    """
    Map novel_not_in_catalog transcripts to disease-associated genes and plot
    by CLINGEN evidence level, comparing Probands vs Parents.

    :param all_tx: Dictionary mapping sample_id to DataFrame of all transcripts.
    :param output_suffix: Suffix for output filename.
    :return: A tuple of (pvalues, effect_sizes) dictionaries keyed by CLINGEN category.
    """
    logger.info("\n" + "=" * 70)
    logger.info("NOVEL_NOT_IN_CATALOG TRANSCRIPTS BY CLINGEN EVIDENCE")
    logger.info("=" * 70)

    da_genes_df = pd.read_csv(
        MENDELIAN_GENE_DISEASE_TABLE_FILEPATH,
        sep='\t',
        usecols=["gene_symbol", "CLINGEN_classification"]
    )

    gene_to_evidence = {}
    for _, row in da_genes_df.iterrows():
        gene = row['gene_symbol']
        evidence = get_highest_evidence_level(row['CLINGEN_classification'])
        if evidence:
            gene_to_evidence[gene] = evidence

    logger.info(f"Loaded {len(gene_to_evidence)} disease-associated genes")

    categories = ['Definitive', 'Strong', 'Moderate', 'Limited', 'Not in DA genes']

    records = []
    for sample_id, df in all_tx.items():
        group = get_sample_group(sample_id)
        if group == 'Unknown':
            continue
        family_id = get_family_id(sample_id)

        novel_df = df[df['structural_category'] == 'novel_not_in_catalog']

        counts = {cat: 0 for cat in categories}

        for _, tx_row in novel_df.iterrows():
            gene = tx_row.get('gene_name', None)
            if pd.isna(gene) or gene is None:
                counts['Not in DA genes'] += 1
            elif gene in gene_to_evidence:
                counts[gene_to_evidence[gene]] += 1
            else:
                counts['Not in DA genes'] += 1

        for cat in categories:
            records.append({
                'sample_id': sample_id,
                'family_id': family_id,
                'group': group,
                'clingen_category': cat,
                'count': counts[cat]
            })

    df_long = pd.DataFrame(records)

    sample_ids = df_long['sample_id'].unique()
    families = {}
    for sample_id in sample_ids:
        family_id = get_family_id(sample_id)
        group = get_sample_group(sample_id)
        if family_id not in families:
            families[family_id] = {'proband': None, 'parents': []}

        sample_data = df_long[df_long['sample_id'] == sample_id]
        counts_dict = dict(zip(sample_data['clingen_category'], sample_data['count']))

        if group == 'Proband':
            families[family_id]['proband'] = counts_dict
        else:
            families[family_id]['parents'].append(counts_dict)

    complete_families = {fid: data for fid, data in families.items()
                         if data['proband'] is not None and len(data['parents']) == 2}

    logger.info(f"Number of complete families: {len(complete_families)}")

    pvalues = {}
    effect_sizes = {}
    for cat in categories:
        proband_vals = []
        parent_avg_vals = []
        for fid, data in complete_families.items():
            proband_vals.append(data['proband'].get(cat, 0))
            parent_avg_vals.append(np.mean([p.get(cat, 0) for p in data['parents']]))

        proband_vals = np.array(proband_vals)
        parent_avg_vals = np.array(parent_avg_vals)
        differences = proband_vals - parent_avg_vals

        if len(proband_vals) > 1:
            _, p = stats.ttest_rel(proband_vals, parent_avg_vals)
            pvalues[cat] = p
            cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0
            effect_sizes[cat] = cohens_d
        else:
            pvalues[cat] = 1.0
            effect_sizes[cat] = 0.0

    fig, ax = plt.subplots(figsize=(12, 7))

    clingen_colors = {
        'Definitive': '#1a5276',
        'Strong': '#2874a6',
        'Moderate': '#5dade2',
        'Limited': '#aed6f1',
        'Not in DA genes': '#bdc3c7'
    }

    sns.boxplot(data=df_long, x='clingen_category', y='count', hue='group',
                order=categories, hue_order=['Proband', 'Parents'],
                palette=GROUP_COLORS, ax=ax, saturation=1.0, linewidth=1, fliersize=0)
    sns.stripplot(data=df_long, x='clingen_category', y='count', hue='group',
                  order=categories, hue_order=['Proband', 'Parents'],
                  palette=GROUP_COLORS, ax=ax, dodge=True, size=4, alpha=0.6, legend=False)

    y_max = ax.get_ylim()[1]
    for i, cat in enumerate(categories):
        p = pvalues.get(cat, 1.0)
        d = effect_sizes.get(cat, 0.0)

        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = 'n.s.'

        cat_data = df_long[df_long['clingen_category'] == cat]['count']
        cat_max = cat_data.max() if len(cat_data) > 0 else 0

        annotation = f'{stars}\np={p:.3f}\nd={d:.2f}'
        ax.annotate(annotation, xy=(i, cat_max + y_max * 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('CLINGEN Evidence Level')
    ax.set_ylabel('Number of novel_not_in_catalog Transcripts')
    ax.set_title('Novel Not In Catalog Transcripts by CLINGEN Evidence\n'
                 '(Paired t-test: * p<0.05, ** p<0.01, *** p<0.001)')
    ax.legend(title='Sample Group', loc='upper left')

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/novel_not_in_catalog_by_clingen{output_suffix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{DATA_DIR}/novel_not_in_catalog_by_clingen{output_suffix}.pdf")

    logger.info("\n--- Paired t-test Results by CLINGEN Category ---")
    for cat in categories:
        p = pvalues[cat]
        d = effect_sizes[cat]
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = ''

        proband_mean = df_long[(df_long['clingen_category'] == cat) &
                               (df_long['group'] == 'Proband')]['count'].mean()
        parent_mean = df_long[(df_long['clingen_category'] == cat) &
                              (df_long['group'] == 'Parents')]['count'].mean()

        logger.info(f"  {cat}:")
        logger.info(f"    Proband mean: {proband_mean:.1f}, Parents mean: {parent_mean:.1f}")
        logger.info(f"    p = {p:.4f} {stars}, Cohen's d = {d:.3f}")

    logger.info("=" * 70)

    return pvalues, effect_sizes


def print_da_genes_summary(summary: dict) -> None:
    """
    Print summary statistics for expressed disease-associated genes.

    :param summary: Summary dictionary from get_expressed_da_genes.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DISEASE-ASSOCIATED GENES EXPRESSION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nTotal disease-associated genes in database: {summary['total_da_genes']:,}")
    logger.info(f"Total unique genes expressed across all samples: {summary['total_expressed_genes']:,}")
    logger.info(f"\nDisease-associated genes with transcripts detected: {summary['expressed_da_genes']:,}")
    logger.info(f"Disease-associated genes NOT expressed: {summary['not_expressed_da_genes']:,}")
    logger.info(f"Percentage of DA genes expressed: {summary['pct_da_genes_expressed']:.1f}%")

    logger.info("\n--- Breakdown by CLINGEN Evidence Level ---")
    evidence_order = ['Definitive', 'Strong', 'Moderate', 'Limited']
    for evidence in evidence_order:
        count = summary['evidence_counts'].get(evidence, 0)
        logger.info(f"  {evidence}: {count:,} genes")

    logger.info("\n--- DA Genes Expressed per Sample ---")
    da_per_sample = summary['da_genes_per_sample']
    values = list(da_per_sample.values())
    logger.info(f"  Mean: {np.mean(values):.0f}")
    logger.info(f"  Std:  {np.std(values):.0f}")
    logger.info(f"  Min:  {np.min(values):,} ({min(da_per_sample, key=da_per_sample.get)})")
    logger.info(f"  Max:  {np.max(values):,} ({max(da_per_sample, key=da_per_sample.get)})")
    logger.info("=" * 70)


def get_all_tx(sample_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load all transcripts for each sample using read_sqanti3_annotated.

    :param sample_ids: List of sample IDs.
    :return: Dictionary mapping sample_id to DataFrame of all transcripts.
    """
    all_tx = {}
    for sample_id in sample_ids:
        logger.info(f"Loading {sample_id}...")
        df = read_sqanti3_annotated(sample_id, rules_filter=False)
        all_tx[sample_id] = df
    return all_tx


def get_category_counts(tx_dict: dict[str, pd.DataFrame], sample_ids: list[str]) -> dict[str, pd.DataFrame]:
    """
    Count transcripts by structural category for each sample.

    :param tx_dict: Dictionary mapping sample_id to DataFrame.
    :param sample_ids: List of sample IDs to include.
    :return: Dictionary mapping sample_id to DataFrame with category counts.
    """
    counts_by_category = {}
    for sample_id in sample_ids:
        if sample_id not in tx_dict:
            continue
        df = tx_dict[sample_id]
        category_counts = df.groupby('structural_category').size().reset_index(name='count')
        counts_by_category[sample_id] = category_counts
    return counts_by_category


def plot_stacked_bar_by_sample(counts_by_category: dict[str, pd.DataFrame], output_suffix: str = "") -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    """
    Create a stacked bar chart showing transcript counts by sample and category.
    Samples are grouped by Probands first, then Parents, sorted by total count
    within each group.

    :param counts_by_category: Dictionary mapping sample_id to DataFrame with counts.
    :param output_suffix: Suffix for output filename.
    :return: A tuple of (data_matrix, sorted_sample_ids, all_categories, percentage_matrix).
    """
    all_categories = set()
    sample_totals = {}
    for sample_id, df in counts_by_category.items():
        all_categories.update(df['structural_category'].tolist())
        sample_totals[sample_id] = df['count'].sum()

    all_categories = sorted(all_categories)
    sample_ids = list(counts_by_category.keys())
    sample_groups = [get_sample_group(s) for s in sample_ids]

    # Sort by group first (Proband=0, Parents=1), then by total count descending within group
    group_order = {'Proband': 0, 'Parents': 1, 'Unknown': 2}
    sort_keys = [(group_order.get(sample_groups[i], 2), -sample_totals[sample_ids[i]]) for i in range(len(sample_ids))]
    sorted_idx = sorted(range(len(sample_ids)), key=lambda i: sort_keys[i])

    sorted_sample_ids = [sample_ids[i] for i in sorted_idx]
    sorted_sample_groups = [sample_groups[i] for i in sorted_idx]

    data_matrix = np.zeros((len(sorted_sample_ids), len(all_categories)))
    for i, sample_id in enumerate(sorted_sample_ids):
        counts_dict = dict(zip(
            counts_by_category[sample_id]['structural_category'],
            counts_by_category[sample_id]['count']
        ))
        for j, cat in enumerate(all_categories):
            data_matrix[i, j] = counts_dict.get(cat, 0)

    row_totals = data_matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (data_matrix / row_totals) * 100

    fig_width = max(12, len(sorted_sample_ids) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(sorted_sample_ids))
    bottom = np.zeros(len(sorted_sample_ids))

    for j, cat in enumerate(all_categories):
        color = CATEGORY_COLORS.get(cat, f'C{j}')
        ax.bar(x, data_matrix[:, j], bottom=bottom, label=cat,
               color=color, edgecolor='white', linewidth=0.5)
        bottom += data_matrix[:, j]

    for i in range(1, len(sorted_sample_groups)):
        if sorted_sample_groups[i] != sorted_sample_groups[i - 1]:
            ax.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=2)

    proband_indices = [i for i, g in enumerate(sorted_sample_groups) if g == 'Proband']
    parent_indices = [i for i, g in enumerate(sorted_sample_groups) if g == 'Parents']

    if proband_indices:
        mid_proband = (proband_indices[0] + proband_indices[-1]) / 2
        ax.text(mid_proband, ax.get_ylim()[1] * 1.02, 'Probands', ha='center', fontsize=14, fontweight='bold')
    if parent_indices:
        mid_parent = (parent_indices[0] + parent_indices[-1]) / 2
        ax.text(mid_parent, ax.get_ylim()[1] * 1.02, 'Parents', ha='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Number of Transcripts')
    ax.set_title('All Transcripts by Structural Category')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_sample_ids, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/all_tx_by_sample{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/all_tx_by_sample{output_suffix}.pdf")

    return data_matrix, sorted_sample_ids, all_categories, percentage_matrix


def plot_percentage_stacked_bar(data_matrix: np.ndarray, sample_ids: list[str], all_categories: list[str], output_suffix: str = "") -> None:
    """
    Create a 100% stacked bar chart showing proportions.
    Samples are expected to be pre-sorted (Probands first, then Parents).

    :param data_matrix: Matrix of counts with shape (samples, categories).
    :param sample_ids: List of sample IDs (pre-sorted by group).
    :param all_categories: List of category names.
    :param output_suffix: Suffix for output filename.
    """
    row_totals = data_matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (data_matrix / row_totals) * 100

    sample_groups = [get_sample_group(s) for s in sample_ids]

    fig_width = max(12, len(sample_ids) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(sample_ids))
    bottom = np.zeros(len(sample_ids))

    for j, cat in enumerate(all_categories):
        color = CATEGORY_COLORS.get(cat, f'C{j}')
        ax.bar(x, percentage_matrix[:, j], bottom=bottom, label=cat,
               color=color, edgecolor='white', linewidth=0.5)
        bottom += percentage_matrix[:, j]

    for i in range(1, len(sample_groups)):
        if sample_groups[i] != sample_groups[i - 1]:
            ax.axvline(x=i - 0.5, color='black', linestyle='--', linewidth=2)

    proband_indices = [i for i, g in enumerate(sample_groups) if g == 'Proband']
    parent_indices = [i for i, g in enumerate(sample_groups) if g == 'Parents']

    if proband_indices:
        mid_proband = (proband_indices[0] + proband_indices[-1]) / 2
        ax.text(mid_proband, 103, 'Probands', ha='center', fontsize=14, fontweight='bold')
    if parent_indices:
        mid_parent = (parent_indices[0] + parent_indices[-1]) / 2
        ax.text(mid_parent, 103, 'Parents', ha='center', fontsize=14, fontweight='bold')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Percentage of Transcripts')
    ax.set_title('Proportion of Transcripts by Structural Category')
    ax.set_xticks(x)
    ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/all_tx_percentage_by_sample{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/all_tx_percentage_by_sample{output_suffix}.pdf")


def plot_category_summary(data_matrix: np.ndarray, sample_ids: list[str], all_categories: list[str], output_suffix: str = "") -> None:
    """
    Create a summary plot showing mean counts with error bars by category.

    :param data_matrix: Matrix of counts with shape (samples, categories).
    :param sample_ids: List of sample IDs.
    :param all_categories: List of category names.
    :param output_suffix: Suffix for output filename.
    """
    means = np.mean(data_matrix, axis=0)
    sems = stats.sem(data_matrix, axis=0)

    sorted_idx = np.argsort(means)[::-1]
    sorted_categories = [all_categories[i] for i in sorted_idx]
    sorted_means = means[sorted_idx]
    sorted_sems = sems[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sorted_categories))
    colors = [CATEGORY_COLORS.get(cat, '#888888') for cat in sorted_categories]

    bars = ax.bar(x, sorted_means, yerr=sorted_sems, capsize=3,
                  color=colors, edgecolor='black', linewidth=0.5)

    for i, (bar, mean, sem) in enumerate(zip(bars, sorted_means, sorted_sems)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + sorted_means.max() * 0.02,
                f'{int(mean):,}', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Structural Category')
    ax.set_ylabel('Mean Number of Transcripts')
    ax.set_title('Mean Transcript Count by Structural Category')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_categories, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/all_tx_category_summary{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/all_tx_category_summary{output_suffix}.pdf")

    logger.info("\n=== Transcript Count Statistics by Category ===")
    row_totals = data_matrix.sum(axis=1)
    total_mean = np.mean(row_totals)
    total_std = np.std(row_totals)
    logger.info(f"Total transcripts per sample: {total_mean:.0f} +/- {total_std:.0f}")
    logger.info(f"\nBy category (sorted by mean count):")
    for cat, mean, sem in zip(sorted_categories, sorted_means, sorted_sems):
        pct = mean / total_mean * 100
        logger.info(f"  {cat}: {mean:.0f} +/- {sem:.0f} ({pct:.1f}%)")


def plot_violin_by_category(data_matrix: np.ndarray, all_categories: list[str], output_suffix: str = "") -> None:
    """
    Create violin plots showing distribution of counts across samples for each category.

    :param data_matrix: Matrix of counts with shape (samples, categories).
    :param all_categories: List of category names.
    :param output_suffix: Suffix for output filename.
    """
    records = []
    for i in range(data_matrix.shape[0]):
        for j, cat in enumerate(all_categories):
            records.append({'category': cat, 'count': data_matrix[i, j]})
    df_long = pd.DataFrame(records)

    category_medians = df_long.groupby('category')['count'].median().sort_values(ascending=False)
    category_order = category_medians.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = [CATEGORY_COLORS.get(cat, '#888888') for cat in category_order]
    sns.violinplot(data=df_long, x='category', y='count', order=category_order,
                   palette=palette, inner='box', cut=0, ax=ax)

    ax.set_xlabel('Structural Category')
    ax.set_ylabel('Number of Transcripts')
    ax.set_title('Distribution of Transcript Counts by Structural Category')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/all_tx_violin{output_suffix}.png")
    plt.savefig(f"{DATA_DIR}/all_tx_violin{output_suffix}.pdf")


def main() -> None:
    """
    Main entry point that generates all transcript analysis plots including
    stacked bar charts, violin plots, decomposition analysis, and
    disease-associated gene expression analysis.
    """
    lr_sample_ids = get_long_read_sample_ids()

    logger.info(f"Loading transcripts for {len(lr_sample_ids)} samples...")

    all_tx = get_all_tx(lr_sample_ids)

    counts_by_category = get_category_counts(all_tx, lr_sample_ids)

    data_matrix, sample_ids, all_categories, pct_matrix = plot_stacked_bar_by_sample(
        counts_by_category, output_suffix=""
    )

    plot_percentage_stacked_bar(data_matrix, sample_ids, all_categories, output_suffix="")
    plot_category_summary(data_matrix, sample_ids, all_categories, output_suffix="")
    plot_violin_by_category(data_matrix, all_categories, output_suffix="")

    plot_all_tx_violin_by_group(all_tx, output_suffix="")

    # Decomposition analysis: Is novel_not_in_catalog significance driven by unique or shared?
    analyze_novel_not_in_catalog_decomposition(all_tx, lr_sample_ids)

    plot_novel_not_in_catalog_by_clingen(all_tx, output_suffix="")

    logger.info("\nAnalyzing disease-associated genes expression...")
    da_genes_df = get_disease_associated_genes()
    expressed_da_genes_df, summary = get_expressed_da_genes(all_tx, da_genes_df)

    print_da_genes_summary(summary)
    plot_expressed_da_genes(summary, output_suffix="")
    plot_da_genes_violin_by_group(summary, output_suffix="")

    expressed_da_genes_df.to_csv(f"{DATA_DIR}/expressed_da_genes.csv", index=False)
    logger.info(f"\nExpressed DA genes saved to {DATA_DIR}/expressed_da_genes.csv")


if __name__ == "__main__":
    main()
