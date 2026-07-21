"""
QC Metrics Plotting Script for Long-Read RNA-seq Manuscript

Generates publication-quality QC visualizations:
1. FLNC reads distribution
2. Read length distribution (pre- and post-hemoglobin depletion)
3. Mapping rate per sample
4. RIN distribution
5. Phenotype distribution
6. Sample-level QC summary table
"""

import logging
from pathlib import Path

import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from rare_disease_lr_rnaseq.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

COLORS = {
    'primary': '#4C72B0',      # Steel blue
    'secondary': '#DD8452',    # Coral orange
    'tertiary': '#55A868',     # Sage green
    'quaternary': '#C44E52',   # Muted red
    'accent': '#8172B3',       # Purple
    'neutral': '#937860',      # Taupe
    'light_primary': '#A8C5E2',
    'light_secondary': '#F5C4A1',
}


def setup_publication_style() -> None:
    """
    Configure matplotlib for publication-quality figures.
    """
    mpl.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.grid': False,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.edgecolor': '#CCCCCC',
        'legend.fancybox': True,
        'legend.shadow': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'figure.facecolor': 'white',
    })


def load_reads_summary(reads_summary_path: Path) -> pd.DataFrame:
    """
    Load reads summary data.

    :param reads_summary_path: Path to the reads summary TSV file.
    :return: DataFrame with reads summary data including computed mapping rate.
    """
    df = pd.read_csv(reads_summary_path, sep='\t')
    df['mapping_rate'] = df['mapped_reads'] / df['flnc_reads'] * 100
    return df


def load_metadata() -> pd.DataFrame:
    """
    Load sample metadata.

    :return: DataFrame with sample metadata including renamed sample IDs.
    """
    metadata_path = Path(DATA_DIR) / "metadata.tsv"
    df = pd.read_csv(metadata_path, sep='\t')
    df = df.rename(columns={"entity:Sample_ID": "sample_id"})
    df["sample_id"] = df["sample_id"].apply(lambda x: f"{x}_R1")
    return df


def bin_by_read_length(dat: pd.DataFrame, bin_size: int = 50, length_limit: int = 3000) -> pd.DataFrame:
    """
    Bin read lengths into intervals.

    :param dat: DataFrame with read_length and count columns.
    :param bin_size: Size of each bin in base pairs.
    :param length_limit: Maximum read length to include.
    :return: DataFrame with binned read lengths and aggregated counts.
    """
    dat = dat[dat["read_length"] <= length_limit].copy()
    bin_labels = (dat['read_length'] // bin_size) * bin_size
    dat = dat.groupby(bin_labels)['count'].sum().reset_index()
    dat.columns = ['read_length', 'count']
    return dat


def load_read_length_data_both() -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """
    Load read length data for both pre- and post-hemoglobin depletion.

    :return: Tuple of (pre_matrix, post_matrix, bins) for error band calculation.
    """
    rl_dir = Path(DATA_DIR) / "rl_distribution"

    pre_binned = []
    post_binned = []

    for filepath in rl_dir.glob("*_read_lengths.txt"):
        dat = pd.read_csv(filepath, sep="\t", header=None)
        dat.columns = ["read_length", "count"]
        binned = bin_by_read_length(dat)

        if "no_top_genes" in filepath.name:
            post_binned.append(binned)
        else:
            pre_binned.append(binned)

    if not pre_binned:
        raise FileNotFoundError(f"No read length files found in {rl_dir}")

    all_bins = sorted(set().union(
        *[set(df['read_length']) for df in pre_binned],
        *[set(df['read_length']) for df in post_binned]
    ))

    def to_matrix(binned_list: list[pd.DataFrame]) -> np.ndarray:
        """
        Convert a list of binned DataFrames into a count matrix.

        :param binned_list: List of binned read length DataFrames.
        :return: Matrix of read counts with shape (n_samples, n_bins).
        """
        matrix = np.zeros((len(binned_list), len(all_bins)))
        for i, df in enumerate(binned_list):
            for _, row in df.iterrows():
                if row['read_length'] in all_bins:
                    bin_idx = all_bins.index(row['read_length'])
                    matrix[i, bin_idx] = row['count']
        return matrix

    pre_matrix = to_matrix(pre_binned)
    post_matrix = to_matrix(post_binned) if post_binned else None

    return pre_matrix, post_matrix, all_bins


def plot_flnc_reads_distribution(reads_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot FLNC reads distribution as a KDE density plot.

    :param reads_df: DataFrame with reads summary data.
    :param output_dir: Output directory for the saved plot.
    """
    setup_publication_style()

    values = reads_df['flnc_reads'] / 1e6

    fig, ax = plt.subplots(figsize=(8, 5))

    kde = stats.gaussian_kde(values)
    x_kde = np.linspace(values.min() - 0.5, values.max() + 0.5, 200)
    y_kde = kde(x_kde)

    ax.fill_between(x_kde, y_kde, alpha=0.3, color=COLORS['primary'])
    ax.plot(x_kde, y_kde, color=COLORS['primary'], linewidth=2.5)

    jitter = np.random.uniform(-0.02, 0.02, len(values)) * y_kde.max()
    ax.scatter(values, jitter + y_kde.max() * 0.05,
               s=40, alpha=0.6, color=COLORS['secondary'],
               edgecolor='white', linewidth=0.5, zorder=3)

    mean_reads = values.mean()
    median_reads = values.median()

    ax.axvline(mean_reads, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Mean: {mean_reads:.1f}M')
    ax.axvline(median_reads, color=COLORS['tertiary'], linestyle=':', linewidth=2,
               label=f'Median: {median_reads:.1f}M')

    ax.set_xlabel('FLNC Reads (millions)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / "flnc_reads_distribution.png")
    plt.close()
    logger.info(f"Saved FLNC reads distribution to {output_dir}/flnc_reads_distribution.png")


def plot_read_length_distribution(output_dir: Path) -> None:
    """
    Plot read length distribution with pre- and post-hemoglobin depletion.

    Uses mean counts across samples with std error shadows.

    :param output_dir: Output directory for the saved plot.
    """
    setup_publication_style()

    pre_matrix, post_matrix, bins = load_read_length_data_both()

    def aggregate_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std across samples for error bands.

        :param matrix: Matrix of read counts with shape (n_samples, n_bins).
        :return: Tuple of (mean_counts, std_counts) arrays.
        """
        mean_counts = np.mean(matrix, axis=0)
        std_counts = np.std(matrix, axis=0)
        return mean_counts, std_counts

    pre_mean, pre_std = aggregate_stats(pre_matrix)
    x = np.array(bins)

    fig, ax = plt.subplots(figsize=(9, 5))

    cumsum_pre = np.cumsum(pre_mean)
    median_idx_pre = np.searchsorted(cumsum_pre, cumsum_pre[-1] / 2)
    median_rl_pre = x[median_idx_pre] if median_idx_pre < len(x) else x[-1]

    ax.fill_between(x, pre_mean - pre_std, pre_mean + pre_std,
                    alpha=0.25, color=COLORS['primary'], linewidth=0)
    line_pre, = ax.plot(x, pre_mean, linewidth=2.5, color=COLORS['primary'], zorder=3)

    handles = [line_pre]
    labels = [f'With hemoglobin (median: {median_rl_pre:.0f} bp)']

    if post_matrix is not None:
        post_mean, post_std = aggregate_stats(post_matrix)

        cumsum_post = np.cumsum(post_mean)
        median_idx_post = np.searchsorted(cumsum_post, cumsum_post[-1] / 2)
        median_rl_post = x[median_idx_post] if median_idx_post < len(x) else x[-1]

        ax.fill_between(x, post_mean - post_std, post_mean + post_std,
                        alpha=0.25, color=COLORS['secondary'], linewidth=0)
        line_post, = ax.plot(x, post_mean, linewidth=2.5, color=COLORS['secondary'], zorder=3)

        handles.append(line_post)
        labels.append(f'Without hemoglobin (median: {median_rl_post:.0f} bp)')

    ax.set_xlabel('Read Length (bp)', fontweight='bold')
    ax.set_ylabel('Mean Read Count', fontweight='bold')
    ax.set_xlim(0, 3000)
    ax.set_ylim(bottom=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(handles, labels, loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / "read_length_distribution.png")
    plt.close()
    logger.info(f"Saved read length distribution to {output_dir}/read_length_distribution.png")


def plot_mapping_rate(reads_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot mapping rate as a dot plot with KDE distribution.

    :param reads_df: DataFrame with reads summary data including mapping_rate column.
    :param output_dir: Output directory for the saved plot.
    """
    setup_publication_style()

    mapping_rates = reads_df['mapping_rate'].values

    fig, ax = plt.subplots(figsize=(8, 5))

    kde = stats.gaussian_kde(mapping_rates)
    x_kde = np.linspace(mapping_rates.min() - 0.5, mapping_rates.max() + 0.5, 200)
    y_kde = kde(x_kde)

    ax.fill_between(x_kde, y_kde, alpha=0.3, color=COLORS['tertiary'])
    ax.plot(x_kde, y_kde, color=COLORS['tertiary'], linewidth=2.5)

    jitter = np.random.uniform(-0.02, 0.02, len(mapping_rates)) * y_kde.max()
    ax.scatter(mapping_rates, jitter + y_kde.max() * 0.05,
               s=40, alpha=0.6, color=COLORS['primary'],
               edgecolor='white', linewidth=0.5, zorder=3)

    mean_rate = mapping_rates.mean()
    median_rate = np.median(mapping_rates)

    ax.axvline(mean_rate, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label=f'Mean: {mean_rate:.2f}%')
    ax.axvline(median_rate, color=COLORS['quaternary'], linestyle=':', linewidth=2,
               label=f'Median: {median_rate:.2f}%')

    ax.set_xlabel('Mapping Rate (%)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_dir / "mapping_rate.png")
    plt.close()
    logger.info(f"Saved mapping rate to {output_dir}/mapping_rate.png")


def plot_rin_distribution(metadata_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot RIN distribution as a histogram with KDE overlay.

    :param metadata_df: DataFrame with sample metadata including RIN column.
    :param output_dir: Output directory for the saved plot.
    """
    setup_publication_style()

    rin_values = metadata_df['RIN'].dropna().values

    fig, ax = plt.subplots(figsize=(7, 5))

    n, bins_hist, patches = ax.hist(rin_values, bins=12, density=True,
                                     alpha=0.7, color=COLORS['accent'],
                                     edgecolor='white', linewidth=1.2)

    for i, patch in enumerate(patches):
        alpha = 0.4 + 0.4 * (i / len(patches))
        patch.set_alpha(alpha)

    kde = stats.gaussian_kde(rin_values)
    x_kde = np.linspace(rin_values.min() - 0.5, rin_values.max() + 0.5, 200)
    ax.plot(x_kde, kde(x_kde), color=COLORS['primary'], linewidth=2.5, label='KDE')

    mean_rin = rin_values.mean()
    median_rin = np.median(rin_values)
    std_rin = rin_values.std()

    ax.axvline(mean_rin, color=COLORS['secondary'], linestyle='--', linewidth=2)
    ax.axvspan(mean_rin - std_rin, mean_rin + std_rin, alpha=0.1, color=COLORS['secondary'])

    stats_text = f"Mean: {mean_rin:.2f}\nMedian: {median_rin:.2f}\nSD: {std_rin:.2f}"
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                   edgecolor='#CCCCCC', alpha=0.9))

    ax.set_xlabel('RNA Integrity Number (RIN)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "rin_distribution.png")
    plt.close()
    logger.info(f"Saved RIN distribution to {output_dir}/rin_distribution.png")


def plot_phenotype_distribution(metadata_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot phenotype distribution as a horizontal bar chart with gradient coloring.

    :param metadata_df: DataFrame with sample metadata including Phenotype_description column.
    :param output_dir: Output directory for the saved plot.
    """
    setup_publication_style()

    phenotypes = metadata_df['Phenotype_description'].dropna()
    phenotype_counts = phenotypes.value_counts().sort_values(ascending=True)

    n_phenotypes = len(phenotype_counts)
    fig, ax = plt.subplots(figsize=(10, max(4, n_phenotypes * 0.5)))

    y = np.arange(n_phenotypes)

    cmap = plt.cm.RdYlBu_r
    colors = [cmap(0.2 + 0.6 * i / n_phenotypes) for i in range(n_phenotypes)]

    bars = ax.barh(y, phenotype_counts.values, color=colors,
                   edgecolor='white', linewidth=1, height=0.7)

    for i, (bar, count) in enumerate(zip(bars, phenotype_counts.values)):
        ax.text(count + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{count}', ha='left', va='center', fontsize=10, fontweight='bold',
                color='#333333')

    ax.set_yticks(y)
    ax.set_yticklabels(phenotype_counts.index, fontsize=10)
    ax.set_xlabel('Number of Probands', fontweight='bold')

    ax.set_xlim(0, phenotype_counts.max() * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    ax.axvline(0, color='#333333', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / "phenotype_distribution.png")
    plt.close()
    logger.info(f"Saved phenotype distribution to {output_dir}/phenotype_distribution.png")


def generate_qc_summary_table(reads_df: pd.DataFrame, metadata_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate QC summary table as CSV and styled PNG.

    :param reads_df: DataFrame with reads summary data.
    :param metadata_df: DataFrame with sample metadata.
    :param output_dir: Output directory for the saved table files.
    """
    merged = reads_df.merge(
        metadata_df[['sample_id', 'RIN', 'Affected_Status', 'Phenotype_description']],
        on='sample_id',
        how='left'
    )

    summary = merged[['sample_id', 'RIN', 'flnc_reads', 'mapped_reads', 'mapping_rate',
                      'Affected_Status', 'Phenotype_description']].copy()
    summary = summary.sort_values('sample_id').reset_index(drop=True)

    summary.to_csv(output_dir / "qc_summary_table.csv", index=False)
    logger.info(f"Saved QC summary table to {output_dir}/qc_summary_table.csv")

    setup_publication_style()

    display_df = summary[['sample_id', 'RIN', 'flnc_reads', 'mapped_reads', 'mapping_rate']].copy()
    display_df['flnc_reads'] = (display_df['flnc_reads'] / 1e6).round(2)
    display_df['mapped_reads'] = (display_df['mapped_reads'] / 1e6).round(2)
    display_df['mapping_rate'] = display_df['mapping_rate'].round(2)
    display_df.columns = ['Sample ID', 'RIN', 'FLNC (M)', 'Mapped (M)', 'Map Rate (%)']

    fig, ax = plt.subplots(figsize=(10, min(18, len(display_df) * 0.28 + 1.5)))
    ax.axis('off')

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        colColours=[COLORS['primary']] * len(display_df.columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.15)

    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(color='white', fontweight='bold')
            cell.set_facecolor(COLORS['primary'])
        else:
            if key[0] % 2 == 0:
                cell.set_facecolor('#F5F5F5')
            else:
                cell.set_facecolor('white')
        cell.set_edgecolor('#DDDDDD')

    plt.tight_layout()
    plt.savefig(output_dir / "qc_summary_table.png", bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved QC summary table figure to {output_dir}/qc_summary_table.png")


def main(reads_summary: Path, output_dir: Path) -> None:
    """
    Generate QC plots for long-read RNA-seq manuscript.

    :param reads_summary: Path to the reads summary TSV file.
    :param output_dir: Output directory for generated plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info("Loading data...")
    reads_df = load_reads_summary(reads_summary)
    metadata_df = load_metadata()

    logger.info(f"Loaded {len(reads_df)} samples from reads_summary")
    logger.info(f"Loaded {len(metadata_df)} samples from metadata")

    logger.info("\nGenerating plots...")

    plot_flnc_reads_distribution(reads_df, output_dir)
    plot_read_length_distribution(output_dir)
    plot_mapping_rate(reads_df, output_dir)
    plot_rin_distribution(metadata_df, output_dir)
    plot_phenotype_distribution(metadata_df, output_dir)
    generate_qc_summary_table(reads_df, metadata_df, output_dir)

    logger.info("\nAll plots generated successfully!")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QC plots for long-read RNA-seq manuscript."
    )
    parser.add_argument(
        "--reads-summary", "-r",
        type=Path,
        default=Path("reads_summary.tsv"),
        help="Path to reads_summary.tsv file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("output/qc_plots"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()
    main(reads_summary=args.reads_summary, output_dir=args.output_dir)
