"""Generate cumulative read distribution plots by transcript expression rank."""

import logging
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids, read_gtf, read_quant_expr, read_sqanti3_annotated

logger = logging.getLogger(__name__)

COLORS = {
    'primary': '#4C72B0',
    'secondary': '#DD8452',
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


# 1-based
GENES_TO_REMOVE = {"chr11:5225464-5229395", # HBB
                    "chr16:176680-177522", # HBA1
                    "chr16:172876-173710", # HBA2
                    }


def intersect(coords1: str, coords2: str) -> bool:
    """
    Check whether two genomic coordinate intervals overlap.

    :param coords1: First genomic interval in "chr:start-end" format.
    :param coords2: Second genomic interval in "chr:start-end" format.
    :return: True if the two intervals overlap, False otherwise.
    """
    chrom1, range1 = coords1.split(":")
    start1, end1 = map(int, range1.split("-"))
    chrom2, range2 = coords2.split(":")
    start2, end2 = map(int, range2.split("-"))

    if chrom1 != chrom2:
        return False

    if start1 > end2 or start2 > end1:
        return False

    return True


def get_gene_read_counts_dict(annotated_transcripts: pd.DataFrame) -> dict[str, int]:
    """
    Aggregate unique read counts per gene and return as a dictionary.

    :param annotated_transcripts: DataFrame with 'gene_name' and 'uniq_reads' columns.
    :return: Dictionary mapping gene names to total unique read counts,
        sorted by count descending.
    """
    gene_read_counts = annotated_transcripts.groupby('gene_name')['uniq_reads'].sum().reset_index()
    gene_read_counts = gene_read_counts.sort_values(by='uniq_reads', ascending=False)
    gene_read_counts_dict = dict(zip(gene_read_counts['gene_name'], gene_read_counts['uniq_reads']))
    return gene_read_counts_dict


def plot_read_count_distribution_by_gene(total_uniq_reads_df_top: pd.DataFrame, total_uniq_reads_df_no_top_genes: pd.DataFrame) -> None:
    """
    Plot cumulative distribution of reads per gene.

    :param total_uniq_reads_df_top: DataFrame with top genes including hemoglobin, containing a 'cum_dist' column.
    :param total_uniq_reads_df_no_top_genes: DataFrame with top genes excluding hemoglobin, containing a 'cum_dist' column.
    """
    setup_publication_style()

    fig, ax = plt.subplots(figsize=(9, 5))

    x_top = np.arange(1, len(total_uniq_reads_df_top) + 1)
    x_no_top = np.arange(1, len(total_uniq_reads_df_no_top_genes) + 1)

    line_with, = ax.plot(x_top, total_uniq_reads_df_top["cum_dist"].values,
                         linewidth=2.5, color=COLORS['primary'], marker='o', markersize=6, zorder=3)
    line_without, = ax.plot(x_no_top, total_uniq_reads_df_no_top_genes["cum_dist"].values,
                            linewidth=2.5, color=COLORS['secondary'], marker='o', markersize=6, zorder=3)

    ax.set_xlabel('Gene Rank', fontweight='bold')
    ax.set_ylabel('Cumulative Fraction of Reads', fontweight='bold')
    ax.set_xlim(1, max(len(total_uniq_reads_df_top), len(total_uniq_reads_df_no_top_genes)))
    ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=0.3)

    ax.legend([line_with, line_without],
              ['With hemoglobin', 'Without hemoglobin'],
              loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/cumulative_reads_distribution_by_genes.png")
    plt.close()


def add_to_total_uniq_reads(total_uniq_reads: dict[str, int], gene_read_counts_dict: dict[str, int]) -> dict[str, int]:
    """
    Accumulate per-gene unique read counts into a running total dictionary.

    :param total_uniq_reads: Running total of unique reads per gene.
    :param gene_read_counts_dict: Per-gene unique read counts from one sample.
    :return: Updated running total dictionary.
    """
    for gene_name, uniq_reads in gene_read_counts_dict.items():
        if gene_name not in total_uniq_reads:
            total_uniq_reads[gene_name] = 0
        total_uniq_reads[gene_name] += uniq_reads

    return total_uniq_reads


def convert_to_df(total_uniq_reads: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a gene-to-read-count dictionary into sorted DataFrames with
    cumulative distribution values.

    :param total_uniq_reads: Dictionary mapping gene names to total unique read counts.
    :return: A tuple of (top_10_df, top_20_df) sorted by descending read count
        with a 'cum_dist' column.
    """
    total_uniq_reads_df = pd.DataFrame(list(total_uniq_reads.items()),
                                       columns=['gene_name', 'total_uniq_reads'])
    total_uniq_reads_df = total_uniq_reads_df.sort_values(by='total_uniq_reads',
                                                          ascending=False)
    total_uniq_reads_df["cum_dist"] = list(
        total_uniq_reads_df['total_uniq_reads'].cumsum() /
        total_uniq_reads_df['total_uniq_reads'].sum())
    return total_uniq_reads_df.head(10), total_uniq_reads_df.head(20)



def get_read_count_distribution_by_gene(samples: list[str]) -> None:
    """
    Compute and plot the cumulative read count distribution by gene,
    both with and without hemoglobin genes, for proband samples only.

    :param samples: List of sample IDs to process (only '_3_R1' suffixed samples are used).
    """
    total_uniq_reads = {}
    total_uniq_reads_no_top_genes = {}

    for cur_sample in samples:
        if not cur_sample.endswith("_3_R1"):
            continue
        cur_annotated_transcripts = read_sqanti3_annotated(cur_sample, rules_filter=False)

        cur_annotated_transcripts = cur_annotated_transcripts[~(cur_annotated_transcripts["gene_type"] == "novel")]

        cur_annotated_transcripts["interval"] = cur_annotated_transcripts.apply(
            lambda row: f"{row['chrom']}:{row['start']}-{row['end']}",
            axis=1)
        cur_annotated_transcripts["is_hemo"] = cur_annotated_transcripts["interval"].map(
            lambda x: any([intersect(x, hemo_coord)
                        for hemo_coord in GENES_TO_REMOVE])
                        )
        cur_annotated_transcripts_no_top_genes = cur_annotated_transcripts[cur_annotated_transcripts["is_hemo"] == False]

        gene_read_counts_dict = get_gene_read_counts_dict(cur_annotated_transcripts)
        gene_read_counts_dict_no_top_genes = get_gene_read_counts_dict(cur_annotated_transcripts_no_top_genes)
        total_uniq_reads = add_to_total_uniq_reads(total_uniq_reads, gene_read_counts_dict)
        total_uniq_reads_no_top_genes = add_to_total_uniq_reads(total_uniq_reads_no_top_genes, gene_read_counts_dict_no_top_genes)


    total_uniq_reads_df_top, total_uniq_reads_df_top_20 = convert_to_df(total_uniq_reads)
    total_uniq_reads_df_no_top_genes, total_uniq_reads_df_no_top_genes_20 = convert_to_df(total_uniq_reads_no_top_genes)

    plot_read_count_distribution_by_gene(total_uniq_reads_df_top, total_uniq_reads_df_no_top_genes)
    total_uniq_reads_df_top_20 = total_uniq_reads_df_top_20.reset_index()
    total_uniq_reads_df_no_top_genes_20 = total_uniq_reads_df_no_top_genes_20.reset_index()
    top_20_genes_df = pd.DataFrame({
        "pre_hemo_depletion": total_uniq_reads_df_top_20["gene_name"],
        "post_hemo_depletion": total_uniq_reads_df_no_top_genes_20["gene_name"],
    })
    logger.info(top_20_genes_df)
    logger.info(total_uniq_reads_df_top_20)
    logger.info(total_uniq_reads_df_no_top_genes_20)
    top_20_genes_df.to_csv(f"{DATA_DIR}/top_20_genes_by_total_uniq_reads.tsv", index=False)



def main() -> None:
    """
    Entry point that loads long-read sample IDs and generates cumulative
    read count distribution plots.
    """
    lr_sample_ids = get_long_read_sample_ids()
    get_read_count_distribution_by_gene(lr_sample_ids)


if __name__ == "__main__":
    main()