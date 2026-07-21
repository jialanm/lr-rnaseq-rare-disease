"""Count and plot the number of transcript isoforms per gene across samples."""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rare_disease_lr_rnaseq.utils import get_long_read_sample_ids, read_sqanti3_filtered, DATA_DIR

logger = logging.getLogger(__name__)

def categorize_isoform_count(count: int) -> str:
    """
    Categorize isoform count into bins.

    :param count: Number of isoforms for a gene.
    :return: Category label for the isoform count.
    """
    if count == 1:
        return "1 isoform"
    elif count <= 3:
        return "2-3 isoforms"
    elif count <= 5:
        return "4-5 isoforms"
    else:
        return ">=6 isoforms"


def get_isoform_counts_per_sample(sample_id: str) -> pd.Series:
    """
    Get number of isoforms per gene for a single sample.

    :param sample_id: Identifier for the sample to process.
    :return: Series indexed by gene name with isoform counts as values.
    """
    sqanti3_df = read_sqanti3_filtered(sample_id)
    isoforms_per_gene = sqanti3_df.groupby("associated_gene").size()
    return isoforms_per_gene


def main() -> None:
    """
    Compute and plot average number of genes by isoform count categories
    across all long-read samples.
    """
    sample_ids = get_long_read_sample_ids()

    all_category_counts = []
    category_order = ["1 isoform", "2-3 isoforms", "4-5 isoforms", ">=6 isoforms"]

    for sample_id in sample_ids:
        logger.info(f"Processing {sample_id}...")
        isoforms_per_gene = get_isoform_counts_per_sample(sample_id)

        categories = isoforms_per_gene.apply(categorize_isoform_count)
        category_counts = categories.value_counts()

        for cat in category_order:
            if cat not in category_counts:
                category_counts[cat] = 0

        all_category_counts.append(category_counts[category_order])

    counts_df = pd.DataFrame(all_category_counts, columns=category_order)
    avg_counts = counts_df.mean()
    std_counts = counts_df.std()

    _, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(category_order))
    bars = ax.bar(x, avg_counts, yerr=std_counts, capsize=5,
                  color='#5B9BD5', edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Isoform Categories", fontsize=12)
    ax.set_ylabel("Average Number of Genes", fontsize=12)
    ax.set_title("Average Number of Genes by Isoform Categories", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(category_order)

    for bar, avg, std in zip(bars, avg_counts, std_counts):
        height = bar.get_height()
        ax.annotate(f'{avg:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 50),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/isoforms_per_gene_categories.png", dpi=300)
    plt.savefig(f"{DATA_DIR}/isoforms_per_gene_categories.pdf")
    plt.show()

    logger.info("\n=== Summary Statistics ===")
    logger.info(f"Number of samples: {len(sample_ids)}")
    logger.info("\nAverage counts per category:")
    for cat, avg, std in zip(category_order, avg_counts, std_counts):
        logger.info(f"  {cat}: {avg:.1f} ± {std:.1f}")


if __name__ == "__main__":
    main()