"""Generate read-length distribution plots comparing pre- and post-depletion samples."""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from rare_disease_lr_rnaseq.utils import DATA_DIR

import logging

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def aggregate_binned_dats(dats: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate multiple binned read-length DataFrames into one.

    Concatenates all DataFrames and sums counts per read-length bin.

    :param dats: List of DataFrames, each with 'read_length' and 'count' columns.
    :return: Aggregated DataFrame with 'read_length' and 'count' columns, sorted by read_length.
    """
    return (
        pd.concat(dats, ignore_index=True)
          .groupby('read_length', as_index=False)['count']
          .sum()
          .sort_values('read_length')
    )


def bin_by_read_length(dat: pd.DataFrame, bin_size: int = 50, length_limit: int = 3000) -> pd.DataFrame:
    """
    Bin read-length counts into fixed-width bins.

    Filters out reads longer than ``length_limit``, then groups counts
    into bins of width ``bin_size``.

    :param dat: DataFrame with 'read_length' and 'count' columns.
    :param bin_size: Width of each read-length bin in bases.
    :param length_limit: Maximum read length to include.
    :return: Binned DataFrame with 'read_length' (bin start) and 'count' columns.
    """
    dat = dat[dat["read_length"] <= length_limit]

    bin_labels = (dat['read_length'] // bin_size) * bin_size
    dat = dat.groupby(bin_labels)['count'].sum().reset_index()
    dat.columns = ['read_length', 'count']

    return dat


def plot_rl(dat: pd.DataFrame, dat_no_top_genes: pd.DataFrame) -> None:
    """
    Plot read-length distributions with and without top-expressed genes.

    Displays a bar chart of the full distribution overlaid with line
    plots comparing the distribution with and without hemoglobin genes.

    :param dat: Binned read-length DataFrame including all genes, with 'read_length' and 'count' columns.
    :param dat_no_top_genes: Binned read-length DataFrame excluding top-expressed genes (hemoglobins), with 'read_length' and 'count' columns.
    """
    plt.figure(figsize=(10, 8))
    plt.bar(dat['read_length'], dat['count'], width=1.0, )
    plt.plot(dat['read_length'], dat['count'], linewidth=2, label="With Hemoglobins")
    plt.plot(dat_no_top_genes['read_length'], dat_no_top_genes['count'], linewidth=2,
             label="Without Hemoglobins")
    plt.xlabel('Read Length')
    plt.ylabel('Count')
    plt.title('Read Length Distribution')
    plt.xlim(0, 3200)
    plt.legend()
    plt.show()


def agg_rl(filepaths: list[str]) -> pd.DataFrame:
    """
    Read, bin, and aggregate read-length files.

    Each file is a tab-separated file with two columns (read_length, count).
    Files are individually binned and then aggregated into a single DataFrame.

    :param filepaths: List of file paths to tab-separated read-length distribution files.
    :return: Aggregated and binned DataFrame with 'read_length' and 'count' columns.
    """
    dats = []
    for cur_filepath in filepaths:
        dat = pd.read_csv(cur_filepath, sep="\t", header=None)
        dat.columns = ["read_length", "count"]
        dat = bin_by_read_length(dat)
        dats.append(dat)
    agg_dat = aggregate_binned_dats(dats)

    return agg_dat


def main() -> None:
    """
    Generate and plot aggregated read-length distributions.

    Reads all read-length distribution files from the data directory,
    aggregates them with and without top-expressed genes, plots the
    comparison, and logs the average and median read lengths.
    """
    data_dir = f"{DATA_DIR}/rl_distribution"
    rl_filepaths = []
    no_top_genes_rl_filepaths = []
    for cur_file in os.listdir(data_dir):
        cur_filepath = os.path.join(data_dir, cur_file)
        if "no_top_genes" in cur_file:
            no_top_genes_rl_filepaths.append(cur_filepath)
        elif cur_file.endswith("_read_lengths.txt"):
            rl_filepaths.append(cur_filepath)
    
    agg_dat = agg_rl(rl_filepaths)
    no_top_genes_agg_dat = agg_rl(no_top_genes_rl_filepaths)
    plot_rl(agg_dat, no_top_genes_agg_dat)
    logger.info(no_top_genes_agg_dat)
    avg_read_length = (no_top_genes_agg_dat["read_length"] * no_top_genes_agg_dat["count"]).sum() / no_top_genes_agg_dat["count"].sum()
    logger.info(avg_read_length)

    cumsum = no_top_genes_agg_dat["count"].cumsum()
    total = no_top_genes_agg_dat["count"].sum()
    median_idx = (cumsum >= total / 2).idxmax()
    median_read_length = no_top_genes_agg_dat.loc[median_idx, "read_length"]
    logger.info(median_read_length)


if __name__ == "__main__":
    main()