"""Annotate SQANTI3 classification tables with gene names and expression values."""

import os
import json
from typing import Optional

import pandas as pd

from rare_disease_lr_rnaseq.utils import get_long_read_sample_ids, read_quant_expr, read_gtf, \
    read_sqanti3_filtered, map_gene_ids_to_gencode_gene_name_gtf, DATA_DIR

import logging

logger = logging.getLogger(__name__)


def annotate_expr(sample_id: str, sqanti3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate transcripts with expression data via inner merge.

    Performs an inner merge between the SQANTI3 classification DataFrame and
    quantification expression data, filtering to transcripts with at least
    2 unique reads and removing artifacts.

    :param sample_id: Sample identifier used to load quantification expression data.
    :param sqanti3_df: SQANTI3 classification DataFrame with an 'isoform' column.
    :return: Merged DataFrame containing SQANTI3 annotations joined with expression data.
    """
    quant_expr_df = read_quant_expr(sample_id, min_uniq_reads=2)
    # Use inner merge:
    # 1. Remove transcripts with less than 2 unique reads
    # 2. Remove transcripts identified as artifacts by SQANTI3
    annotated_df = sqanti3_df.merge(quant_expr_df,
                                    left_on="isoform",
                                    right_on="transcript_id",
                                    how="inner")
    return annotated_df


def load_gene_id_map() -> dict[str, list[str]]:
    """
    Load the gene-ID-to-name/type mapping, parsing the GTF at most once.

    Returns a cached JSON mapping if available; otherwise, parses the full
    GENCODE GTF, writes the cache, and returns the result.

    :return: Dictionary mapping gene ID to a two-element list of [gene_name, gene_type].
    """
    cache_path = f"{DATA_DIR}/gene_id_to_gene_name_map.json"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)

    gtf_file_path = f"{DATA_DIR}/gencode.v47.annotation.gtf.gz"
    logger.info("Parsing GTF to build gene-ID map (one-time)...")
    gene_id_map = map_gene_ids_to_gencode_gene_name_gtf([], gtf_file_path)

    with open(cache_path, 'w') as f:
        json.dump(gene_id_map, f)
    logger.info("Wrote gene-ID map cache to %s", cache_path)

    return gene_id_map


def main(rules_filter: bool = True, min_uniq_reads: int = 2) -> None:
    """
    Build and save fully annotated transcript DataFrames for all samples.

    For each sample: filters SQANTI3 artifacts, merges expression data,
    annotates gene names/types from GENCODE, and adds genomic coordinates
    from the LRAA GTF. Saves per-sample annotated CSVs and caches the
    gene ID to gene name mapping as JSON.

    :param rules_filter: If True, use rules-filtered SQANTI3 classification; otherwise use ML-filtered classification.
    :param min_uniq_reads: Minimum number of unique reads required for a transcript to be retained during expression annotation.
    """
    sample_ids = get_long_read_sample_ids()
    gene_id_to_gene_name_map = load_gene_id_map()

    for cur_sample in sample_ids:
        logger.info(cur_sample)
        if rules_filter:
            logger.info("Using rules filtered SQANTI3 classification file")
            sqanti3_df = read_sqanti3_filtered(cur_sample)
        else:
            logger.info("Using ML filtered SQANTI3 classification file")
            sqanti3_df = read_sqanti3_filtered(cur_sample, rules_filter=False)

        sqanti3_df = annotate_expr(cur_sample, sqanti3_df)

        sqanti3_df["gene_name"] = sqanti3_df["associated_gene"].map(
            lambda x: gene_id_to_gene_name_map.get(x.split(".")[0], ["unknown", "unknown"])[0])
        sqanti3_df["gene_type"] = sqanti3_df["associated_gene"].map(
            lambda x: gene_id_to_gene_name_map.get(x.split(".")[0], ["unknown", "unknown"])[1])

        gtf_df = read_gtf(cur_sample)
        tx_gtf_df = gtf_df[gtf_df["feature"] == "transcript"]
        shape_before_merge = sqanti3_df.shape[0]
        sqanti3_df = sqanti3_df.merge(tx_gtf_df[["transcript_id", "chrom", "start", "end", "strand"]],
                                      left_on="isoform",
                                      right_on="transcript_id",
                                      how="left",
                                      suffixes=("", "_gtf"))
        sqanti3_df = sqanti3_df.drop(columns=["transcript_id_gtf"])

        shape_after_merge = sqanti3_df.shape[0]
        if shape_before_merge != shape_after_merge:
            raise ValueError(f"Merge changed number of rows from {shape_before_merge} to {shape_after_merge}")

        if rules_filter:
            sqanti3_df.to_csv(f"{DATA_DIR}/annotated_sqanti3"
                              f"/{cur_sample}_annotated_transcripts.csv",
                              index=False)
        else:
            sqanti3_df.to_csv(f"{DATA_DIR}/annotated_sqanti3"
                              f"/{cur_sample}_ml_annotated_transcripts.csv",
                              index=False)


if __name__ == "__main__":
    main()