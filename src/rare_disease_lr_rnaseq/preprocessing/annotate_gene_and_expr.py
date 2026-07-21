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


def map_gene_id_to_gene_name_and_type(gene_ids: set[str]) -> dict[str, list[str]]:
    """
    Map gene IDs to gene names and gene types using a GENCODE GTF file.

    Loads a cached JSON mapping if available; otherwise, builds the mapping
    from the GTF file.

    :param gene_ids: Set of gene IDs (without version suffix) to map.
    :return: Dictionary mapping gene ID to a two-element list of [gene_name, gene_type].
    """
    gene_id_to_gene_name_map_json_filepath = f"{DATA_DIR}/gene_id_to_gene_name_map.json"
    if os.path.exists(gene_id_to_gene_name_map_json_filepath):
        with open(gene_id_to_gene_name_map_json_filepath, 'r') as f:
            gene_id_to_gene_name_map = json.load(f)
        return gene_id_to_gene_name_map

    gtf_file_path = f"{DATA_DIR}/gencode.v47.annotation.gtf.gz"
    gene_id_to_gene_name_map = map_gene_ids_to_gencode_gene_name_gtf(gene_ids, gtf_file_path)
    return gene_id_to_gene_name_map


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

    gene_id_to_gene_name_map_json_filepath = f"{DATA_DIR}/gene_id_to_gene_name_map.json"
    save_gene_id_to_gene_name_map = False
    if not os.path.exists(gene_id_to_gene_name_map_json_filepath):
        save_gene_id_to_gene_name_map = True
        all_gene_id_to_gene_name_map = {}

    for cur_sample in sample_ids:
        logger.info(cur_sample)
        if rules_filter:
            logger.info("Using rules filtered SQANTI3 classification file")
            sqanti3_df = read_sqanti3_filtered(cur_sample)
        else:
            logger.info("Using ML filtered SQANTI3 classification file")
            sqanti3_df = read_sqanti3_filtered(cur_sample, rules_filter=False)

        sqanti3_df = annotate_expr(cur_sample, sqanti3_df)
        cur_gene_ids = set(
            sqanti3_df["associated_gene"].map(lambda x: x.split(".")[0]).tolist())
        gene_id_to_gene_name_map = map_gene_id_to_gene_name_and_type(cur_gene_ids)
        if save_gene_id_to_gene_name_map:
            all_gene_id_to_gene_name_map.update(gene_id_to_gene_name_map)

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

    if save_gene_id_to_gene_name_map:
        with open(gene_id_to_gene_name_map_json_filepath, 'w') as f:
            json.dump(all_gene_id_to_gene_name_map, f)


if __name__ == "__main__":
    main()