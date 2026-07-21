"""Generate a per-sample summary table with read counts, RIN, and isoform statistics."""

import pandas as pd
import numpy as np
import math
from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids, \
    read_sqanti3_annotated, get_unique_tx

import logging

logger = logging.getLogger(__name__)

"""
Sample table containing the following columns:
sample_id
total_readcount
RIN
total_transcripts_count
unqiue_transcripts_count

* can include more columns for structural categories
* ['internal_fragment' 'at_least_one_novel_splicesite' 'reference_match'
 'alternative_3end5end' '3prime_fragment' 'combination_of_known_junctions'
 'intron_retention' 'combination_of_known_splicesites' 'alternative_3end'
 '5prime_fragment' 'alternative_5end' 'multi-exon']
"""


def get_metadata() -> pd.DataFrame:
    """
    Load and format sample metadata from the metadata TSV file.

    Reads the metadata file, renames columns for consistency, selects
    relevant columns, and appends '_R1' to sample IDs.

    :return: DataFrame with columns 'sample_id', 'RIN', and 'total_readcount'.
    """
    md_df = pd.read_csv(f"{DATA_DIR}/metadata.tsv", sep='\t')
    logger.info(md_df.columns)
    md_df = md_df.rename(columns={"entity:Sample_ID":"sample_id", "Total_readcount": "total_readcount"})
    use_cols = ["sample_id", "RIN", "total_readcount"]
    md_df = md_df[use_cols]
    md_df["sample_id"] = md_df["sample_id"].apply(lambda x: f"{x}_R1")
    return md_df


def main() -> None:
    """
    Build and save summary data tables for all samples and probands.

    Loads metadata, computes unique and total transcript counts per
    sample, merges them, and saves the combined tables as TSV files
    for all samples and for probands only.
    """
    lr_sample_ids = get_long_read_sample_ids()
    metadata = get_metadata()

    uniq_tx = get_unique_tx(lr_sample_ids, rule_filter=False)
    uniq_tx_count_df = pd.DataFrame([(k, len(v)) for k, v in uniq_tx.items()],
                                    columns=["sample_id", "unique_transcripts_count"])
    metadata = metadata.merge(uniq_tx_count_df,
                              on="sample_id",
                              how="left")
    logger.info(metadata)

    total_tx = {}
    for cur_sample in lr_sample_ids:
        cur_sample_table = read_sqanti3_annotated(cur_sample, rules_filter=False)
        cur_total_transcripts = cur_sample_table.shape[0]
        total_tx[cur_sample] = cur_total_transcripts

    total_tx_count_df = pd.DataFrame([(k, v) for k, v in total_tx.items()],
                                     columns=["sample_id", "total_transcripts_count"])
    metadata = metadata.merge(total_tx_count_df,
                              on="sample_id",
                              how="left")
    metadata_probands = metadata[metadata["sample_id"].str.endswith("_3_R1")]
    logger.info(metadata)
    logger.info(metadata_probands)
    metadata.to_csv(f"{DATA_DIR}/summary_data_table.tsv", index=False)
    metadata_probands.to_csv(f"{DATA_DIR}/summary_data_table_probands.tsv", index=False)
    

if __name__ == "__main__":
    main()