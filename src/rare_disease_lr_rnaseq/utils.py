"""Shared utilities for reading, filtering, and annotating long-read RNA-seq data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
import json
import gzip
import logging
from typing import Any

logger = logging.getLogger(__name__)

from rare_disease_lr_rnaseq.config import DATA_DIR


def create_symbolic_links(job: Any, path: str, link_path: str) -> None:
    """
    Copy a file from a GCS path to a local path using gcloud storage.

    :param job: Hail Batch job object on which the command will be executed.
    :param path: Source GCS path to copy from.
    :param link_path: Destination local path to copy to.
    """
    job.command(f"""
        gcloud storage cp '{path}' '{link_path}'
    """)


def get_exon_signature(exon_df: pd.DataFrame) -> tuple[tuple, ...]:
    """
    Get exon signature from DataFrame with 'chrom', 'start', 'end' columns.

    :param exon_df: DataFrame containing at least 'chrom', 'start', and 'end' columns representing exon coordinates.
    :return: Tuple of (chrom, start, end) tuples representing the exon signature.
    """
    return tuple(map(tuple, exon_df[['chrom', 'start', 'end']].values))


def map_tx_to_exon_signature(transcript_id: str, gtf: pd.DataFrame) -> tuple[tuple, ...]:
    """
    Map a transcript ID to its exon signature using a GTF DataFrame.

    :param transcript_id: Transcript identifier to look up.
    :param gtf: GTF DataFrame with 'transcript_id', 'feature', 'chrom', 'start', and 'end' columns.
    :return: Tuple of (chrom, start, end) tuples representing the exon signature.
    """
    cur_gtf = gtf[gtf["transcript_id"] == transcript_id]
    exon_df = cur_gtf[cur_gtf["feature"] == "exon"]
    return get_exon_signature(exon_df)


def compute_transcript_features_vectorized(gtf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute intron signature and terminal positions using vectorized operations.

    Excludes mono-exon transcripts.

    :param gtf_df: GTF DataFrame with columns: chrom, feature, start, end, strand, transcript_id.
    :return: DataFrame with columns: transcript_id, chrom, strand, intron_signature, tss, pas, length.
    """
    exon_gtf = gtf_df[gtf_df["feature"] == "exon"].copy()

    if exon_gtf.empty:
        return pd.DataFrame(columns=['transcript_id', 'chrom', 'strand', 'intron_signature', 'tss', 'pas', 'length'])

    exon_gtf = exon_gtf.sort_values(['transcript_id', 'start']).reset_index(drop=True)

    exon_counts = exon_gtf.groupby('transcript_id', sort=False).size()
    multi_exon_tx = exon_counts[exon_counts >= 2].index

    exon_gtf = exon_gtf[exon_gtf['transcript_id'].isin(multi_exon_tx)].reset_index(drop=True)

    if exon_gtf.empty:
        return pd.DataFrame(columns=['transcript_id', 'chrom', 'strand', 'intron_signature', 'tss', 'pas', 'length'])

    # Mark last exon per transcript (boundary detection)
    tx_ids = exon_gtf['transcript_id'].values
    is_last = np.concatenate([tx_ids[:-1] != tx_ids[1:], [True]])

    # Compute introns: intron_start = exon_end, intron_end = next_exon_start
    # Only for non-last exons within each transcript
    exon_ends = exon_gtf['end'].values
    exon_starts = exon_gtf['start'].values
    next_starts = np.roll(exon_starts, -1)

    intron_mask = ~is_last
    intron_tx_ids = tx_ids[intron_mask]
    intron_starts_arr = exon_ends[intron_mask]
    intron_ends_arr = next_starts[intron_mask]

    # Build intron signature per transcript using efficient string aggregation
    if len(intron_tx_ids) > 0:
        intron_pairs = [f"{s},{e}" for s, e in zip(intron_starts_arr, intron_ends_arr)]
        intron_df = pd.DataFrame({
            'transcript_id': intron_tx_ids,
            'intron_pair': intron_pairs,
        })
        intron_sig_strs = intron_df.groupby('transcript_id', sort=False)['intron_pair'].agg('|'.join)

        def parse_intron_sig(s: str) -> tuple[tuple[int, ...], ...]:
            """
            Parse a pipe-delimited intron signature string.

            :param s: Pipe-delimited intron pairs, e.g. "100,200|300,400".
            :return: Tuple of (start, end) integer tuples.
            """
            return tuple(tuple(map(int, p.split(','))) for p in s.split('|'))

        intron_sigs = intron_sig_strs.apply(parse_intron_sig)
        intron_sigs.name = 'intron_signature'
    else:
        intron_sigs = pd.Series(name='intron_signature', dtype=object)

    # Compute exon lengths first, then aggregate (avoid slow lambda)
    exon_gtf = exon_gtf.copy()
    exon_gtf['exon_length'] = exon_gtf['end'] - exon_gtf['start']

    agg_df = exon_gtf.groupby('transcript_id', sort=False).agg(
        chrom=('chrom', 'first'),
        strand=('strand', 'first'),
        genomic_start=('start', 'first'),
        genomic_end=('end', 'last'),
        length=('exon_length', 'sum')
    ).reset_index()

    # TSS/PAS swap based on strand: plus strand TSS=start, minus strand TSS=end
    plus_mask = agg_df['strand'] == '+'
    agg_df['tss'] = np.where(plus_mask, agg_df['genomic_start'], agg_df['genomic_end'])
    agg_df['pas'] = np.where(plus_mask, agg_df['genomic_end'], agg_df['genomic_start'])

    features_df = agg_df.merge(intron_sigs, on='transcript_id', how='left')
    features_df = features_df[['transcript_id', 'chrom', 'strand', 'intron_signature', 'tss', 'pas', 'length']]

    return features_df


def get_all_tx(sample_ids: list[str], rule_filter: bool = True) -> dict[str, pd.DataFrame]:
    """
    Get transcripts for each sample.

    :param sample_ids: List of sample identifiers.
    :param rule_filter: Whether to use the rules-filtered SQANTI3 classification file (True) or the ML-filtered SQANTI3 classification file (False).
    :return: Mapping of sample ID to its SQANTI3 annotated transcript DataFrame.
    """
    tx = {}
    for cur_sample in sample_ids:
        logger.info(cur_sample)
        if rule_filter:
            logger.info("Using rules filtered SQANTI3 classification file")
        else:
            logger.info("Using ML filtered SQANTI3 classification file")
        sqanti3_df = read_sqanti3_annotated(cur_sample, rules_filter=rule_filter)

        tx[cur_sample] = sqanti3_df

    return tx


def _greedy_cluster(group_df: pd.DataFrame, tss_tolerance: int, pas_tolerance: int) -> pd.DataFrame:
    """
    Apply greedy seed-based clustering on a DataFrame group.

    :param group_df: DataFrame of transcripts sharing the same intron chain, with columns: tss, pas, length.
    :param tss_tolerance: Maximum allowed TSS difference in base pairs.
    :param pas_tolerance: Maximum allowed PAS difference in base pairs.
    :return: Input DataFrame with an additional 'cluster_id' column.
    """
    if group_df.empty:
        return group_df

    n = len(group_df)

    if n == 1:
        group_df = group_df.copy()
        group_df['cluster_id'] = 0
        return group_df

    sorted_df = group_df.sort_values('length', ascending=False).reset_index(drop=True)

    cluster_ids = np.full(n, -1, dtype=np.int32)
    tss_arr = sorted_df['tss'].values.astype(np.int64)
    pas_arr = sorted_df['pas'].values.astype(np.int64)
    current_cluster = 0

    for i in range(n):
        if cluster_ids[i] >= 0:
            continue

        seed_tss = tss_arr[i]
        seed_pas = pas_arr[i]

        unclustered_mask = cluster_ids < 0
        if not unclustered_mask.any():
            break

        tss_diff = np.abs(tss_arr - seed_tss)
        pas_diff = np.abs(pas_arr - seed_pas)
        match_mask = unclustered_mask & (tss_diff <= tss_tolerance) & (pas_diff <= pas_tolerance)
        cluster_ids[match_mask] = current_cluster

        current_cluster += 1

    sorted_df['cluster_id'] = cluster_ids
    return sorted_df


def _compute_features_cached(
    gtf_df: pd.DataFrame, sample_id: str, cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute intron signature and terminal positions, with per-sample caching.

    Delegates to :func:`compute_transcript_features_vectorized` and stores the
    result in *cache* so repeated calls for the same sample are free.

    :param gtf_df: GTF DataFrame with columns: transcript_id, feature, start, end, chrom, strand.
    :param sample_id: Sample identifier used as cache key.
    :param cache: Mutable dict used as the cache store.
    :return: DataFrame with columns: transcript_id, chrom, strand, intron_signature, tss, pas, length.
    """
    if sample_id in cache:
        return cache[sample_id]
    features_df = compute_transcript_features_vectorized(gtf_df)
    cache[sample_id] = features_df
    return features_df


def _cache_path(prefix: str, rule_filter: bool, tss_tolerance: int, pas_tolerance: int) -> str:
    """
    Build a parameter-aware cache filename.

    :param prefix: Base name for the cache file (e.g. 'unique_transcripts_intron_terminal').
    :param rule_filter: Whether rules-filtered SQANTI3 data was used.
    :param tss_tolerance: TSS tolerance in base pairs.
    :param pas_tolerance: PAS tolerance in base pairs.
    :return: Full path to the cache JSON file.
    """
    filt = "rules" if rule_filter else "ml"
    return f"{DATA_DIR}/{prefix}_{filt}_tss{tss_tolerance}_pas{pas_tolerance}.json"


def _find_unique_tx(
    sample_ids: list[str],
    group_col: str,
    output_filepath: str,
    rule_filter: bool,
    tss_tolerance: int,
    pas_tolerance: int,
    force_recompute: bool,
) -> dict[str, pd.DataFrame]:
    """
    Shared implementation for transcript uniqueness analysis.

    Clusters transcripts by intron chain and terminal proximity, then marks a
    transcript as unique if its cluster appears in only one group (defined by
    *group_col*).

    :param sample_ids: List of sample identifiers.
    :param group_col: Column used to define uniqueness groups ('trio_id' or 'sample_id').
    :param output_filepath: Path where the cached JSON result is stored.
    :param rule_filter: Whether to use the rules-filtered SQANTI3 classification file.
    :param tss_tolerance: Maximum allowed TSS difference in base pairs for clustering.
    :param pas_tolerance: Maximum allowed PAS difference in base pairs for clustering.
    :param force_recompute: If True, delete the cached result and recompute.
    :return: Mapping of sample ID to DataFrame of unique transcripts.
    """
    uniqueness_col = 'unique_tx' if group_col == 'trio_id' else 'individual_unique_tx'

    if force_recompute and os.path.exists(output_filepath):
        logger.info(f"Force recompute: deleting cache {output_filepath}")
        os.remove(output_filepath)

    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            serialized_tx = json.load(f)
            uniq_tx = {sid: pd.DataFrame(records)
                       for sid, records in serialized_tx.items()}
            uniq_tx = {sid: df for sid, df in uniq_tx.items() if sid in sample_ids}
            return uniq_tx

    tx_df_cache: dict[str, pd.DataFrame] = {}
    gtf_cache: dict[str, pd.DataFrame] = {}
    features_df_cache: dict[str, pd.DataFrame] = {}

    def get_cached_data(sample_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if sample_id not in tx_df_cache:
            tx_df_cache[sample_id] = read_sqanti3_annotated(sample_id, rule_filter)
            gtf_cache[sample_id] = read_gtf(sample_id)
        return tx_df_cache[sample_id], gtf_cache[sample_id]

    # --- Build feature table for every sample ---
    all_features_list: list[pd.DataFrame] = []

    if group_col == 'trio_id':
        trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])
        for cur_trio in trio_ids:
            logger.info(f"Processing trio {cur_trio}")
            for count in range(1, 4):
                cur_sample = f"{cur_trio}_{count}_R1"
                logger.info(f"  {cur_sample}")

                cur_tx_df, cur_gtf = get_cached_data(cur_sample)
                features_df = _compute_features_cached(cur_gtf, cur_sample, features_df_cache)

                if features_df.empty:
                    logger.info(f"    WARNING: No multi-exon transcripts found in GTF")
                    continue

                valid_tx = set(cur_tx_df['isoform'])
                features_df = features_df[features_df['transcript_id'].isin(valid_tx)].copy()

                if features_df.empty:
                    logger.info(f"    WARNING: No transcripts matched between GTF and tx_df")
                    continue

                features_df['sample_id'] = cur_sample
                features_df['trio_id'] = cur_trio
                all_features_list.append(features_df)
    else:
        for cur_sample in sample_ids:
            logger.info(f"Processing sample {cur_sample}")

            cur_tx_df, cur_gtf = get_cached_data(cur_sample)
            features_df = _compute_features_cached(cur_gtf, cur_sample, features_df_cache)

            if features_df.empty:
                logger.info(f"    WARNING: No multi-exon transcripts found in GTF")
                continue

            valid_tx = set(cur_tx_df['isoform'])
            features_df = features_df[features_df['transcript_id'].isin(valid_tx)].copy()

            if features_df.empty:
                logger.info(f"    WARNING: No transcripts matched between GTF and tx_df")
                continue

            features_df['sample_id'] = cur_sample
            all_features_list.append(features_df)

    if not all_features_list:
        logger.info("WARNING: No features found for any transcripts.")
        empty_result = {sid: pd.DataFrame() for sid in sample_ids}
        with open(output_filepath, "w") as f:
            json.dump({sid: [] for sid in sample_ids}, f, indent=2)
        return empty_result

    logger.info(f"Collected {len(all_features_list)} feature DataFrames")
    all_features_df = pd.concat(all_features_list, ignore_index=True)

    # --- Cluster ---
    all_features_df['intron_sig_str'] = all_features_df['intron_signature'].astype(str)

    group_keys, _ = pd.factorize(
        all_features_df['chrom'].astype(str) + '|' +
        all_features_df['strand'].astype(str) + '|' +
        all_features_df['intron_sig_str']
    )
    all_features_df['group_key'] = group_keys

    logger.info("Applying greedy clustering...")
    clustered_df = all_features_df.groupby('group_key', group_keys=False, sort=False).apply(
        lambda g: _greedy_cluster(g, tss_tolerance, pas_tolerance)
    ).reset_index(drop=True)

    max_clusters = clustered_df['cluster_id'].max() + 1 if len(clustered_df) > 0 else 1
    clustered_df['global_cluster_id'] = (
        clustered_df['group_key'].astype(np.int64) * max_clusters +
        clustered_df['cluster_id'].astype(np.int64)
    )

    # --- Mark uniqueness ---
    group_counts = clustered_df.groupby('global_cluster_id', sort=False)[group_col].nunique()
    clustered_df['_group_count'] = clustered_df['global_cluster_id'].map(group_counts)
    clustered_df[uniqueness_col] = clustered_df['_group_count'] == 1

    # --- Build per-sample result ---
    clustered_df = clustered_df.set_index(['sample_id', 'transcript_id'])
    uniqueness_map = clustered_df[uniqueness_col]
    intron_sig_map = clustered_df['intron_sig_str']

    result: dict[str, pd.DataFrame] = {}
    for cur_sample in sample_ids:
        logger.info(f"Filtering unique for {cur_sample}")
        cur_tx_df, _ = get_cached_data(cur_sample)
        cur_tx_df = cur_tx_df.copy()

        cur_tx_df['_sample_id'] = cur_sample
        lookup_idx = pd.MultiIndex.from_arrays([
            cur_tx_df['_sample_id'],
            cur_tx_df['isoform']
        ])

        cur_tx_df[uniqueness_col] = uniqueness_map.reindex(lookup_idx).fillna(False).values
        cur_tx_df['intron_signature'] = intron_sig_map.reindex(lookup_idx).values
        cur_tx_df = cur_tx_df.drop(columns=['_sample_id'])

        result[cur_sample] = cur_tx_df[cur_tx_df[uniqueness_col]]

    serializable_tx = {sid: df.to_dict(orient='records') for sid, df in result.items()}
    with open(output_filepath, "w") as f:
        json.dump(serializable_tx, f, indent=2)

    return result


def get_unique_tx(sample_ids: list[str], rule_filter: bool = True, tss_tolerance: int = 15,
                  pas_tolerance: int = 25,
                  force_recompute: bool = False) -> dict[str, pd.DataFrame]:
    """
    Get trio-unique transcripts for each sample.

    A transcript is unique if its cluster (by intron chain and terminal
    proximity) appears in only one trio. Mono-exon transcripts are excluded.

    :param sample_ids: List of sample identifiers.
    :param rule_filter: Whether to use the rules-filtered SQANTI3 classification file.
    :param tss_tolerance: Maximum allowed TSS difference in base pairs for clustering.
    :param pas_tolerance: Maximum allowed PAS difference in base pairs for clustering.
    :param force_recompute: If True, delete the cached result and recompute.
    :return: Mapping of sample ID to DataFrame of unique transcripts.
    """
    return _find_unique_tx(
        sample_ids,
        group_col='trio_id',
        output_filepath=_cache_path("unique_transcripts_intron_terminal", rule_filter, tss_tolerance, pas_tolerance),
        rule_filter=rule_filter,
        tss_tolerance=tss_tolerance,
        pas_tolerance=pas_tolerance,
        force_recompute=force_recompute,
    )


def get_individual_unique_tx(sample_ids: list[str], rule_filter: bool = True,
                             tss_tolerance: int = 15, pas_tolerance: int = 25,
                             force_recompute: bool = False) -> dict[str, pd.DataFrame]:
    """
    Get individual-unique transcripts for each sample.

    A transcript is individual-unique if its cluster (by intron chain and
    terminal proximity) contains only one sample. Mono-exon transcripts are
    excluded.

    :param sample_ids: List of sample identifiers.
    :param rule_filter: Whether to use the rules-filtered SQANTI3 classification file.
    :param tss_tolerance: Maximum allowed TSS difference in base pairs for clustering.
    :param pas_tolerance: Maximum allowed PAS difference in base pairs for clustering.
    :param force_recompute: If True, delete the cached result and recompute.
    :return: Mapping of sample ID to DataFrame of individual-unique transcripts.
    """
    return _find_unique_tx(
        sample_ids,
        group_col='sample_id',
        output_filepath=_cache_path("individual_unique_transcripts", rule_filter, tss_tolerance, pas_tolerance),
        rule_filter=rule_filter,
        tss_tolerance=tss_tolerance,
        pas_tolerance=pas_tolerance,
        force_recompute=force_recompute,
    )


def get_long_read_sample_ids() -> list[str]:
    """
    Get long-read sample IDs from lr_sample_ids.txt.

    :return: List of sample identifiers read from the file.
    """
    sample_ids = pd.read_table(f"{DATA_DIR}/lr_sample_ids.txt",
                               header=None).iloc[:, 0].tolist()
    return sample_ids


def read_gtf(sample_id: str) -> pd.DataFrame:
    """
    Read a GTF file for a sample and extract relevant columns.

    :param sample_id: Sample identifier used to locate the GTF file.
    :return: DataFrame with columns: chrom, feature, start, end, strand, gene_id, transcript_id.
    """
    cur_gtf = pd.read_csv(f"{DATA_DIR}/tx_gtf/{sample_id}.LRAA.gtf",
                          sep='\t',
                          header=None,
                          dtype={0: 'category', 2: 'category', 6: 'category'})
    cur_gtf.columns = ["chrom", "source", "feature", "start", "end", "score",
                       "strand", "frame", "attributes"]

    extracted = cur_gtf['attributes'].str.extract(
        r'gene_id "(?P<gene_id>[^"]+)".*?transcript_id "(?P<transcript_id>[^"]+)"'
    )
    cur_gtf = pd.concat([cur_gtf[['chrom', 'feature', 'start', 'end', 'strand']], extracted], axis=1)

    return cur_gtf


def read_quant_expr(sample_id: str, min_uniq_reads: int = 2) -> pd.DataFrame:
    """
    Read quant.expr file and filter by minimum unique reads.

    :param sample_id: Sample identifier used to locate the expression file.
    :param min_uniq_reads: Minimum number of unique reads required to keep a transcript.
    :return: Filtered DataFrame with expression columns: gene_id, transcript_id, uniq_reads, all_reads, isoform_fraction, unique_gene_read_fraction, TPM.
    """
    cur_quant_expr = pd.read_csv(f"{DATA_DIR}/tx_expr"
                                 f"/{sample_id}.LRAA.quant.expr",
                                 sep='\t')
    use_expr_cols = ['gene_id', 'transcript_id', 'uniq_reads', 'all_reads',
                     'isoform_fraction', 'unique_gene_read_fraction', 'TPM']
    cur_quant_expr = cur_quant_expr[use_expr_cols]
    cur_quant_expr = cur_quant_expr[cur_quant_expr["uniq_reads"] >= min_uniq_reads]
    return cur_quant_expr


def read_sqanti3_annotated(sample_id: str, rules_filter: bool = True) -> pd.DataFrame:
    """
    Read an annotated SQANTI3 classification CSV file for a sample.

    :param sample_id: Sample identifier used to locate the classification file.
    :param rules_filter: If True, read the rules-filtered annotated file; if False, read the ML-filtered annotated file.
    :return: DataFrame of annotated SQANTI3 transcript classifications.
    """
    use_sqanti3_cols = ['isoform', 'chrom', 'start', 'end', 'strand', 'length', 'exons',
                        'structural_category',
                        'associated_gene', 'associated_transcript', 'subcategory',
                        'gene_type', 'gene_name', 'RTS_stage',
                        'predicted_NMD', 'all_canonical', 'uniq_reads',
                        'all_reads', 'isoform_fraction',
                        'unique_gene_read_fraction', 'TPM']

    # Specify dtypes for string columns to avoid mixed type warnings
    dtype_spec = {
        'isoform': str, 'chrom': str, 'strand': str,
        'structural_category': str, 'associated_gene': str,
        'associated_transcript': str, 'subcategory': str,
        'gene_type': str, 'gene_name': str, 'RTS_stage': str,
        'predicted_NMD': str, 'all_canonical': str
    }

    if rules_filter:
        cur_sqanti3_class = pd.read_csv(
            f"{DATA_DIR}/annotated_sqanti3/{sample_id}_annotated_transcripts.csv",
            usecols=use_sqanti3_cols,
            dtype=dtype_spec
        )
    else:
        cur_sqanti3_class = pd.read_csv(
            f"{DATA_DIR}/annotated_sqanti3/{sample_id}_ml_annotated_transcripts.csv",
            usecols=use_sqanti3_cols,
            dtype=dtype_spec
        )

    return cur_sqanti3_class


def read_sqanti3_filtered(sample_id: str, rules_filter: bool = True) -> pd.DataFrame:
    """
    Read a SQANTI3 filtered classification file and keep only isoforms.

    If rules_filter is True, reads the rules-filtered file; otherwise reads
    the ML-filtered file. In both cases, rows where filter_result is not
    "Isoform" are removed.

    :param sample_id: Sample identifier used to locate the classification file.
    :param rules_filter: If True, read the rules-filtered classification file; if False, read the ML-filtered classification file.
    :return: Filtered DataFrame containing only isoform rows with selected SQANTI3 annotation columns.
    """
    if rules_filter:
        cur_sqanti3_class = pd.read_csv(f"{DATA_DIR}/sqanti3_rules_filtered/"
                                        f"{sample_id}_sqanti3_filtered_RulesFilter_result_classification.txt",
                                        sep='\t')
    else:
        cur_sqanti3_class = pd.read_csv(f"{DATA_DIR}/sqanti3_ml_filtered/"
                                        f"{sample_id}_sqanti3_ml_filtered_MLresult_classification.txt",
                                        sep='\t')
    cur_sqanti3_class = cur_sqanti3_class[
        cur_sqanti3_class["filter_result"] == "Isoform"]
    
    if rules_filter:
        use_sqanti3_cols = ['isoform', 'chrom', 'strand', 'length', 'exons', 'structural_category', 
                            'associated_gene', 'associated_transcript', 'subcategory', 'RTS_stage', 
                            'predicted_NMD', 'all_canonical', 'ORF_length', 'CDS_length', 'length', 'ref_length',
                            'coding', 'FL', 'diff_to_gene_TSS', 'diff_to_gene_TTS', 'CDS_genomic_start', 'CDS_genomic_end',
                            'filter_result']
    else:
        use_sqanti3_cols = ['isoform', 'chrom', 'strand', 'length', 'exons', 'structural_category', 
                            'associated_gene', 'associated_transcript', 'subcategory', 'RTS_stage', 
                            'predicted_NMD', 'all_canonical', 'ORF_length', 'CDS_length', 'length', 'ref_length',
                            'coding', 'FL', 'diff_to_gene_TSS', 'diff_to_gene_TTS', 'CDS_genomic_start', 'CDS_genomic_end',
                            'intra_priming', 'filter_result']
    cur_sqanti3_class = cur_sqanti3_class[use_sqanti3_cols]
    return cur_sqanti3_class


def map_gene_ids_to_gencode_gene_name_gtf(gene_ids: list[str], gtf_file_path: str) -> dict[str, list[str]]:
    """
    Map gene IDs to gene names and gene types using a GENCODE GTF file.

    :param gene_ids: List of gene IDs to look up (e.g. ENSEMBL gene IDs).
    :param gtf_file_path: Path to the GTF file (supports .gz compressed files).
    :return: Mapping of gene_id to [gene_name, gene_type]. Returns ["novel", "novel"] for IDs with value "novel" (case-insensitive), and ["NA", "NA"] for IDs not found in the GTF.
    """
    result = {}
    for gid in gene_ids:
        if gid.lower() == "novel":
            result[gid] = ["novel", "novel"]

    gene_id_to_gene_name = {}

    open_func = gzip.open if gtf_file_path.endswith('.gz') else open
    with open_func(gtf_file_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'transcript':
                continue

            attributes = fields[8]

            gene_id = None
            gene_name = None
            gene_type = None

            for part in attributes.split(';'):
                part = part.strip()
                if part.startswith('gene_id'):
                    gene_id = part.split('"')[1]
                elif part.startswith('gene_name'):
                    gene_name = part.split('"')[1]
                elif part.startswith('gene_type'):
                    gene_type = part.split('"')[1]

            if gene_id and gene_name and gene_type:
                gene_id_to_gene_name[gene_id] = [gene_name, gene_type]
                # Also store without version number for ENSG lookups without version suffix
                gene_id_to_gene_name[gene_id.split('.')[0]] = [gene_name, gene_type]

    for gid in gene_ids:
        if gid.lower() != "novel":
            result[gid] = gene_id_to_gene_name.get(gid, ["NA", "NA"])

    return result
