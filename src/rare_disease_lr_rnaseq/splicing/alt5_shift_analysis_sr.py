#!/usr/bin/env python3
"""Alt 5' splice site shift analysis for short-read RNA-seq (STAR SJ.out.tab).

Applies the same alt 5' shift analysis used for long-read data to short-read
STAR SJ.out.tab files stored on GCS. Runs as a single Hail Batch job since
SJ.out.tab files are small (~1-5 MB each) and cross-sample z-scores need all
samples loaded simultaneously.

Execution flow:
  1. Local (launcher): Query Airtable for SR sample metadata, build job, submit
  2. Cloud (single job): Download GTF + all SJ files -> parse GTF -> load +
     annotate all samples -> run analysis -> upload results to GCS
  3. Local (--analyze): Download results from GCS for inspection
"""

import argparse
import logging
import re
import subprocess
from pathlib import Path
import hailtop.batch as hb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import (
    read_from_airtable,
    RNA_SEQ_BASE_ID,
    DATA_PATHS_TABLE_ID,
    DATA_PATHS_VIEW_ID,
)

from rare_disease_lr_rnaseq.config import DOCKER_PARTIAL_IR, GCS_REF_GTF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REGION = ["us-central1"]

GTF_GCS_PATH = GCS_REF_GTF

ANALYSIS_SCRIPT = r'''#!/usr/bin/env python3
"""Alt 5' shift analysis for short-read STAR SJ.out.tab files.

Self-contained script intended to run inside a Hail Batch job container.
Analytical functions replicated from alt5_shift_analysis.py (long-read version).
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PSI_EPS = 0.001  # offset for logit transform at boundaries
MIN_ANCHOR_READS = 20  # min anchor coverage to set PSI=0 for absent junctions
MIN_CANONICAL_READS = 5  # min canonical junction reads at anchor to trust PSI

# Strand inference from STAR intron motif codes:
#   1=GT/AG, 3=GC/AG, 5=AT/AC, 6=GT/AT -> "+"
#   2=CT/AC, 4=CT/GC -> "-"
#   0=non-canonical -> skip
MOTIF_TO_STRAND = {1: "+", 2: "-", 3: "+", 4: "-", 5: "+", 6: "+"}


def parse_gtf_splice_sites(
    gtf_path: str,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Parse GENCODE GTF to extract known donor and acceptor positions.

    :param gtf_path: Path to the GENCODE GTF file.
    :return: Tuple of (known_donors, known_acceptors) sets of (chrom, position) tuples.
    """
    log.info("Parsing GTF for known splice sites: %s", gtf_path)
    transcript_exons: dict[str, list[tuple[str, int, int]]] = {}

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "exon":
                continue

            chrom = fields[0]
            start = int(fields[3])  # 1-based
            end = int(fields[4])    # 1-based

            attrs = fields[8]
            m = re.search(r'transcript_id "([^"]+)"', attrs)
            if not m:
                continue
            tx_id = m.group(1)

            transcript_exons.setdefault(tx_id, []).append((chrom, start, end))

    log.info("  Parsed exons for %d transcripts", len(transcript_exons))

    known_donors: set[tuple[str, int]] = set()
    known_acceptors: set[tuple[str, int]] = set()

    for tx_id, exons in transcript_exons.items():
        if len(exons) < 2:
            continue
        exons.sort(key=lambda x: x[1])
        for i in range(len(exons) - 1):
            chrom_i, _, end_i = exons[i]
            chrom_next, start_next, _ = exons[i + 1]
            if chrom_i != chrom_next:
                continue
            # Intron: (end_i + 1) to (start_next - 1)
            known_donors.add((chrom_i, end_i + 1))
            known_acceptors.add((chrom_next, start_next - 1))

    log.info(
        "  Known splice sites: %d donors, %d acceptors",
        len(known_donors), len(known_acceptors),
    )
    return known_donors, known_acceptors


def load_sj_out_tab(
    filepath: str,
    known_donors: set[tuple[str, int]],
    known_acceptors: set[tuple[str, int]],
) -> pd.DataFrame:
    """Load and annotate a STAR SJ.out.tab file.

    :param filepath: Path to the STAR SJ.out.tab file.
    :param known_donors: Set of (chrom, position) tuples for known donor sites.
    :param known_acceptors: Set of (chrom, position) tuples for known acceptor sites.
    :return: Annotated DataFrame with junction_key, coordinates, site categories,
        and read counts.
    """
    col_names = [
        "chrom", "intron_start", "intron_end", "strand_code",
        "intron_motif", "annotated", "unique_reads", "multi_reads",
        "max_overhang",
    ]
    df = pd.read_csv(filepath, sep="\t", header=None, names=col_names)

    strand_map = {1: "+", 2: "-"}
    df["strand"] = df["strand_code"].map(strand_map)
    undefined_mask = df["strand_code"] == 0
    df.loc[undefined_mask, "strand"] = df.loc[undefined_mask, "intron_motif"].map(MOTIF_TO_STRAND)
    df = df.dropna(subset=["strand"])

    df["chrom"] = df["chrom"].astype(str)
    donor_tuples = list(zip(df["chrom"], df["intron_start"]))
    acceptor_tuples = list(zip(df["chrom"], df["intron_end"]))
    df["start_site_category"] = np.where(
        pd.array([t in known_donors for t in donor_tuples]), "known", "novel",
    )
    df["end_site_category"] = np.where(
        pd.array([t in known_acceptors for t in acceptor_tuples]), "known", "novel",
    )

    df["junction_key"] = (
        df["chrom"] + ":" + df["intron_start"].astype(str) + "-" + df["intron_end"].astype(str)
    )
    df = df.rename(columns={
        "intron_start": "genomic_start_coord",
        "intron_end": "genomic_end_coord",
        "unique_reads": "junction_unique_read_counts",
    })

    return df[[
        "junction_key", "chrom", "strand", "genomic_start_coord",
        "genomic_end_coord", "start_site_category", "end_site_category",
        "junction_unique_read_counts",
    ]].reset_index(drop=True)


def is_alt5_vec(df: pd.DataFrame) -> pd.Series:
    """Classify junctions as alt 5' events using vectorized operations.

    :param df: DataFrame with 'strand', 'start_site_category', and 'end_site_category' columns.
    :return: Boolean Series indicating alt 5' (alt donor) junctions.
    """
    plus = df["strand"] == "+"
    return (
        (plus & (df["start_site_category"] == "novel") & (df["end_site_category"] == "known"))
        | (~plus & (df["start_site_category"] == "known") & (df["end_site_category"] == "novel"))
    )


def has_any_novel_site_vec(df: pd.DataFrame) -> pd.Series:
    """Check if junctions have at least one novel splice site using vectorized operations.

    :param df: DataFrame with 'start_site_category' and 'end_site_category' columns.
    :return: Boolean Series indicating junctions with at least one novel site.
    """
    return (df["start_site_category"] == "novel") | (df["end_site_category"] == "novel")


def get_anchor_vec(df: pd.DataFrame) -> pd.Series:
    """Compute anchor coordinate for PSI grouping using vectorized operations.

    :param df: DataFrame with 'strand', 'chrom', 'genomic_start_coord', and
        'genomic_end_coord' columns.
    :return: Series of anchor coordinate strings in 'chrom:coord' format.
    """
    plus = df["strand"] == "+"
    return np.where(
        plus,
        df["chrom"] + ":" + df["genomic_end_coord"].astype(str),
        df["chrom"] + ":" + df["genomic_start_coord"].astype(str),
    )


def compute_psi_and_anchors(
    all_junctions_df: pd.DataFrame,
    alt5_junctions: set[str],
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Compute true PSI for each alt 5' junction and return anchor totals.

    :param all_junctions_df: DataFrame containing all junctions for a sample.
    :param alt5_junctions: Set of junction keys identified as alt 5' events.
    :return: Tuple of (psi, anchor_totals, canonical_at_anchor) dicts.
    """
    df = all_junctions_df.copy()
    df["anchor"] = get_anchor_vec(df)

    anchor_totals = (
        df.groupby("anchor")["junction_unique_read_counts"]
        .sum()
        .to_dict()
    )

    canonical_mask = (
        (df["start_site_category"] == "known") & (df["end_site_category"] == "known")
    )
    canonical_at_anchor = (
        df[canonical_mask]
        .groupby("anchor")["junction_unique_read_counts"]
        .sum()
        .to_dict()
    )

    alt5_df = df[df["junction_key"].isin(alt5_junctions)].copy()
    alt5_df["anchor_total"] = alt5_df["anchor"].map(anchor_totals).fillna(0)
    alt5_df["psi"] = np.where(
        alt5_df["anchor_total"] > 0,
        alt5_df["junction_unique_read_counts"] / alt5_df["anchor_total"],
        0.0,
    )
    psi = dict(zip(alt5_df["junction_key"], alt5_df["psi"]))

    return psi, anchor_totals, canonical_at_anchor


def logit(psi: float, eps: float = PSI_EPS) -> float:
    """Logit-transform a PSI value, clamping to [eps, 1-eps].

    :param psi: PSI value in [0, 1].
    :param eps: Offset used to clamp values away from 0 and 1.
    :return: Logit-transformed PSI value.
    """
    p = np.clip(psi, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_junction_matrix_sr(
    sj_files: dict[str, str],
    known_donors: set[tuple[str, int]],
    known_acceptors: set[tuple[str, int]],
    min_reads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build alt 5' junction x sample count and PSI matrices from SJ.out.tab files.

    :param sj_files: Mapping of sample_id to SJ.out.tab file path.
    :param known_donors: Set of (chrom, position) tuples for known donor sites.
    :param known_acceptors: Set of (chrom, position) tuples for known acceptor sites.
    :param min_reads: Minimum unique reads to consider a junction present.
    :return: Tuple of (counts_df, psi_df, junction_meta_df, all_novel_df,
        sample_total_junctions).
    """
    log.info("Loading %d SJ.out.tab files", len(sj_files))

    # Sanity check: track GTF annotation match rate
    total_both_known = 0
    total_star_annotated_1 = 0

    all_counts: dict[str, dict[str, int]] = {}
    all_psi: dict[str, dict[str, float]] = {}
    junction_meta: dict[str, dict] = {}
    junction_anchors: dict[str, str] = {}
    all_novel_counts: dict[str, dict[str, int]] = {}
    sample_anchor_totals: dict[str, dict[str, int]] = {}
    sample_canonical_at_anchor: dict[str, dict[str, int]] = {}
    sample_total_junctions: dict[str, int] = {}  # depth proxy

    for sample_id, filepath in sorted(sj_files.items()):
        log.info("  Processing %s", sample_id)

        df = load_sj_out_tab(filepath, known_donors, known_acceptors)
        if df.empty:
            continue

        sample_total_junctions[sample_id] = len(df)

        df["is_alt5"] = is_alt5_vec(df)
        df["has_novel_site"] = has_any_novel_site_vec(df)
        alt5_keys = set(df.loc[df["is_alt5"], "junction_key"])

        psi_dict, anchor_totals, canonical_reads = compute_psi_and_anchors(df, alt5_keys)
        sample_anchor_totals[sample_id] = anchor_totals
        sample_canonical_at_anchor[sample_id] = canonical_reads

        alt5_df = df[df["is_alt5"]]
        alt5_anchors = get_anchor_vec(alt5_df)
        for jk, chrom, strand, start, end, reads, anchor in zip(
            alt5_df["junction_key"], alt5_df["chrom"], alt5_df["strand"],
            alt5_df["genomic_start_coord"], alt5_df["genomic_end_coord"],
            alt5_df["junction_unique_read_counts"], alt5_anchors,
        ):
            reads = int(reads)
            all_counts.setdefault(jk, {})[sample_id] = reads
            if jk in psi_dict:
                all_psi.setdefault(jk, {})[sample_id] = psi_dict[jk]
            if jk not in junction_meta:
                junction_meta[jk] = {
                    "chrom": chrom, "strand": strand,
                    "genomic_start_coord": start, "genomic_end_coord": end,
                }
                junction_anchors[jk] = anchor

        novel_df = df[df["has_novel_site"]]
        for jk, reads in zip(novel_df["junction_key"], novel_df["junction_unique_read_counts"]):
            all_novel_counts.setdefault(jk, {})[sample_id] = int(reads)

        both_known = (df["start_site_category"] == "known") & (df["end_site_category"] == "known")
        total_both_known += int(both_known.sum())

    log.info(
        "GTF annotation sanity check: %d junctions with both sites 'known' (from GTF)",
        total_both_known,
    )

    if not all_counts:
        log.warning("No alt 5' junctions found across any samples")
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {}

    sample_ids = sorted({s for d in all_counts.values() for s in d})

    count_rows = []
    psi_rows = []
    for jk in sorted(all_counts):
        crow = {s: all_counts[jk].get(s, 0) for s in sample_ids}
        crow["junction_key"] = jk
        count_rows.append(crow)

        anchor = junction_anchors[jk]
        prow = {"junction_key": jk}
        for s in sample_ids:
            canonical = sample_canonical_at_anchor.get(s, {}).get(anchor, 0)
            if jk in all_psi and s in all_psi[jk]:
                if canonical >= MIN_CANONICAL_READS:
                    prow[s] = all_psi[jk][s]
                else:
                    prow[s] = np.nan
            else:
                anchor_total = sample_anchor_totals.get(s, {}).get(anchor, 0)
                if anchor_total >= MIN_ANCHOR_READS and canonical >= MIN_CANONICAL_READS:
                    prow[s] = 0.0
                else:
                    prow[s] = np.nan
        psi_rows.append(prow)

    counts_df = pd.DataFrame(count_rows).set_index("junction_key")
    counts_df = counts_df[sample_ids]

    psi_df = pd.DataFrame(psi_rows).set_index("junction_key")
    psi_df = psi_df[sample_ids]

    meta_rows = []
    for jk in counts_df.index:
        n_present = int((counts_df.loc[jk] >= min_reads).sum())
        meta_rows.append({
            "junction_key": jk,
            **junction_meta[jk],
            "n_samples_present": n_present,
        })
    junction_meta_df = pd.DataFrame(meta_rows).set_index("junction_key")

    all_novel_rows = []
    for jk in sorted(all_novel_counts):
        crow = {s: all_novel_counts[jk].get(s, 0) for s in sample_ids}
        crow["junction_key"] = jk
        all_novel_rows.append(crow)
    all_novel_df = pd.DataFrame(all_novel_rows).set_index("junction_key")
    all_novel_df = all_novel_df[sample_ids]

    n_valid = int(psi_df.notna().sum().sum())
    n_nan = int(psi_df.isna().sum().sum())
    n_total = n_valid + n_nan
    present_mask = counts_df > 0
    n_present_nan = int((present_mask & psi_df.isna()).sum().sum())
    log.info(
        "Alt 5' junction matrix: %d junctions x %d samples",
        len(counts_df), len(sample_ids),
    )
    log.info(
        "PSI quality: %d/%d cells valid (%.1f%%), %d NaN "
        "(canonical < %d: %d present junctions masked)",
        n_valid, n_total, 100 * n_valid / n_total if n_total > 0 else 0, n_nan,
        MIN_CANONICAL_READS, n_present_nan,
    )
    log.info("All novel-site junction matrix: %d junctions", len(all_novel_df))
    return counts_df, psi_df, junction_meta_df, all_novel_df, sample_total_junctions


def categorize_junctions(psi_df: pd.DataFrame) -> dict[str, str]:
    """Categorise each junction as 'novel' or 'shared' by observability.

    :param psi_df: PSI matrix with junctions as rows and samples as columns.
    :return: Mapping of junction_key to category ('novel' or 'shared').
    """
    n_observable = psi_df.notna().sum(axis=1)
    labels = np.where(n_observable <= 2, "novel", "shared")
    return dict(zip(psi_df.index, labels))


def compute_psi_outliers(
    psi_df: pd.DataFrame,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Compute robust z-score of logit-transformed PSI across samples per junction.

    :param psi_df: PSI matrix with junctions as rows and samples as columns.
    :param min_samples: Minimum number of non-NaN samples required to compute z-scores.
    :return: Same-shape DataFrame of robust z-scores (NaN where PSI is NaN).
    """
    psi_arr = psi_df.values.astype(float)
    clipped = np.clip(psi_arr, PSI_EPS, 1.0 - PSI_EPS)
    logit_arr = np.log(clipped / (1.0 - clipped))
    logit_arr[np.isnan(psi_arr)] = np.nan

    zscore_arr = np.full_like(logit_arr, np.nan)
    n_valid = np.sum(~np.isnan(logit_arr), axis=1)

    for i in range(logit_arr.shape[0]):
        if n_valid[i] < min_samples:
            continue
        row = logit_arr[i]
        valid = row[~np.isnan(row)]
        med = np.median(valid)
        mad = median_abs_deviation(valid, nan_policy="omit")
        mask = ~np.isnan(row)
        if mad == 0:
            zscore_arr[i, mask] = 0.0
        else:
            zscore_arr[i, mask] = (row[mask] - med) / (1.4826 * mad)

    return pd.DataFrame(zscore_arr, index=psi_df.index, columns=psi_df.columns)


def compute_sample_metrics(
    psi_zscore_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    junction_categories: dict[str, str],
    all_novel_df: pd.DataFrame,
    min_reads: int,
    sample_total_junctions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Compute shift and novel metrics per sample.

    :param psi_zscore_df: Robust z-score matrix for PSI values (junctions x samples).
    :param counts_df: Alt 5' read count matrix (junctions x samples).
    :param junction_categories: Mapping of junction_key to 'novel' or 'shared'.
    :param all_novel_df: Count matrix for all novel-site junctions (junctions x samples).
    :param min_reads: Minimum read count to consider a junction present.
    :param sample_total_junctions: Optional mapping of sample_id to total junction count.
    :return: Per-sample metrics DataFrame.
    """
    samples = counts_df.columns.tolist()
    shared_junctions = [jk for jk, cat in junction_categories.items() if cat == "shared"]
    novel_alt5_junctions = [jk for jk, cat in junction_categories.items() if cat == "novel"]

    all_novel_rare = [
        jk for jk in all_novel_df.index
        if int((all_novel_df.loc[jk] >= min_reads).sum()) <= 2
    ]

    rows = []
    for s in samples:
        z_vals = psi_zscore_df.loc[shared_junctions, s].dropna() if shared_junctions else pd.Series(dtype=float)
        n_shared_present = len(z_vals)
        n_psi_outliers = int((z_vals >= 2).sum()) if len(z_vals) > 0 else 0

        n_novel_alt5 = int(
            (counts_df.loc[novel_alt5_junctions, s] >= min_reads).sum()
        ) if novel_alt5_junctions else 0

        n_novel_any = int(
            (all_novel_df.loc[all_novel_rare, s] >= min_reads).sum()
        ) if all_novel_rare else 0

        total_alt5 = int((counts_df[s] >= min_reads).sum())

        n_total_junc = (
            sample_total_junctions.get(s, 0)
            if sample_total_junctions is not None
            else 0
        )

        rows.append({
            "sample_id": s,
            "n_shared_present": n_shared_present,
            "n_psi_outliers": n_psi_outliers,
            "n_novel_alt5": n_novel_alt5,
            "n_novel_any": n_novel_any,
            "total_alt5": total_alt5,
            "n_total_junctions": n_total_junc,
        })

    return pd.DataFrame(rows)


def compute_depth_controlled_novel_z(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Regress n_novel_alt5 on log(n_total_junctions) to remove depth effect.

    :param metrics_df: Per-sample metrics DataFrame with 'n_novel_alt5' and
        'n_total_junctions' columns.
    :return: Copy of metrics_df with a 'residual_z' column added.
    """
    from scipy.stats import linregress

    df = metrics_df.copy()
    # Only fit on samples with non-zero total junctions
    mask = df["n_total_junctions"] > 0
    if mask.sum() < 3:
        log.warning(
            "Too few samples with total junctions for regression; skipping depth control"
        )
        df["residual_z"] = np.nan
        return df

    x = np.log(df.loc[mask, "n_total_junctions"].values.astype(float))
    y = df.loc[mask, "n_novel_alt5"].values.astype(float)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    predicted = slope * x + intercept
    residuals = y - predicted
    res_std = np.std(residuals)
    if res_std == 0:
        log.warning("Zero residual variance; all samples identical")
        df["residual_z"] = 0.0
        return df

    residual_z = residuals / res_std

    df["residual_z"] = np.nan
    df.loc[mask, "residual_z"] = residual_z

    log.info(
        "Depth-controlled novel enrichment: n_novel_alt5 ~ log(n_total_junctions), "
        "slope=%.2f, intercept=%.2f, R²=%.3f, p=%.2e",
        slope, intercept, r_value**2, p_value,
    )
    return df


def compute_n_expressed_genes_sr(tpm_dir: str) -> pd.DataFrame:
    """Count expressed genes per sample from RNASeQC gene TPM files.

    :param tpm_dir: Directory containing per-sample *.gene_tpm.gct files.
    :return: DataFrame with columns 'sample_id' and 'n_expressed'.
    """
    tpm_path = Path(tpm_dir)
    rows: list[dict] = []

    for f in sorted(tpm_path.glob("*.gene_tpm.gct")):
        sample_id = f.name.replace(".gene_tpm.gct", "")
        df = pd.read_csv(f, sep="\t", comment="#", skiprows=2)
        # GCT: first 2 columns are Name, Description; column index 2 is TPM
        n_expressed = int((df.iloc[:, 2] > 1).sum())
        rows.append({"sample_id": sample_id, "n_expressed": n_expressed})
        log.info("  %s: n_expressed=%d", sample_id, n_expressed)

    log.info("Computed n_expressed for %d samples", len(rows))
    return pd.DataFrame(rows)


def compute_truly_novel_counts(
    psi_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    min_reads: int,
    min_observable: int = 10,
    max_present: int = 2,
) -> pd.DataFrame:
    """Count truly novel alt 5' junctions per sample.

    :param psi_df: PSI matrix with junctions as rows and samples as columns.
    :param counts_df: Alt 5' read count matrix (junctions x samples).
    :param min_reads: Minimum read count to consider a junction present.
    :param min_observable: Minimum number of samples where the junction must be observable.
    :param max_present: Maximum number of samples where the junction is present.
    :return: DataFrame with columns 'sample_id' and 'n_truly_novel_alt5'.
    """
    truly_novel = []
    for jk in psi_df.index:
        n_observable = int(psi_df.loc[jk].notna().sum())
        n_present = int((counts_df.loc[jk] >= min_reads).sum())
        if n_observable >= min_observable and n_present <= max_present:
            truly_novel.append(jk)

    log.info(
        "Truly novel alt 5' junctions: %d / %d (observable >= %d, present <= %d)",
        len(truly_novel), len(psi_df), min_observable, max_present,
    )

    sample_ids = counts_df.columns.tolist()
    rows = []
    for s in sample_ids:
        n = int((counts_df.loc[truly_novel, s] >= min_reads).sum()) if truly_novel else 0
        rows.append({"sample_id": s, "n_truly_novel_alt5": n})

    return pd.DataFrame(rows)


def compute_gene_enrichment(
    metrics_df: pd.DataFrame,
    truly_novel_df: pd.DataFrame,
    n_expressed_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Regress n_truly_novel_alt5 on n_expressed to produce depth-controlled z-scores.

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id' column.
    :param truly_novel_df: DataFrame with columns 'sample_id' and 'n_truly_novel_alt5'.
    :param n_expressed_df: DataFrame with columns 'sample_id' and 'n_expressed'.
    :param output_dir: Directory to save the gene enrichment TSV file.
    :return: Input metrics_df with 'n_truly_novel_alt5', 'n_expressed', and
        'gene_enrichment_z' columns added.
    """
    from scipy.stats import linregress

    df = metrics_df.merge(truly_novel_df, on="sample_id", how="left")
    df = df.merge(n_expressed_df, on="sample_id", how="left")

    mask = df["n_expressed"].notna() & (df["n_expressed"] > 0)
    if mask.sum() < 3:
        log.warning("Too few samples for gene enrichment regression")
        df["gene_enrichment_z"] = np.nan
        return df

    x = df.loc[mask, "n_expressed"].values.astype(float)
    y = df.loc[mask, "n_truly_novel_alt5"].values.astype(float)

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    predicted = slope * x + intercept
    residuals = y - predicted
    res_std = np.std(residuals)

    if res_std == 0:
        log.warning("Zero residual variance in gene enrichment")
        df["gene_enrichment_z"] = 0.0
    else:
        df["gene_enrichment_z"] = np.nan
        df.loc[mask, "gene_enrichment_z"] = residuals / res_std

    log.info(
        "Gene enrichment: n_truly_novel_alt5 ~ n_expressed, "
        "slope=%.4f, intercept=%.2f, R²=%.3f, p=%.2e",
        slope, intercept, r_value**2, p_value,
    )

    enrich_df = df.loc[
        mask,
        ["sample_id", "n_truly_novel_alt5", "n_expressed", "gene_enrichment_z"],
    ].copy()
    enrich_path = output_dir / "alt5_gene_enrichment.tsv"
    enrich_df.to_csv(enrich_path, sep="\t", index=False)
    log.info("Saved %s", enrich_path)

    return df


def main() -> None:
    """Run alt 5' shift analysis for short-read STAR SJ.out.tab files."""
    parser = argparse.ArgumentParser(description="Alt 5' shift analysis (short-read)")
    parser.add_argument("--gtf", required=True, help="Path to GENCODE GTF file")
    parser.add_argument("--sj-dir", required=True, help="Directory with *_SJ.out.tab files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-reads", type=int, default=2,
                        help="Min unique reads to consider junction present")
    parser.add_argument("--proband-pattern", type=str, default="_3",
                        help="Regex for proband identification in sample names")
    parser.add_argument("--tpm-dir", type=str, default=None,
                        help="Directory with RNASeQC *.gene_tpm.gct files for n_expressed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    known_donors, known_acceptors = parse_gtf_splice_sites(args.gtf)

    sj_dir = Path(args.sj_dir)
    sj_files: dict[str, str] = {}
    for f in sorted(sj_dir.glob("*_SJ.out.tab")):
        sample_id = f.name.replace("_SJ.out.tab", "")
        sj_files[sample_id] = str(f)
    log.info("Found %d SJ.out.tab files", len(sj_files))

    counts_df, psi_df, junction_meta_df, all_novel_df, sample_total_junctions = (
        build_junction_matrix_sr(
            sj_files, known_donors, known_acceptors, args.min_reads,
        )
    )
    if counts_df.empty:
        log.error("No data to analyse. Exiting.")
        return

    junction_categories = categorize_junctions(psi_df)
    n_shared = sum(1 for v in junction_categories.values() if v == "shared")
    n_novel = sum(1 for v in junction_categories.values() if v == "novel")
    log.info(
        "Junction categories: %d shared (observable >= 3), %d novel (observable <= 2)",
        n_shared, n_novel,
    )

    psi_zscore_df = compute_psi_outliers(psi_df)

    metrics_df = compute_sample_metrics(
        psi_zscore_df, counts_df, junction_categories, all_novel_df, args.min_reads,
        sample_total_junctions,
    )

    metrics_df = compute_depth_controlled_novel_z(metrics_df)

    truly_novel_df = compute_truly_novel_counts(psi_df, counts_df, args.min_reads)
    if args.tpm_dir and Path(args.tpm_dir).exists():
        n_expressed_df = compute_n_expressed_genes_sr(args.tpm_dir)
        if not n_expressed_df.empty:
            metrics_df = compute_gene_enrichment(
                metrics_df, truly_novel_df, n_expressed_df, output_dir,
            )
        else:
            log.warning("No TPM files found — skipping gene enrichment")
    else:
        log.info("No --tpm-dir provided — skipping gene enrichment")

    log.info(f"\n{'='*70}")
    log.info("Alt 5' Shift Analysis — Short-Read (robust logit-PSI z-scores)")
    log.info(f"{'='*70}")
    log.info(f"Total alt 5' junctions: {len(counts_df)}")
    log.info(f"  Shared (observable in >= 3 samples): {n_shared}")
    log.info(f"  Novel  (observable in <= 2 samples): {n_novel}")
    log.info(f"Total novel-site junctions (any type): {len(all_novel_df)}")
    log.info(f"Samples: {len(counts_df.columns)}")
    log.info(f"Min reads threshold: {args.min_reads}")

    log.info(f"\n--- Top 20 shift signal (sorted by n_psi_outliers, z >= 2) ---")
    shift_sorted = metrics_df.sort_values("n_psi_outliers", ascending=False).head(20)
    for _, row in shift_sorted.iterrows():
        log.info(
            f"  {row['sample_id']:30s}  n_psi_outliers={row['n_psi_outliers']:4d}  "
            f"n_shared={row['n_shared_present']:4d}"
        )

    log.info(f"\n--- Top 20 novel signal (sorted by n_novel_alt5) ---")
    novel_sorted = metrics_df.sort_values("n_novel_alt5", ascending=False).head(20)
    for _, row in novel_sorted.iterrows():
        ratio = row["n_novel_alt5"] / row["n_novel_any"] * 100 if row["n_novel_any"] > 0 else 0
        log.info(
            f"  {row['sample_id']:30s}  n_novel_alt5={row['n_novel_alt5']:4d}  "
            f"n_novel_any={row['n_novel_any']:4d}  ratio={ratio:5.1f}%"
        )

    metrics_df.to_csv(output_dir / "sample_shift_metrics.tsv", sep="\t", index=False)
    log.info("Saved sample_shift_metrics.tsv")

    counts_df.to_csv(output_dir / "junction_counts_matrix.tsv", sep="\t")
    log.info("Saved junction_counts_matrix.tsv")

    psi_df.to_csv(output_dir / "junction_psi_matrix.tsv", sep="\t")
    log.info("Saved junction_psi_matrix.tsv")

    log.info("All outputs saved to: %s", output_dir)


if __name__ == "__main__":
    main()
'''


def get_sr_sample_metadata(sj_column: str) -> dict[str, str]:
    """Query Airtable for short-read sample metadata and return SJ file paths.

    :param sj_column: Airtable column name containing GCS paths to SJ.out.tab files.
    :return: Mapping of sample_id to GCS SJ.out.tab file path.
    """
    log.info("Reading sample metadata from Airtable...")
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["imputed_tissue"] == "whole_blood"]
    dat = dat[~(dat["exclude"] == "yes")]
    dat = dat[dat["watchmaker"] == "yes"]
    log.info("  %d samples after filtering", len(dat))

    dat = dat[dat[sj_column].notna() & (dat[sj_column] != "")]
    log.info("  %d samples with %s column", len(dat), sj_column)

    sj_paths: dict[str, str] = {}
    for _, row in dat.iterrows():
        sample_id = str(row["sample_id"])
        sj_path = str(row[sj_column])
        sj_paths[sample_id] = sj_path

    return sj_paths


def get_sr_tpm_metadata(tpm_column: str, sample_ids: set[str]) -> dict[str, str]:
    """Query Airtable for gene TPM paths, filtered to sample_ids with SJ data.

    :param tpm_column: Airtable column name containing GCS paths to gene TPM files.
    :param sample_ids: Set of sample IDs to filter results to (those with SJ data).
    :return: Mapping of sample_id to GCS gene TPM file path.
    """
    log.info("Reading TPM metadata from Airtable...")
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["imputed_tissue"] == "whole_blood"]
    dat = dat[~(dat["exclude"] == "yes")]
    dat = dat[dat["watchmaker"] == "yes"]

    dat = dat[dat[tpm_column].notna() & (dat[tpm_column] != "")]
    dat = dat[dat["sample_id"].isin(sample_ids)]
    log.info("  %d samples with %s column (matching SJ samples)", len(dat), tpm_column)

    tpm_paths: dict[str, str] = {}
    for _, row in dat.iterrows():
        sample_id = str(row["sample_id"])
        tpm_path = str(row[tpm_column])
        tpm_paths[sample_id] = tpm_path

    return tpm_paths


def _sample_color(sample: str, proband_pattern: str) -> str:
    """Return bar color based on whether sample matches the proband pattern.

    :param sample: Sample ID string.
    :param proband_pattern: Regex pattern used to identify proband samples.
    :return: 'coral' for probands, 'steelblue' for parents.
    """
    return "coral" if re.search(proband_pattern, sample) else "steelblue"


def plot_ranked_shift(
    metrics_df: pd.DataFrame, output_dir: Path, proband_pattern: str,
) -> None:
    """Generate ranked bar chart of directional PSI outlier count (shift signal).

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id' and 'n_psi_outliers'.
    :param output_dir: Directory to save the output PNG file.
    :param proband_pattern: Regex pattern used to identify proband samples for color coding.
    """
    df = metrics_df.sort_values("n_psi_outliers", ascending=False)
    colors = [_sample_color(s, proband_pattern) for s in df["sample_id"]]

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.1), 7))
    ax.bar(range(len(df)), df["n_psi_outliers"].values, color=colors,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["sample_id"].values, rotation=90, fontsize=4)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Number of alt 5' ss outliers")
    ax.set_title("Alt 5' usage shift in existing isoforms")

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()
    if outlier_std > 0:
        ax.axhline(outlier_mean + 2 * outlier_std, color="red",
                    linestyle="--", linewidth=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    legend_elements = [
        Patch(facecolor="coral", edgecolor="black", label="Proband"),
        Patch(facecolor="steelblue", edgecolor="black", label="Parent"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="+2 SD"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    out = output_dir / "alt5_shift_ranked_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_ranked_novel(
    metrics_df: pd.DataFrame, output_dir: Path, proband_pattern: str,
) -> None:
    """Generate ranked bar chart of alt 5' fraction of rare junctions.

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id', 'n_novel_alt5',
        and 'n_novel_any' columns.
    :param output_dir: Directory to save the output PNG file.
    :param proband_pattern: Regex pattern used to identify proband samples for color coding.
    """
    df = metrics_df.copy()
    df["novel_ratio"] = df["n_novel_alt5"] / df["n_novel_any"].replace(0, np.nan)
    df = df.sort_values("novel_ratio", ascending=False)
    colors = [_sample_color(s, proband_pattern) for s in df["sample_id"]]

    fig, ax = plt.subplots(figsize=(max(14, len(df) * 0.1), 7))
    ax.bar(range(len(df)), df["novel_ratio"].values * 100, color=colors,
           edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["sample_id"].values, rotation=90, fontsize=4)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Alt 5' fraction of rare junctions (%)")
    ax.set_title("Alt 5' novel signal — fraction of rare junctions that are alt 5'")
    legend_elements = [
        Patch(facecolor="coral", edgecolor="black", label="Proband"),
        Patch(facecolor="steelblue", edgecolor="black", label="Parent"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    out = output_dir / "alt5_novel_ranked_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_shift_vs_novel(
    metrics_df: pd.DataFrame, output_dir: Path, proband_pattern: str,
) -> None:
    """Generate 2D scatter of gene_enrichment_z vs n_psi_outliers.

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id', 'n_psi_outliers',
        and optionally 'gene_enrichment_z' columns.
    :param output_dir: Directory to save the output PNG file.
    :param proband_pattern: Regex pattern used to identify proband samples for color coding.
    """
    if "gene_enrichment_z" not in metrics_df.columns:
        log.warning("gene_enrichment_z not in metrics — skipping gene enrichment plot")
        return

    df = metrics_df.dropna(subset=["gene_enrichment_z"]).copy()
    if df.empty:
        log.warning("No valid gene_enrichment_z values — skipping plot")
        return

    colors = [_sample_color(s, proband_pattern) for s in df["sample_id"]]

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df["gene_enrichment_z"].values, df["n_psi_outliers"].values,
        c=colors, edgecolors="black", linewidth=0.5, s=60, zorder=3,
    )
    ax.set_xlabel("Novel alt 5' ss burden z-score")
    ax.set_ylabel("Alt 5' usage shift in existing isoforms")
    ax.set_title("Alt 5' splice site events")

    if outlier_std > 0:
        ax.axhline(outlier_mean + 2 * outlier_std, color="red",
                    linestyle="--", linewidth=2)
    ax.axvline(3, color="blue", linestyle=":", linewidth=2)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for _, row in df.iterrows():
        yz = (row["n_psi_outliers"] - outlier_mean) / outlier_std if outlier_std > 0 else 0
        if abs(row["gene_enrichment_z"]) > 2 or abs(yz) > 2 or "<target_sample>" in row["sample_id"]:
            label = row["sample_id"].replace("_R1", "").replace("_watch_maker", "")
            ax.annotate(
                label, (row["gene_enrichment_z"], row["n_psi_outliers"]),
                fontsize=6, xytext=(5, 5), textcoords="offset points",
            )

    legend_elements = [
        Patch(facecolor="coral", edgecolor="black", label="Proband"),
        Patch(facecolor="steelblue", edgecolor="black", label="Parent"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="+2 SD"),
        Line2D([0], [0], color="blue", linestyle=":", linewidth=2, label="+3 SD"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    out = output_dir / "alt5_shift_vs_novel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


from rare_disease_lr_rnaseq.utils import DATA_DIR as _DATA_DIR
_LOCAL_DATA_DIR = Path(_DATA_DIR)


def analyze_results(output: str, proband_pattern: str) -> None:
    """Download results from GCS and generate plots locally.

    :param output: GCS path to the results directory.
    :param proband_pattern: Regex pattern used to identify proband samples for color coding.
    """
    local_dir = _LOCAL_DATA_DIR / "outrider" / "alt5_shift_analysis_sr"
    local_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading results from %s ...", output)
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{output.rstrip('/')}/*", str(local_dir)],
        check=True,
    )

    metrics_path = local_dir / "sample_shift_metrics.tsv"
    if not metrics_path.exists():
        log.error("sample_shift_metrics.tsv not found in downloaded results")
        return

    df = pd.read_csv(metrics_path, sep="\t")
    log.info(f"\n{'=' * 70}")
    log.info("Alt 5' Shift Analysis — Short-Read Results")
    log.info(f"{'=' * 70}")
    log.info(f"Total samples: {len(df)}")
    log.info("\n--- Top 20 shift signal (sorted by n_psi_outliers) ---")
    shift_sorted = df.sort_values("n_psi_outliers", ascending=False).head(20)
    for _, row in shift_sorted.iterrows():
        log.info(
            f"  {row['sample_id']:30s}  n_psi_outliers={row['n_psi_outliers']:4d}  "
            f"n_shared={row['n_shared_present']:4d}"
        )
    log.info("\n--- Top 20 novel signal (sorted by n_novel_alt5) ---")
    novel_sorted = df.sort_values("n_novel_alt5", ascending=False).head(20)
    for _, row in novel_sorted.iterrows():
        ratio = (
            row["n_novel_alt5"] / row["n_novel_any"] * 100
            if row["n_novel_any"] > 0
            else 0
        )
        log.info(
            f"  {row['sample_id']:30s}  n_novel_alt5={row['n_novel_alt5']:4d}  "
            f"n_novel_any={row['n_novel_any']:4d}  ratio={ratio:5.1f}%"
        )

    plot_ranked_shift(df, local_dir, proband_pattern)
    plot_ranked_novel(df, local_dir, proband_pattern)
    plot_shift_vs_novel(df, local_dir, proband_pattern)

    log.info(f"\nResults and plots saved to: {local_dir}")
    for f in sorted(local_dir.glob("*.png")):
        log.info("  %s", f)


def main(
    billing_project: str,
    output: str,
    min_reads: int,
    proband_pattern: str,
    analyze: bool,
    dry_run: bool,
    sj_column: str,
    tpm_column: str,
) -> None:
    """Run alt 5' splice site shift analysis for short-read RNA-seq (STAR SJ.out.tab).

    :param billing_project: Hail Batch billing project.
    :param output: GCS output directory for results.
    :param min_reads: Minimum unique reads to consider a junction present.
    :param proband_pattern: Regex pattern for proband identification in sample names.
    :param analyze: If True, download results from GCS and display locally.
    :param dry_run: If True, print sample list without submitting batch job.
    :param sj_column: Airtable column name for SJ.out.tab GCS paths.
    :param tpm_column: Airtable column name for RNASeQC gene TPM GCS paths.
    """
    if analyze:
        analyze_results(output, proband_pattern)
        return

    sj_paths = get_sr_sample_metadata(sj_column)
    tpm_paths = get_sr_tpm_metadata(tpm_column, set(sj_paths.keys()))

    if dry_run:
        log.info("=== DRY RUN ===")
        log.info("Total SJ samples: %d", len(sj_paths))
        log.info("Total TPM samples: %d", len(tpm_paths))
        log.info("Output: %s", output)
        log.info("Min reads: %d", min_reads)
        log.info("Proband pattern: %s", proband_pattern)
        log.info("SJ column: %s", sj_column)
        log.info("TPM column: %s", tpm_column)
        log.info("")
        log.info("Sample list (SJ):")
        for sample_id, sj_path in sorted(sj_paths.items()):
            log.info("  %s -> %s", sample_id, sj_path)
        return

    if not sj_paths:
        log.error("No samples found with %s paths. Exiting.", sj_column)
        return

    backend = hb.ServiceBackend(
        billing_project=billing_project,
        remote_tmpdir=f"{output.rstrip('/')}/tmp",
        regions=REGION,
    )
    batch = hb.Batch(
        backend=backend,
        name="alt5_shift_analysis_sr",
        requester_pays_project="cmg-analysis",
        default_image=DOCKER_PARTIAL_IR,
    )

    job = batch.new_job("alt5_shift_sr")
    job.cpu(4)
    job.memory("highmem")
    job.storage("50G")

    job.command("pip install scipy")

    job.command(f"gcloud storage cp '{GTF_GCS_PATH}' /io/gencode.gtf")

    job.command("mkdir -p /io/sj_files /io/gene_tpm /io/results")
    for sample_id, gcs_path in sorted(sj_paths.items()):
        job.command(
            f"gcloud storage cp '{gcs_path}' '/io/sj_files/{sample_id}_SJ.out.tab.gz'"
            f" && gunzip '/io/sj_files/{sample_id}_SJ.out.tab.gz'"
        )

    for sample_id, gcs_path in sorted(tpm_paths.items()):
        job.command(
            f"gcloud storage cp '{gcs_path}' '/io/gene_tpm/{sample_id}.gene_tpm.gct.gz'"
            f" && gunzip '/io/gene_tpm/{sample_id}.gene_tpm.gct.gz'"
        )

    job.command(f"""cat > /io/analysis.py << 'SCRIPT_EOF'
{ANALYSIS_SCRIPT}
SCRIPT_EOF
""")

    analysis_cmd = (
        f"python /io/analysis.py"
        f" --gtf /io/gencode.gtf"
        f" --sj-dir /io/sj_files"
        f" --output-dir /io/results"
        f" --min-reads {min_reads}"
        f" --proband-pattern '{proband_pattern}'"
    )
    if tpm_paths:
        analysis_cmd += " --tpm-dir /io/gene_tpm"
    job.command(analysis_cmd)

    job.command(f"gcloud storage cp -r '/io/results/*' '{output.rstrip('/')}/'")

    log.info(
        "Submitting batch with 1 job (%d SJ samples, %d TPM samples)",
        len(sj_paths), len(tpm_paths),
    )
    batch.run()
    log.info("Batch submitted. Results will be uploaded to: %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run alt 5' splice site shift analysis for short-read RNA-seq."
    )
    parser.add_argument("--billing-project", type=str, default="tgg-rare-disease",
                        help="Hail Batch billing project.")
    parser.add_argument("--output", type=str, required=True,
                        help="GCS output directory for results.")
    parser.add_argument("--min-reads", type=int, default=2,
                        help="Minimum unique reads to consider a junction present.")
    parser.add_argument("--proband-pattern", type=str, default="_3",
                        help="Regex pattern for proband identification in sample names.")
    parser.add_argument("--analyze", action="store_true",
                        help="Download results from GCS and display locally.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample list without submitting batch job.")
    parser.add_argument("--sj-column", type=str, default="star_SJ_out_tab",
                        help="Airtable column name for SJ.out.tab GCS paths.")
    parser.add_argument("--tpm-column", type=str, default="rnaseqc_gene_tpm",
                        help="Airtable column name for RNASeQC gene TPM GCS paths.")
    args = parser.parse_args()
    main(
        billing_project=args.billing_project,
        output=args.output,
        min_reads=args.min_reads,
        proband_pattern=args.proband_pattern,
        analyze=args.analyze,
        dry_run=args.dry_run,
        sj_column=args.sj_column,
        tpm_column=args.tpm_column,
    )
