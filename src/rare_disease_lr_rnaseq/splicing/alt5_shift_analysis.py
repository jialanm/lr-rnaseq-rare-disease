#!/usr/bin/env python3
"""Alt 5' splice site shift analysis — cross-sample junction comparison.

Reads per-sample junction count TSVs (from count_alt5_junction_support_batch.py),
identifies alt 5' junctions, computes true PSI by grouping junctions that share
the same known splice site, then detects PSI outliers across samples.

True PSI for an alt 5' junction:
  PSI = junction_reads / sum(reads for all junctions sharing the same acceptor)

For + strand alt 5': novel donor + known acceptor → group by (chrom, end_coord)
For - strand alt 5': known donor + novel acceptor → group by (chrom, start_coord)

Outlier detection uses:
  - Logit-transformed PSI: logit(PSI + eps) to handle bounded [0,1] distribution
  - Robust z-scores: (logit_PSI - median) / (1.4826 * MAD) to resist outlier masking
  - Absence handling: PSI = 0 when anchor coverage >= MIN_ANCHOR_READS but junction absent
  - Directional counting: z >= 2 (elevated alt usage) as the primary signal

Produces:
  - Shift signal: count of shared alt 5' junctions with elevated PSI (robust z >= 2)
  - Novel signal: alt 5' fraction of rare junctions
  - Ranked bar charts, 2D scatter, and summary tables
"""

import logging
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import linregress, median_abs_deviation

from rare_disease_lr_rnaseq.utils import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PSI_EPS = 0.001  # offset for logit transform at boundaries
MIN_ANCHOR_READS = 20  # min anchor coverage to set PSI=0 for absent junctions
MIN_CANONICAL_READS = 5  # min canonical junction reads at anchor to trust PSI


def is_alt5(row: pd.Series) -> bool:
    """Return True if junction is an alt 5' (alt donor) event.

    :param row: A row from a junction DataFrame with 'strand', 'start_site_category',
        and 'end_site_category' columns.
    :return: True if the junction is an alt 5' (alt donor) event.
    """
    if row["strand"] == "+":
        return (
            row["start_site_category"] == "novel"
            and row["end_site_category"] == "known"
        )
    else:
        return (
            row["start_site_category"] == "known"
            and row["end_site_category"] == "novel"
        )


def has_any_novel_site(row: pd.Series) -> bool:
    """Return True if junction has at least one novel splice site.

    :param row: A row from a junction DataFrame with 'start_site_category' and
        'end_site_category' columns.
    :return: True if at least one splice site is novel.
    """
    return row["start_site_category"] == "novel" or row["end_site_category"] == "novel"


def get_anchor(row: pd.Series) -> str:
    """Return the anchor coordinate (known splice site) for PSI grouping.

    :param row: A row from a junction DataFrame with 'strand', 'chrom',
        'genomic_start_coord', and 'genomic_end_coord' columns.
    :return: Anchor coordinate string in the format 'chrom:coord'.
    """
    if row["strand"] == "+":
        return f"{row['chrom']}:{row['genomic_end_coord']}"
    else:
        return f"{row['chrom']}:{row['genomic_start_coord']}"


def load_sample_junctions(file_path: Path) -> pd.DataFrame:
    """Load a single sample's junction count TSV.

    :param file_path: Path to a tab-separated junction count file.
    :return: DataFrame with junction count data.
    """
    df = pd.read_csv(file_path, sep="\t")
    return df


def compute_psi_and_anchors(
    all_junctions_df: pd.DataFrame,
    alt5_junctions: set[str],
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """Compute true PSI for each alt 5' junction and return anchor totals.

    :param all_junctions_df: DataFrame containing all junctions for a sample, with columns
        'junction_key', 'junction_unique_read_counts', 'start_site_category',
        and 'end_site_category'.
    :param alt5_junctions: Set of junction keys identified as alt 5' events.
    :return: Tuple of (psi, anchor_totals, canonical_at_anchor) where psi maps
        junction_key to PSI, anchor_totals maps anchor to total reads, and
        canonical_at_anchor maps anchor to canonical junction reads.
    """
    df = all_junctions_df.copy()
    df["anchor"] = df.apply(get_anchor, axis=1)

    anchor_totals = df.groupby("anchor")["junction_unique_read_counts"].sum().to_dict()

    canonical_mask = (df["start_site_category"] == "known") & (
        df["end_site_category"] == "known"
    )
    canonical_at_anchor = (
        df[canonical_mask]
        .groupby("anchor")["junction_unique_read_counts"]
        .sum()
        .to_dict()
    )

    psi = {}
    alt5_rows = df[df["junction_key"].isin(alt5_junctions)]
    for _, row in alt5_rows.iterrows():
        jk = row["junction_key"]
        reads = row["junction_unique_read_counts"]
        total = anchor_totals.get(row["anchor"], 0)
        if total > 0:
            psi[jk] = reads / total
        else:
            psi[jk] = 0.0

    return psi, anchor_totals, canonical_at_anchor


def logit(psi: float, eps: float = PSI_EPS) -> float:
    """Logit-transform a PSI value, clamping to [eps, 1-eps].

    :param psi: PSI value in [0, 1].
    :param eps: Offset used to clamp values away from 0 and 1.
    :return: Logit-transformed PSI value.
    """
    p = np.clip(psi, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_junction_matrix(
    data_dir: Path, min_reads: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build alt 5' junction x sample count and PSI matrices.

    :param data_dir: Directory containing per-sample *_all_junctions.tsv files.
    :param min_reads: Minimum junction_unique_read_counts to consider a junction present.
    :return: Tuple of (counts_df, psi_df, junction_meta_df, all_novel_df) containing
        alt 5' read count matrix, true PSI matrix, junction metadata, and all
        novel-site junction count matrix.
    """
    sample_files = sorted(data_dir.glob("*_all_junctions.tsv"))
    log.info("Found %d sample files in %s", len(sample_files), data_dir)

    all_counts: dict[str, dict[str, int]] = {}
    all_psi: dict[str, dict[str, float]] = {}
    junction_meta: dict[str, dict] = {}
    junction_anchors: dict[str, str] = {}  # junction_key -> anchor
    all_novel_counts: dict[str, dict[str, int]] = {}
    # Per-sample anchor totals and canonical reads for absence/quality handling
    sample_anchor_totals: dict[str, dict[str, int]] = {}
    sample_canonical_at_anchor: dict[str, dict[str, int]] = {}

    for f in sample_files:
        sample_id = f.name.replace("_all_junctions.tsv", "")
        log.info("  Processing %s", sample_id)

        df = load_sample_junctions(f)
        if df.empty:
            continue

        df["is_alt5"] = df.apply(is_alt5, axis=1)
        df["has_novel_site"] = df.apply(has_any_novel_site, axis=1)
        alt5_keys = set(df.loc[df["is_alt5"], "junction_key"])

        psi_dict, anchor_totals, canonical_reads = compute_psi_and_anchors(
            df, alt5_keys
        )
        sample_anchor_totals[sample_id] = anchor_totals
        sample_canonical_at_anchor[sample_id] = canonical_reads

        alt5_df = df[df["is_alt5"]].copy()
        for _, row in alt5_df.iterrows():
            jk = row["junction_key"]
            reads = int(row["junction_unique_read_counts"])

            all_counts.setdefault(jk, {})[sample_id] = reads
            if jk in psi_dict:
                all_psi.setdefault(jk, {})[sample_id] = psi_dict[jk]

            if jk not in junction_meta:
                junction_meta[jk] = {
                    "chrom": row["chrom"],
                    "strand": row["strand"],
                    "genomic_start_coord": row["genomic_start_coord"],
                    "genomic_end_coord": row["genomic_end_coord"],
                }
                if row["strand"] == "+":
                    junction_anchors[jk] = f"{row['chrom']}:{row['genomic_end_coord']}"
                else:
                    junction_anchors[jk] = (
                        f"{row['chrom']}:{row['genomic_start_coord']}"
                    )

        novel_df = df[df["has_novel_site"]].copy()
        for _, row in novel_df.iterrows():
            jk = row["junction_key"]
            reads = int(row["junction_unique_read_counts"])
            all_novel_counts.setdefault(jk, {})[sample_id] = reads

    if not all_counts:
        log.warning("No alt 5' junctions found across any samples")
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    sample_ids = sorted({s for d in all_counts.values() for s in d})

    count_rows = []
    psi_rows = []
    for jk in sorted(all_counts):
        crow = {s: all_counts[jk].get(s, 0) for s in sample_ids}
        crow["junction_key"] = jk
        count_rows.append(crow)

        # PSI with absence handling + canonical reads filter
        anchor = junction_anchors[jk]
        prow = {"junction_key": jk}
        for s in sample_ids:
            canonical = sample_canonical_at_anchor.get(s, {}).get(anchor, 0)
            if jk in all_psi and s in all_psi[jk]:
                # Junction present — but only trust PSI if canonical reads >= threshold
                if canonical >= MIN_CANONICAL_READS:
                    prow[s] = all_psi[jk][s]
                else:
                    prow[s] = np.nan  # can't trust PSI without canonical reference
            else:
                # Junction absent — check anchor coverage
                anchor_total = sample_anchor_totals.get(s, {}).get(anchor, 0)
                if (
                    anchor_total >= MIN_ANCHOR_READS
                    and canonical >= MIN_CANONICAL_READS
                ):
                    prow[s] = 0.0  # truly absent, PSI = 0
                else:
                    prow[s] = np.nan  # low coverage, uninformative
        psi_rows.append(prow)

    counts_df = pd.DataFrame(count_rows).set_index("junction_key")
    counts_df = counts_df[sample_ids]

    psi_df = pd.DataFrame(psi_rows).set_index("junction_key")
    psi_df = psi_df[sample_ids]

    meta_rows = []
    for jk in counts_df.index:
        n_present = int((counts_df.loc[jk] >= min_reads).sum())
        meta_rows.append(
            {
                "junction_key": jk,
                **junction_meta[jk],
                "n_samples_present": n_present,
            }
        )
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
    # Among present junctions (reads > 0), how many were set to NaN due to low canonical
    present_mask = counts_df > 0
    n_present_nan = int((present_mask & psi_df.isna()).sum().sum())
    log.info(
        "Alt 5' junction matrix: %d junctions x %d samples",
        len(counts_df),
        len(sample_ids),
    )
    log.info(
        "PSI quality: %d/%d cells valid (%.1f%%), %d NaN "
        "(canonical < %d: %d present junctions masked)",
        n_valid,
        n_total,
        100 * n_valid / n_total,
        n_nan,
        MIN_CANONICAL_READS,
        n_present_nan,
    )
    log.info("All novel-site junction matrix: %d junctions", len(all_novel_df))
    return counts_df, psi_df, junction_meta_df, all_novel_df


def categorize_junctions(
    psi_df: pd.DataFrame,
) -> dict[str, str]:
    """Categorise each junction as 'novel' or 'shared' by observability.

    :param psi_df: PSI matrix with junctions as rows and samples as columns. NaN indicates
        the junction is not observable in that sample.
    :return: Mapping of junction_key to category ('novel' or 'shared').
    """
    categories: dict[str, str] = {}
    for jk in psi_df.index:
        n_observable = int(psi_df.loc[jk].notna().sum())
        categories[jk] = "novel" if n_observable <= 2 else "shared"
    return categories


def compute_psi_outliers(
    psi_df: pd.DataFrame,
    min_samples: int = 3,
) -> pd.DataFrame:
    """Compute robust z-score of logit-transformed PSI across samples per junction.

    :param psi_df: PSI matrix with junctions as rows and samples as columns.
    :param min_samples: Minimum number of non-NaN samples required to compute z-scores
        for a junction.
    :return: Same-shape DataFrame of robust z-scores (NaN where PSI is NaN).
    """
    zscore_df = pd.DataFrame(np.nan, index=psi_df.index, columns=psi_df.columns)

    for jk in psi_df.index:
        row = psi_df.loc[jk]
        valid_mask = row.notna()
        valid = row[valid_mask]
        if len(valid) < min_samples:
            continue

        logit_vals = valid.apply(logit)

        med = logit_vals.median()
        mad = median_abs_deviation(logit_vals.values, nan_policy="omit")

        if mad == 0:
            # All values identical after logit → no variation
            zscore_df.loc[jk, valid.index] = 0.0
        else:
            zscore_df.loc[jk, valid.index] = (logit_vals - med) / (1.4826 * mad)

    return zscore_df


def compute_sample_metrics(
    psi_zscore_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    junction_categories: dict[str, str],
    all_novel_df: pd.DataFrame,
    min_reads: int,
) -> pd.DataFrame:
    """Compute shift and novel metrics per sample.

    :param psi_zscore_df: Robust z-score matrix for PSI values (junctions x samples).
    :param counts_df: Alt 5' read count matrix (junctions x samples).
    :param junction_categories: Mapping of junction_key to 'novel' or 'shared'.
    :param all_novel_df: Count matrix for all novel-site junctions (junctions x samples).
    :param min_reads: Minimum read count to consider a junction present.
    :return: Per-sample metrics including n_shared_present, n_psi_outliers,
        n_novel_alt5, n_novel_any, and total_alt5.
    """
    samples = counts_df.columns.tolist()
    shared_junctions = [
        jk for jk, cat in junction_categories.items() if cat == "shared"
    ]
    novel_alt5_junctions = [
        jk for jk, cat in junction_categories.items() if cat == "novel"
    ]

    # Categorize ALL novel-site junctions as rare (1-2 samples)
    all_novel_rare = [
        jk
        for jk in all_novel_df.index
        if int((all_novel_df.loc[jk] >= min_reads).sum()) <= 2
    ]

    rows = []
    for s in samples:
        # Shift signal — directional: z >= 2 (elevated alt PSI)
        z_vals = (
            psi_zscore_df.loc[shared_junctions, s].dropna()
            if shared_junctions
            else pd.Series(dtype=float)
        )
        n_shared_present = len(z_vals)
        n_psi_outliers = int((z_vals >= 2).sum()) if len(z_vals) > 0 else 0

        n_novel_alt5 = (
            int((counts_df.loc[novel_alt5_junctions, s] >= min_reads).sum())
            if novel_alt5_junctions
            else 0
        )

        n_novel_any = (
            int((all_novel_df.loc[all_novel_rare, s] >= min_reads).sum())
            if all_novel_rare
            else 0
        )

        total_alt5 = int((counts_df[s] >= min_reads).sum())

        rows.append(
            {
                "sample_id": s,
                "n_shared_present": n_shared_present,
                "n_psi_outliers": n_psi_outliers,
                "n_novel_alt5": n_novel_alt5,
                "n_novel_any": n_novel_any,
                "total_alt5": total_alt5,
            }
        )

    return pd.DataFrame(rows)


DEFAULT_LRAA_EXPR_DIR = Path(DATA_DIR) / "tx_expr"


def compute_n_expressed_genes(
    lraa_expr_dir: Path,
    sample_ids: list[str],
    tpm_threshold: float = 1.0,
) -> pd.DataFrame:
    """Count expressed genes per sample from LRAA expression files.

    :param lraa_expr_dir: Directory containing per-sample LRAA quant.expr files.
    :param sample_ids: List of sample IDs to process.
    :param tpm_threshold: Minimum TPM to consider a gene expressed.
    :return: DataFrame with columns 'sample_id' and 'n_expressed'.
    """
    rows: list[dict] = []

    for sample_id in sample_ids:
        expr_path = lraa_expr_dir / f"{sample_id}.LRAA.quant.expr"
        if not expr_path.exists():
            log.warning("LRAA expr not found for %s — skipping", sample_id)
            continue

        df = pd.read_csv(expr_path, sep="\t", usecols=["gene_id", "TPM"])
        n_expressed = int(df[df["TPM"] > tpm_threshold]["gene_id"].nunique())
        rows.append({"sample_id": sample_id, "n_expressed": n_expressed})
        log.info("  %s: n_expressed=%d", sample_id, n_expressed)

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
    :param min_observable: Minimum number of samples where the junction must be observable
        (non-NaN PSI) to be considered truly novel.
    :param max_present: Maximum number of samples where the junction is present (>= min_reads)
        to qualify as truly novel.
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
        len(truly_novel),
        len(psi_df),
        min_observable,
        max_present,
    )

    sample_ids = counts_df.columns.tolist()
    rows = []
    for s in sample_ids:
        n = int((counts_df.loc[truly_novel, s] >= min_reads).sum())
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
        slope,
        intercept,
        r_value**2,
        p_value,
    )

    enrich_df = df.loc[
        mask,
        ["sample_id", "n_truly_novel_alt5", "n_expressed", "gene_enrichment_z"],
    ].copy()
    enrich_path = output_dir / "alt5_gene_enrichment.tsv"
    enrich_df.to_csv(enrich_path, sep="\t", index=False)
    log.info("Saved %s", enrich_path)

    return df


def _sample_color(sample: str) -> str:
    """Return bar color based on whether sample is a proband or parent.

    :param sample: Sample ID string.
    :return: 'coral' for probands (containing '_3_R1'), 'steelblue' for parents.
    """
    return "coral" if "_3_R1" in sample else "steelblue"


def _add_proband_legend(ax: plt.Axes) -> None:
    """Add a proband/parent color legend to a matplotlib axes.

    :param ax: Matplotlib axes to add the legend to.
    """
    legend_elements = [
        Patch(facecolor="coral", edgecolor="black", label="Proband"),
        Patch(facecolor="steelblue", edgecolor="black", label="Parent"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def plot_ranked_shift(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate ranked bar chart of directional PSI outlier count (shift signal).

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id' and 'n_psi_outliers'.
    :param output_dir: Directory to save the output PNG file.
    """
    df = metrics_df.sort_values("n_psi_outliers", ascending=False)
    colors = [_sample_color(s) for s in df["sample_id"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        range(len(df)),
        df["n_psi_outliers"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    labels = [s.replace("_R1", "") for s in df["sample_id"].values]
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Number of alt 5' ss outliers")
    ax.set_title("Alt 5' usage shift in existing isoforms")

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()
    ax.axhline(
        outlier_mean + 2 * outlier_std,
        color="red",
        linestyle="--",
        linewidth=2,
        label="+2 SD",
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    from matplotlib.lines import Line2D

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


def plot_ranked_novel(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate ranked bar chart of alt 5' fraction of rare junctions.

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id', 'n_novel_alt5',
        and 'n_novel_any' columns.
    :param output_dir: Directory to save the output PNG file.
    """
    df = metrics_df.copy()
    df["novel_ratio"] = df["n_novel_alt5"] / df["n_novel_any"].replace(0, np.nan)
    df = df.sort_values("novel_ratio", ascending=False)
    colors = [_sample_color(s) for s in df["sample_id"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        range(len(df)),
        df["novel_ratio"].values * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    labels = [s.replace("_R1", "") for s in df["sample_id"].values]
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Alt 5' fraction of rare junctions (%)")
    ax.set_title("Alt 5' Novel Signal — fraction of rare junctions that are alt 5'")
    _add_proband_legend(ax)
    plt.tight_layout()
    out = output_dir / "alt5_novel_ranked_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def plot_shift_vs_novel(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate 2D scatter of gene_enrichment_z vs n_psi_outliers.

    :param metrics_df: Per-sample metrics DataFrame with 'sample_id', 'n_psi_outliers',
        and optionally 'gene_enrichment_z' columns.
    :param output_dir: Directory to save the output PNG file.
    """
    if "gene_enrichment_z" not in metrics_df.columns:
        log.warning("gene_enrichment_z not in metrics — skipping plot")
        return

    df = metrics_df.dropna(subset=["gene_enrichment_z"]).copy()
    if df.empty:
        log.warning("No valid gene_enrichment_z values — skipping plot")
        return

    colors = [_sample_color(s) for s in df["sample_id"]]

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        df["gene_enrichment_z"].values,
        df["n_psi_outliers"].values,
        c=colors,
        edgecolors="black",
        linewidth=0.5,
        s=60,
        zorder=3,
    )
    ax.set_xlabel("Novel alt 5' ss burden z-score")
    ax.set_ylabel("Alt 5' usage shift in existing isoforms")
    ax.set_title("Alt 5' splice site events")

    if outlier_std > 0:
        ax.axhline(
            outlier_mean + 2 * outlier_std, color="red", linestyle="--", linewidth=2
        )
    ax.axvline(3, color="blue", linestyle=":", linewidth=2)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    for _, row in df.iterrows():
        yz = (
            (row["n_psi_outliers"] - outlier_mean) / outlier_std
            if outlier_std > 0
            else 0
        )
        if abs(row["gene_enrichment_z"]) > 2 or abs(yz) > 2 or row["sample_id"] == "<target_sample>":
            ax.annotate(
                row["sample_id"].replace("_R1", ""),
                (row["gene_enrichment_z"], row["n_psi_outliers"]),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    from matplotlib.lines import Line2D

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


def main(
    data_dir: Path, min_reads: int, output_dir: Path | None, lraa_expr_dir: Path
) -> None:
    """Run alt 5' splice site shift analysis across samples.

    :param data_dir: Directory with *_all_junctions.tsv files.
    :param min_reads: Minimum junction_unique_read_counts to consider a junction present.
    :param output_dir: Output directory, or None to use default.
    :param lraa_expr_dir: Directory with per-sample LRAA quant.expr files.
    """
    if output_dir is None:
        output_dir = data_dir.parent / "alt5_shift_analysis_bam"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    counts_df, psi_df, junction_meta_df, all_novel_df = build_junction_matrix(
        data_dir, min_reads
    )
    if counts_df.empty:
        log.error("No data to analyse. Exiting.")
        return

    junction_categories = categorize_junctions(psi_df)
    n_shared = sum(1 for v in junction_categories.values() if v == "shared")
    n_novel = sum(1 for v in junction_categories.values() if v == "novel")
    log.info(
        "Junction categories: %d shared (observable in >= 3 samples), %d novel (observable in <= 2)",
        n_shared,
        n_novel,
    )

    psi_zscore_df = compute_psi_outliers(psi_df)

    metrics_df = compute_sample_metrics(
        psi_zscore_df, counts_df, junction_categories, all_novel_df, min_reads
    )

    truly_novel_df = compute_truly_novel_counts(psi_df, counts_df, min_reads)
    sample_ids = counts_df.columns.tolist()
    n_expressed_df = compute_n_expressed_genes(lraa_expr_dir, sample_ids)
    metrics_df = compute_gene_enrichment(
        metrics_df, truly_novel_df, n_expressed_df, output_dir
    )

    log.info(f"\n{'=' * 70}")
    log.info("Alt 5' Shift Analysis (robust logit-PSI z-scores, directional)")
    log.info(f"{'=' * 70}")
    log.info(f"Total alt 5' junctions: {len(counts_df)}")
    log.info(f"  Shared (observable in >= 3 samples): {n_shared}")
    log.info(f"  Novel  (observable in <= 2 samples): {n_novel}")
    log.info(f"Total novel-site junctions (any type): {len(all_novel_df)}")
    log.info(f"Samples: {len(counts_df.columns)}")
    log.info(f"Min reads threshold: {min_reads}")
    log.info(f"Min anchor reads for PSI=0: {MIN_ANCHOR_READS}")
    log.info(f"Min canonical reads to trust PSI: {MIN_CANONICAL_READS}")
    log.info(f"PSI logit offset: {PSI_EPS}")

    log.info("\n--- Shift signal (sorted by n_psi_outliers, z >= 2) ---")
    shift_sorted = metrics_df.sort_values("n_psi_outliers", ascending=False)
    for _, row in shift_sorted.iterrows():
        log.info(
            f"  {row['sample_id']:30s}  n_psi_outliers={row['n_psi_outliers']:4d}  "
            f"n_shared={row['n_shared_present']:4d}"
        )

    log.info("\n--- Novel signal (sorted by n_novel_alt5) ---")
    novel_sorted = metrics_df.sort_values("n_novel_alt5", ascending=False)
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

    metrics_df.to_csv(output_dir / "sample_shift_metrics.tsv", sep="\t", index=False)
    log.info("Saved sample_shift_metrics.tsv")

    counts_df.to_csv(output_dir / "junction_counts_matrix.tsv", sep="\t")
    log.info("Saved junction_counts_matrix.tsv")

    psi_df.to_csv(output_dir / "junction_psi_matrix.tsv", sep="\t")
    log.info("Saved junction_psi_matrix.tsv")

    plot_ranked_shift(metrics_df, output_dir)
    plot_ranked_novel(metrics_df, output_dir)
    plot_shift_vs_novel(metrics_df, output_dir)

    log.info(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run alt 5' splice site shift analysis across samples."
    )
    parser.add_argument("--data-dir", type=Path,
                        default=Path(DATA_DIR) / "outrider" / "all_junction_counts",
                        help="Directory with *_all_junctions.tsv files.")
    parser.add_argument("--min-reads", type=int, default=2,
                        help="Minimum junction_unique_read_counts to consider a junction present.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: DATA_DIR/../alt5_shift_analysis_bam/).")
    parser.add_argument("--lraa-expr-dir", type=Path, default=DEFAULT_LRAA_EXPR_DIR,
                        help="Directory with per-sample LRAA quant.expr files.")
    args = parser.parse_args()
    main(data_dir=args.data_dir, min_reads=args.min_reads, output_dir=args.output_dir,
         lraa_expr_dir=args.lraa_expr_dir)
