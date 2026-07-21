"""Compute and plot CDF of transcript detection as a function of transcript length."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
import subprocess
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import hailtop.batch as hb
import hailtop.fs as hfs
from pathlib import Path
from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids, read_gtf, read_sqanti3_annotated, create_symbolic_links
from rare_disease_lr_rnaseq.config import DOCKER_GCLOUD_R, GCS_REF_GTF, GCS_TMP_BUCKET

from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import read_from_airtable, RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID

logger = logging.getLogger(__name__)

COLORS = {
    'primary': '#4C72B0',
    'secondary': '#DD8452',
}

REGION = ["us-central1"]
DEFAULT_CPU = 2 ** 4
DEFAULT_MEMORY = "highmem"

NUM_SR_SAMPLES = 60
RANDOM_SEED = 42


def get_sr_transcripts(gtf: str, bam: str, output_file: str) -> str:
    """
    Generate R script to count unique read pairs per transcript for short-read data.

    :param gtf: Path to the GTF annotation file.
    :param bam: Path to the BAM alignment file.
    :param output_file: Path to the output TSV file for results.
    :return: R script as a string.
    """
    return f"""
    library(GenomicAlignments)
    library(GenomicFeatures)
    library(data.table)

    cat("Loading GTF...\\n")
    txdb <- makeTxDbFromGFF('{gtf}', format = "gtf")
    all_transcripts <- exonsBy(txdb, by="tx", use.names=TRUE)
    seqlevelsStyle(all_transcripts) <- "UCSC"

    # Pre-compute all transcript lengths (memory efficient)
    tx_lengths <- sum(width(all_transcripts))

    cat("Getting chromosomes from BAM...\\n")
    bam_header <- scanBamHeader("{bam}")
    chrom_lengths <- bam_header[[1]]$targets

    # Filter to standard chromosomes only
    standard_chroms <- paste0("chr", c(1:22, "X", "Y"))
    chromosomes <- intersect(names(chrom_lengths), standard_chroms)
    cat("Processing", length(chromosomes), "standard chromosomes\\n\\n")

    tx_by_chrom <- split(all_transcripts,
                     as.character(seqnames(unlist(range(all_transcripts)))))

    # Remove all_transcripts to free memory
    rm(all_transcripts); gc()

    # Write header to output file
    cat("transcript_id\\ttranscript_length\\tunique_read_count\\n", file = "{output_file}")
    total_tx_written <- 0

    for(chrom in chromosomes) {{
        cat("Processing", chrom, "\\n")

        chrom_length <- chrom_lengths[chrom]

        # Only primary alignments, paired and properly paired
        flag <- scanBamFlag(
            isSecondaryAlignment = FALSE,
            isSupplementaryAlignment = FALSE,
            isPaired = TRUE,
            isProperPair = TRUE
        )

        param <- ScanBamParam(
            which = GRanges(chrom, IRanges(1, chrom_length)),
            flag = flag
        )

        reads <- readGAlignmentPairs("{bam}", param=param, strandMode = 2)
        cat("  Primary read pairs:", length(reads), "\\n")

        if(length(reads) == 0) {{
            cat("  Skipping (no reads)\\n\\n")
            next
        }}

        # Convert pairs to GRangesList and name by pair index for tracking
        grl <- grglist(reads)
        names(grl) <- seq_along(grl)
        rm(reads); gc()

        # Union overlapping blocks within each pair, names preserved
        read_blocks <- unlist(union(grl, grl))
        rm(grl); gc()

        # Filter transcripts to this chromosome
        tx_on_chrom <- tx_by_chrom[[chrom]]
        cat("  Transcripts:", length(tx_on_chrom), "\\n")

        if(length(tx_on_chrom) == 0) {{
            rm(read_blocks); gc()
            cat("  Skipping (no transcripts)\\n\\n")
            next
        }}

        # Map to transcript coordinates (stranded)
        tx_reads <- mapToTranscripts(read_blocks, tx_on_chrom, ignore.strand = FALSE)
        cat("  Mapped ranges:", length(tx_reads), "\\n")

        if(length(tx_reads) == 0) {{
            rm(read_blocks, tx_on_chrom); gc()
            cat("  Skipping (no mapped reads)\\n\\n")
            next
        }}

        # Get original read pair index for each mapped range using xHits
        # Use data.table for memory efficiency
        dt <- data.table(
            pair_id = names(read_blocks)[mcols(tx_reads)$xHits],
            transcript = as.character(seqnames(tx_reads))
        )
        rm(tx_reads); gc()

        # Count unique read pairs per transcript
        dt <- unique(dt)
        read_counts <- dt[, .N, by = transcript]
        rm(dt); gc()

        cat("  Unique transcripts with reads:", nrow(read_counts), "\\n")

        # Build results for transcripts with reads only (memory efficient)
        tx_names_on_chrom <- names(tx_on_chrom)
        results <- data.table(
            transcript_id = read_counts$transcript,
            transcript_length = tx_lengths[read_counts$transcript],
            unique_read_count = read_counts$N
        )

        cat("  Total reads assigned:", sum(results$unique_read_count), "\\n")

        # Append to output file (no header)
        fwrite(results, file = "{output_file}", append = TRUE, sep = "\\t", col.names = FALSE)
        total_tx_written <- total_tx_written + nrow(results)

        cat("  Wrote", nrow(results), "transcripts\\n\\n")

        rm(read_blocks, tx_on_chrom, read_counts, results); gc()
    }}

    cat("All done! Wrote", total_tx_written, "transcripts total to {output_file}\\n")
    """


def compute_sr_batch_job(batch: hb.Batch, args: argparse.Namespace, bam_filepath: str, gtf_filepath: str, sample_id: str) -> None:
    """
    Submit a batch job to compute SR transcript read counts.

    :param batch: Hail Batch object for job submission.
    :param args: Parsed command-line arguments containing output path.
    :param bam_filepath: GCS path to the BAM file.
    :param gtf_filepath: GCS path to the GTF annotation file.
    :param sample_id: Identifier for the sample.
    """
    output_file = f"sr_{sample_id}_tx_read_counts.tsv"
    if hfs.is_file(f"{args.output}/{output_file}"):
        logger.info(f"  Skipping {sample_id} (already exists)")
        return

    cur_job = batch.new_job(f"sr_tx_read_counts_{sample_id}")
    cur_job.storage("30G")
    cur_job.command("cd /io")

    local_gtf_filepath = os.path.basename(gtf_filepath)
    local_bam_filepath = os.path.basename(bam_filepath)
    create_symbolic_links(cur_job, gtf_filepath, local_gtf_filepath)
    create_symbolic_links(cur_job, bam_filepath, local_bam_filepath)
    create_symbolic_links(cur_job, f"{bam_filepath}.bai", f"{local_bam_filepath}.bai")
    cur_job.command("ls -lh .")

    r_script = get_sr_transcripts(local_gtf_filepath, local_bam_filepath, output_file)
    cur_job.command(f"""cat > compute_tx_reads.R << 'RSCRIPT'
{r_script}
RSCRIPT
xvfb-run Rscript compute_tx_reads.R""")

    cur_job.command(f"gcloud storage cp {output_file} {args.output}/{output_file}")


def submit_sr_batch_jobs(batch: hb.Batch, args: argparse.Namespace) -> None:
    """
    Submit batch jobs for short-read samples from airtable.

    :param batch: Hail Batch object for job submission.
    :param args: Parsed command-line arguments containing output path.
    """
    short_read_dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    short_read_dat = short_read_dat[short_read_dat["imputed_tissue"] == "whole_blood"]
    short_read_dat = short_read_dat[~(short_read_dat["exclude"] == "yes")]
    short_read_dat = short_read_dat[~(short_read_dat["new_sample"] == "yes")]
    logger.info(f"Total eligible SR samples: {short_read_dat.shape[0]}")

    np.random.seed(RANDOM_SEED)
    sampled_indices = np.random.choice(len(short_read_dat), size=NUM_SR_SAMPLES, replace=False)
    short_read_dat = short_read_dat.iloc[sampled_indices]

    sample_ids = list(short_read_dat["sample_id"])
    bam_filepaths = list(short_read_dat["star_bam"])

    logger.info(f"Selected {len(sample_ids)} SR samples")
    for i, (sample_id, bam_filepath) in enumerate(zip(sample_ids, bam_filepaths)):
        logger.info(f"Submitting SR job {i+1}/{len(sample_ids)}: {sample_id}")
        compute_sr_batch_job(batch, args, bam_filepath, GCS_REF_GTF, sample_id)


def get_lr_transcript_read_counts(sample_id: str) -> pd.DataFrame:
    """
    Get transcript read counts for a long-read sample from SQANTI3 output.

    :param sample_id: Identifier for the long-read sample.
    :return: DataFrame with isoform, unique reads, associated transcript, and
        transcript length columns.
    """
    cur_gtf = read_gtf(sample_id)
    cur_transcripts_coords = cur_gtf[cur_gtf["feature"] == "transcript"]
    cur_annotated_transcripts = read_sqanti3_annotated(sample_id, rules_filter=False)

    logger.info(cur_annotated_transcripts.columns)
    annotated_use_cols = ["isoform", "uniq_reads", "associated_transcript"]
    cur_annotated_transcripts = cur_annotated_transcripts[annotated_use_cols]
    cur_annotated_transcripts = cur_annotated_transcripts.merge(
        cur_transcripts_coords[["transcript_id", "chrom", "start", "end"]],
        left_on="isoform",
        right_on="transcript_id",
        how="inner"
    )
    cur_annotated_transcripts["transcript_length"] = (
        cur_annotated_transcripts["end"] - cur_annotated_transcripts["start"] + 1
    )
    cur_annotated_transcripts.drop(columns=["chrom", "start", "end"], inplace=True)

    return cur_annotated_transcripts


def process_lr_samples() -> None:
    """
    Process all long-read samples and save transcript read counts.
    """
    sample_ids = get_long_read_sample_ids()
    logger.info(f"Processing {len(sample_ids)} LR samples")

    output_dir = f"{DATA_DIR}/lr_transcript_read_counts"
    os.makedirs(output_dir, exist_ok=True)

    for i, sample_id in enumerate(sample_ids):
        logger.info(f"Processing LR sample {i+1}/{len(sample_ids)}: {sample_id}")
        output_path = f"{output_dir}/{sample_id}_tx_read_counts.tsv"

        if os.path.exists(output_path):
            logger.info(f"  Skipping (already exists)")
            continue

        df = get_lr_transcript_read_counts(sample_id)
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"  Wrote {len(df)} transcripts")


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


def compute_cdf_at_lengths(df: pd.DataFrame, read_col: str, length_bins: np.ndarray) -> np.ndarray:
    """
    Compute CDF values at specified transcript length bins.

    :param df: DataFrame with transcript_length and read count columns.
    :param read_col: Column name for read counts.
    :param length_bins: Array of transcript lengths at which to evaluate CDF.
    :return: CDF values at each length bin.
    """
    df = df.sort_values('transcript_length').copy()

    total_reads = df[read_col].sum()
    if total_reads == 0:
        return np.zeros(len(length_bins))

    df['cumsum'] = df[read_col].cumsum()
    df['cdf'] = df['cumsum'] / total_reads

    cdf_values = np.interp(
        length_bins,
        df['transcript_length'].values,
        df['cdf'].values,
        left=0.0,
        right=1.0
    )

    return cdf_values


def load_sr_sample_cdfs(sr_data_dir: str, length_bins: np.ndarray) -> np.ndarray:
    """
    Load SR samples and compute CDF for each.

    :param sr_data_dir: Directory containing short-read transcript read count TSV files.
    :param length_bins: Array of transcript lengths at which to evaluate CDF.
    :return: CDF matrix of shape (n_samples, n_bins).
    """
    cdf_list = []
    files = list(Path(sr_data_dir).glob('sr_*_tx_read_counts.tsv'))

    for filepath in files:
        df = pd.read_csv(filepath, sep='\t')
        cdf = compute_cdf_at_lengths(df, 'unique_read_count', length_bins)
        cdf_list.append(cdf)

    logger.info(f"Loaded {len(cdf_list)} SR samples")
    return np.array(cdf_list)


def load_lr_sample_cdfs(lr_data_dir: str, length_bins: np.ndarray) -> np.ndarray:
    """
    Load LR samples and compute CDF for each.

    :param lr_data_dir: Directory containing long-read transcript read count TSV files.
    :param length_bins: Array of transcript lengths at which to evaluate CDF.
    :return: CDF matrix of shape (n_samples, n_bins).
    """
    cdf_list = []
    files = list(Path(lr_data_dir).glob('*_tx_read_counts.tsv'))

    for filepath in files:
        df = pd.read_csv(filepath, sep='\t')
        cdf = compute_cdf_at_lengths(df, 'uniq_reads', length_bins)
        cdf_list.append(cdf)

    logger.info(f"Loaded {len(cdf_list)} LR samples")
    return np.array(cdf_list)


def plot_cdf_with_error(sr_cdfs: np.ndarray, lr_cdfs: np.ndarray, length_bins: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot CDF of uniquely-mapped reads by transcript length with error shadows.

    :param sr_cdfs: Short-read CDF matrix of shape (n_samples, n_bins).
    :param lr_cdfs: Long-read CDF matrix of shape (n_samples, n_bins).
    :param length_bins: Transcript length values for x-axis.
    :param save_path: Path to save the figure. If None, saves to default location.
    """
    setup_publication_style()

    sr_mean = np.mean(sr_cdfs, axis=0)
    sr_std = np.std(sr_cdfs, axis=0)
    lr_mean = np.mean(lr_cdfs, axis=0)
    lr_std = np.std(lr_cdfs, axis=0)

    logger.info(f"SR CDFs shape: {sr_cdfs.shape}, LR CDFs shape: {lr_cdfs.shape}")
    logger.info(f"SR std - min: {sr_std.min():.6f}, max: {sr_std.max():.6f}, mean: {sr_std.mean():.6f}")
    logger.info(f"LR std - min: {lr_std.min():.6f}, max: {lr_std.max():.6f}, mean: {lr_std.mean():.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))

    line_lr, = ax.plot(length_bins, lr_mean, linewidth=2.5, color=COLORS['primary'])
    line_sr, = ax.plot(length_bins, sr_mean, linewidth=2.5, color=COLORS['secondary'])

    sr_99_idx = np.searchsorted(sr_mean, 0.99)
    sr_99_length = length_bins[min(sr_99_idx, len(length_bins) - 1)]

    ax.axvline(x=sr_99_length, color='red', linestyle='--', linewidth=1.5)
    ax.text(sr_99_length + 2000, 0.5, f'SR 99% at {int(sr_99_length)} bp',
            rotation=90, color='red', va='center', fontsize=10)

    lr_99_idx = np.searchsorted(length_bins, sr_99_length)
    lr_cdf_at_sr99 = lr_mean[min(lr_99_idx, len(lr_mean) - 1)]
    lr_reads_beyond = (1 - lr_cdf_at_sr99) * 100

    ax.axhline(y=lr_cdf_at_sr99, color='red', linestyle='--', linewidth=1.5)

    ax.text(100000, 0.82,
            f'{lr_reads_beyond:.2f}% unique long reads are from LR transcripts > SR 99% length',
            fontsize=10, color='#333333')

    ax.set_xlabel('Transcript Length (bp)', fontweight='bold')
    ax.set_ylabel('Cumulative Fraction of Reads', fontweight='bold')
    ax.set_xlim(0, length_bins[-1])
    ax.set_ylim(0, 1.05)

    ax.legend([line_lr, line_sr], ['Long Read', 'Short Read'],
              loc='lower right', framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = f"{DATA_DIR}/cdf_by_tx_length.png"

    plt.savefig(save_path)
    plt.savefig(save_path.replace('.png', '.pdf'))
    logger.info(f"Saved figure to {save_path}")
    plt.show()


def analyze_results(args: argparse.Namespace) -> None:
    """
    Download data and generate CDF plot.

    :param args: Parsed command-line arguments containing output path.
    """
    sr_data_dir = f"{DATA_DIR}/sr_transcript_read_counts"
    os.makedirs(sr_data_dir, exist_ok=True)
    existing_sr_files = list(Path(sr_data_dir).glob('sr_*_tx_read_counts.tsv'))
    logger.info(f"Found {len(existing_sr_files)} SR files, skipping download")

    lr_data_dir = f"{DATA_DIR}/lr_transcript_read_counts"
    if not os.path.exists(lr_data_dir) or len(list(Path(lr_data_dir).glob('*.tsv'))) == 0:
        process_lr_samples()

    length_bins = np.linspace(0, 350000, 1000)

    sr_cdfs = load_sr_sample_cdfs(sr_data_dir, length_bins)
    lr_cdfs = load_lr_sample_cdfs(lr_data_dir, length_bins)

    plot_cdf_with_error(sr_cdfs, lr_cdfs, length_bins)


def main(batch: hb.Batch, args: argparse.Namespace) -> None:
    """
    Submit batch jobs for SR samples.

    :param batch: Hail Batch object for job submission.
    :param args: Parsed command-line arguments.
    """
    submit_sr_batch_jobs(batch, args)
    batch.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--billing-project", type=str, default="tgg-rare-disease")
    parser.add_argument("--requester-pays-project", type=str, default="cmg-analysis")
    parser.add_argument("--output", type=str,
                        default=f"{GCS_TMP_BUCKET}/long_read/cdf_tx_length")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze results instead of running batch jobs.")
    args = parser.parse_args()

    if args.analyze:
        analyze_results(args)
    else:
        backend = hb.ServiceBackend(
            billing_project=args.billing_project,
            remote_tmpdir=args.output,
            regions=REGION
        )
        batch = hb.Batch(
            backend=backend,
            name="cdf_tx_length",
            requester_pays_project=args.requester_pays_project,
            default_image=DOCKER_GCLOUD_R,
            default_cpu=DEFAULT_CPU,
            default_memory=DEFAULT_MEMORY
        )
        main(batch, args)
