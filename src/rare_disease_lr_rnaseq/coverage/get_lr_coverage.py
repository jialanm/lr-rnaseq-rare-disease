"""Compute binned transcript coverage profiles from long-read RNA-seq BAMs on Hail Batch."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from typing import Optional

import hailtop.batch as hb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import (
    DATA_PATHS_TABLE_ID,
    DATA_PATHS_VIEW_ID,
    RNA_SEQ_BASE_ID,
    read_from_airtable,
)

from rare_disease_lr_rnaseq.config import (
    DOCKER_GCLOUD_R,
    GCS_BAM_DIR,
    GCS_REF_GTF,
    GCS_TMP_BUCKET,
    MENDELIAN_GENE_DISEASE_TABLE_FILEPATH,
)
from rare_disease_lr_rnaseq.utils import DATA_DIR, create_symbolic_links, get_long_read_sample_ids

logger = logging.getLogger(__name__)

REGION = ["us-central1"]
DEFAULT_CPU = 4
DEFAULT_MEMORY = "highmem"
N_BINS = 100

GTF_FILEPATH = GCS_REF_GTF
CLINGEN_EVIDENCE = {"Definitive", "Strong", "Moderate"}


def get_da_gene_ids() -> set[str]:
    """
    Return Ensembl gene IDs for disease-associated genes.

    :return: Set of Ensembl gene IDs for disease-associated genes.
    """
    df = pd.read_csv(
        MENDELIAN_GENE_DISEASE_TABLE_FILEPATH,
        sep="\t",
        usecols=["gene_id", "CLINGEN_classification"],
    )
    mask = df["CLINGEN_classification"].apply(
        lambda x: all(v.strip() in CLINGEN_EVIDENCE for v in str(x).split(";"))
    )
    return set(df.loc[mask, "gene_id"].tolist())

def bin_coverage(coverage_array: np.ndarray, n_bins: int) -> list[float]:
    """
    Bin a coverage array into a fixed number of equal-width bins.

    :param coverage_array: Per-base coverage values for a transcript.
    :param n_bins: Number of bins to divide the coverage array into.
    :return: Mean coverage value within each bin.
    """
    length = len(coverage_array)
    if length == 0:
        return [0.0] * n_bins

    bin_size = length / n_bins
    binned = []

    for i in range(n_bins):
        start_idx = int(i * bin_size)
        end_idx = int((i + 1) * bin_size)
        end_idx = min(end_idx, length)

        if start_idx < end_idx:
            binned.append(np.mean(coverage_array[start_idx:end_idx]))
        else:
            binned.append(0.0)

    return binned


def plot_coverage_profile(sample_means: np.ndarray, title: str, show_error: bool = True) -> None:
    """
    Plot the average coverage profile across samples.

    :param sample_means: 2-D array of shape (n_samples, n_bins) with per-sample coverage profiles.
    :param title: Title for the plot.
    :param show_error: Whether to show a shaded standard-deviation band.
    """
    overall_mean = np.mean(sample_means, axis=0)  # (bins,)
    overall_mean = overall_mean / np.max(overall_mean)

    plt.figure(figsize=(10, 8))
    plt.plot(overall_mean, linewidth=2, color='blue')

    if show_error:
        sample_std = np.std(sample_means, axis=0)
        x = np.arange(len(overall_mean))
        plt.fill_between(x, overall_mean - sample_std, overall_mean + sample_std,
                         alpha=0.3)

    plt.ylabel('Coverage', fontsize=14)
    plt.ylim(bottom=0)
    plt.title(title)
    plt.xticks([0, sample_means.shape[1] - 1], ["5'", "3'"], fontsize=12)
    plt.show()


def compute_final_coverage_workflow(
    gtf: str,
    bam: str,
    output_file: str,
    min_tx_length: int = 200,
    n_bins: int = 100,
    da_gene_ids_file: Optional[str] = None,
) -> str:
    """
    Generate an R script that computes a binned normalized coverage profile for long-read data.

    :param gtf: Path to the reference GTF annotation file.
    :param bam: Path to the long-read BAM file.
    :param output_file: Path where the coverage TSV will be written.
    :param min_tx_length: Minimum transcript length to include.
    :param n_bins: Number of bins for the coverage profile.
    :param da_gene_ids_file: Path to a file listing disease-associated gene IDs for filtering.
    :return: R script as a string.
    """
    if da_gene_ids_file:
        da_filter_block = f"""
    # Filter transcripts to disease-associated genes
    library(rtracklayer)
    cat("Parsing GTF for gene-transcript mapping...\\n")
    gtf_gr <- import("{gtf}", format="gtf")
    tx_rows <- gtf_gr[gtf_gr$type == "transcript"]
    gene_tx_map <- data.frame(
        tx_name = tx_rows$transcript_id,
        gene_id = sub("\\\\.[0-9]+$", "", tx_rows$gene_id),
        stringsAsFactors = FALSE
    )
    da_genes <- readLines("{da_gene_ids_file}")
    da_genes <- sub("\\\\.[0-9]+$", "", da_genes)
    da_tx_names <- gene_tx_map$tx_name[gene_tx_map$gene_id %in% da_genes]
    keep <- intersect(names(all_transcripts), da_tx_names)
    all_transcripts <- all_transcripts[keep]
    tx_lengths <- tx_lengths[names(all_transcripts)]
    cat("After DA-gene filter:", length(all_transcripts), "transcripts\\n")
"""
    else:
        da_filter_block = ""

    return f"""
    library(GenomicAlignments)
    library(GenomicFeatures)
    library(Rsamtools)
    library(data.table)

    # 1. Load Annotation and Filter for Protein Coding
    cat("Loading and filtering GTF...\\n")
    txdb <- suppressWarnings(makeTxDbFromGFF('{gtf}', format = "gtf"))

    # Extract metadata including strand for reversal logic
    tx_metadata <- as.data.frame(transcripts(txdb, columns=c("tx_name", "tx_type")))
    setDT(tx_metadata)

    all_transcripts <- exonsBy(txdb, by="tx", use.names=TRUE)

    # Filter by length
    tx_lengths <- sum(width(all_transcripts))
    all_transcripts <- all_transcripts[tx_lengths >= {min_tx_length}]
    tx_lengths <- tx_lengths[names(all_transcripts)]
{da_filter_block}
    cat("Processing", length(all_transcripts), "protein-coding transcripts...\\n")

    # 2. Chromosome Style Sync
    bf <- BamFile("{bam}")
    bam_chroms <- names(scanBamHeader(bf)$targets)
    if(any(grepl("chr", bam_chroms)) && !any(grepl("chr", seqlevels(all_transcripts)))) {{
        seqlevelsStyle(all_transcripts) <- "UCSC"
    }} else if(!any(grepl("chr", bam_chroms)) && any(grepl("chr", seqlevels(all_transcripts)))) {{
        seqlevelsStyle(all_transcripts) <- "Ensembl"
    }}

    # 3. Process by Chromosome
    global_profile_sum <- numeric({n_bins})
    total_tx_counted <- 0
    common_chroms <- intersect(seqlevels(all_transcripts), bam_chroms)

    for(chrom in common_chroms) {{
        cat("Processing", chrom, "\\n")
        tx_on_chrom <- all_transcripts[seqnames(unlist(range(all_transcripts))) == chrom]
        if(length(tx_on_chrom) == 0) next

        param <- ScanBamParam(
            which = GRanges(chrom, IRanges(1, scanBamHeader(bf)$targets[chrom])),
            flag = scanBamFlag(isSecondaryAlignment=FALSE, isSupplementaryAlignment=FALSE)
        )
        reads <- readGAlignments("{bam}", param=param)
        if(length(reads) == 0) next

        # Bridge Deletions (D) but keep Intron gaps (N)
        read_segments <- unlist(grglist(reads, drop.D.ranges = FALSE))

        # Map segments to spliced transcripts
        tx_hits <- mapToTranscripts(x = read_segments, transcripts = tx_on_chrom)

        if(length(tx_hits) > 0) {{
            seqlengths(tx_hits) <- tx_lengths[levels(seqnames(tx_hits))]
            chunk_cov <- coverage(tx_hits)

            for(tn in names(chunk_cov)) {{
                cov_rle <- chunk_cov[[tn]]
                full_len <- tx_lengths[tn]
                if(full_len < {n_bins}) next
                
                # Resample into {n_bins} bins
                bins <- seq(1, full_len + 1, length.out = {n_bins + 1})
                bin_means <- as.numeric(aggregate(cov_rle, FUN=mean, 
                                               start=head(bins, -1), 
                                               end=tail(bins, -1) - 1))
                
                # FIX 2: Handle Strand Reversal (5' -> 3')
                # Check strand from metadata
                t_strand <- tx_metadata[tx_name == tn, as.character(strand)]
                if(length(t_strand) > 0 && t_strand == "-") {{
                    bin_means <- rev(bin_means)
                }}

                # FIX 3: Equal Weighting (Normalize per transcript)
                max_val <- max(bin_means)
                if(!is.na(max_val) && max_val > 0) {{
                    norm_bins <- bin_means / max_val
                    global_profile_sum <- global_profile_sum + norm_bins
                    total_tx_counted <- total_tx_counted + 1
                }}
            }}
        }}
        rm(reads, read_segments, tx_hits); gc()
    }}

    # 4. Final Export
    if(total_tx_counted > 0) {{
        final_vector <- global_profile_sum / total_tx_counted
        output_data <- data.frame(bin = 1:{n_bins}, relative_coverage = final_vector)
        write.table(output_data, file = "{output_file}", 
                    sep = "\\t", quote = FALSE, row.names = FALSE)
        cat("Finished. Processed", total_tx_counted, "transcripts equally weighted.\\n")
    }} else {{
        stop("Error: No transcripts could be mapped.")
    }}
    """

def compute_coverage_batch_job(
    batch: hb.Batch,
    args: argparse.Namespace,
    bam_filepath: str,
    gtf_filepath: str,
    sample_id: str,
    prefix: str,
    da_gene_ids: Optional[set[str]] = None,
) -> None:
    """
    Create and configure a Hail Batch job to compute long-read coverage for one sample.

    :param batch: The Hail Batch to add the job to.
    :param args: Parsed command-line arguments containing output directory.
    :param bam_filepath: GCS path to the BAM file.
    :param gtf_filepath: GCS path to the GTF annotation file.
    :param sample_id: Sample identifier.
    :param prefix: Output file prefix (e.g. "lr", "lr_da").
    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    cur_job = batch.new_job(f"{prefix}_compute_coverage_{sample_id}")
    cur_job.cpu(8 if da_gene_ids else DEFAULT_CPU)
    cur_job.storage("15G" if da_gene_ids else "20G")
    cur_job.command("cd /io")

    local_gtf_filepath = os.path.basename(gtf_filepath)
    local_bam_filepath = os.path.basename(bam_filepath)
    create_symbolic_links(cur_job, gtf_filepath, local_gtf_filepath)
    create_symbolic_links(cur_job, bam_filepath, local_bam_filepath)
    create_symbolic_links(cur_job, f"{bam_filepath}.bai", f"{local_bam_filepath}.bai")
    cur_job.command("ls -lh .")

    da_gene_ids_file = None
    if da_gene_ids:
        da_gene_ids_file = "da_gene_ids.txt"
        gene_ids_str = "\n".join(sorted(da_gene_ids))
        cur_job.command(f"""cat > {da_gene_ids_file} << 'GENELIST'
{gene_ids_str}
GENELIST
wc -l {da_gene_ids_file}""")

    output_file = f"{prefix}_{sample_id}_coverage.csv"
    r_script = compute_final_coverage_workflow(local_gtf_filepath, local_bam_filepath, output_file, da_gene_ids_file=da_gene_ids_file)
    cur_job.command(f"""cat > compute_coverage.R << 'RSCRIPT'
{r_script}
RSCRIPT
xvfb-run Rscript compute_coverage.R""")

    cur_job.command(f"gcloud storage cp {output_file} {args.output}/{output_file}")



def compute_coverage_lr(
    batch: hb.Batch,
    args: argparse.Namespace,
    da_gene_ids: Optional[set[str]] = None,
) -> None:
    """
    Submit batch jobs for all long-read samples.

    :param batch: The Hail Batch to add jobs to.
    :param args: Parsed command-line arguments.
    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    sample_ids = get_long_read_sample_ids()
    prefix = "lr_da" if da_gene_ids else "lr"

    for cur_sample in sample_ids:
        cur_bam_filepath = f"{GCS_BAM_DIR}/{cur_sample}.aligned.sorted.bam"

        logger.info(f"Submitting job for {cur_sample}: {cur_bam_filepath}")
        compute_coverage_batch_job(batch, args, cur_bam_filepath, GTF_FILEPATH, cur_sample, prefix, da_gene_ids=da_gene_ids)


def main(da_gene_ids: Optional[set[str]] = None) -> None:
    """
    Submit long-read coverage batch jobs and run the batch.

    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    compute_coverage_lr(batch, args, da_gene_ids=da_gene_ids)
    batch.run()


def analyze_results(args: argparse.Namespace) -> None:
    """
    Download and analyze coverage results from GCS.

    :param args: Parsed command-line arguments containing the output GCS directory.
    """
    cur_data_dir = f"{DATA_DIR}/lr_coverage"
    os.makedirs(cur_data_dir, exist_ok=True)
    os.system(f"""gcloud storage cp "{args.output}/lr_*_coverage.csv" {cur_data_dir}""")

    # Each CSV already contains a pre-normalized 100-bin coverage profile
    normalized_tx_all = []
    for fname in os.listdir(cur_data_dir):
        if fname.endswith("_coverage.csv"):
            filepath = os.path.join(cur_data_dir, fname)
            print(f"Reading coverage for {filepath}...")
            df = pd.read_table(filepath)
            print(df)
            normalized_tx_all.append(df["relative_coverage"].values)

    if normalized_tx_all:
        normalized_matrix = np.array(normalized_tx_all)
        print(f"Normalized coverage matrix shape: {normalized_matrix.shape}")
        plot_coverage_profile(normalized_matrix,
                              title="Transcript Coverage (Long-Read)",
                              show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--billing-project", type=str, help="Project to bill under.",
                        default="tgg-rare-disease")
    parser.add_argument("--requester-pays-project", type=str,
                        help="Requester pays project to bill under.",
                        default="cmg-analysis")
    parser.add_argument("--output", type=str,
                        help="The directory to store results.",
                        default=f"{GCS_TMP_BUCKET}/long_read/sr_coverage")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze results instead of running batch jobs.")
    parser.add_argument("--da-genes", action="store_true",
                        help="Restrict coverage to disease-associated gene transcripts only.")
    args = parser.parse_args()

    if args.analyze:
        analyze_results(args)
    else:
        da_ids = get_da_gene_ids() if args.da_genes else None
        if da_ids:
            print(f"DA gene IDs: {len(da_ids)}")

        backend = hb.ServiceBackend(billing_project=args.billing_project,
                                    remote_tmpdir=args.output,
                                    regions=REGION)

        batch_name = "lr_compute_coverage"
        batch = hb.Batch(backend=backend, name=batch_name,
                         requester_pays_project=args.requester_pays_project,
                         default_image=DOCKER_GCLOUD_R,
                         default_cpu=DEFAULT_CPU,
                         default_memory=DEFAULT_MEMORY)
        main(da_gene_ids=da_ids)
