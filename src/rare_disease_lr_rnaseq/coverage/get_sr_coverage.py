"""Compute binned transcript coverage profiles from short-read RNA-seq BAMs on Hail Batch."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from typing import Optional

import hailtop.batch as hb
import hailtop.fs as hfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rare_disease_lr_rnaseq.config import DOCKER_GCLOUD_R, GCS_REF_GTF, GCS_TMP_BUCKET
from rare_disease_lr_rnaseq.utils import DATA_DIR, create_symbolic_links
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import read_from_airtable, RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID

logger = logging.getLogger(__name__)

REGION = ["us-central1"]
DEFAULT_CPU = 2 ** 4
DEFAULT_MEMORY = "standard"
N_BINS = 100

GTF_FILEPATH = GCS_REF_GTF
MENDELIAN_GENE_DISEASE_TABLE_FILEPATH = f"{DATA_DIR}/mendelian_gene_disease_table_1_16_2026.tsv"
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


def compute_coverage(
    gtf: str,
    bam: str,
    output_file: str,
    min_tx_length: int = 200,
    n_bins: int = 100,
    cds_only: bool = False,
    da_gene_ids_file: Optional[str] = None,
) -> str:
    """
    Generate an R script string that computes binned transcript coverage from a BAM file.

    :param gtf: Path to the reference GTF annotation file.
    :param bam: Path to the short-read BAM file.
    :param output_file: Path where the coverage TSV will be written.
    :param min_tx_length: Minimum transcript length to include.
    :param n_bins: Number of bins for the coverage profile.
    :param cds_only: If True, compute coverage over CDS regions only.
    :param da_gene_ids_file: Path to a file listing disease-associated gene IDs for filtering.
    :return: R script as a string.
    """
    region_cmd = 'cdsBy(txdb, by="tx", use.names=TRUE)' if cds_only else 'exonsBy(txdb, by="tx", use.names=TRUE)'
    region_label = "CDS" if cds_only else "exon"

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

    cat("Loading GTF...\\n")
    txdb <- suppressWarnings(makeTxDbFromGFF('{gtf}', format = "gtf"))
    cat("Extracting {region_label} regions...\\n")
    all_transcripts <- {region_cmd}
    tx_lengths <- sum(width(all_transcripts))
    all_transcripts <- all_transcripts[tx_lengths >= {min_tx_length}]
{da_filter_block}
    cat("Getting chromosomes from BAM...\\n")
    bf <- BamFile("{bam}")
    bam_header <- scanBamHeader(bf)
    chrom_lengths <- bam_header$targets

    standard_chroms <- paste0("chr", c(1:22, "X", "Y"))
    chromosomes <- intersect(names(chrom_lengths), standard_chroms)

    # Initialize
    global_profile_sum <- numeric({n_bins})
    total_tx_counted <- 0

    for(chrom in chromosomes) {{
        cat("Processing", chrom, "\\n")
        tx_on_chrom <- all_transcripts[seqnames(unlist(range(all_transcripts))) == chrom]
        if(length(tx_on_chrom) == 0) next

        param <- ScanBamParam(
            which = GRanges(chrom, IRanges(1, chrom_lengths[chrom])),
            flag = scanBamFlag(isSecondaryAlignment=FALSE, isSupplementaryAlignment=FALSE, 
                               isPaired=TRUE, isProperPair=TRUE)
        )

        reads <- readGAlignmentPairs("{bam}", param=param, strandMode = 2)
        if(length(reads) == 0) next

        grl <- grglist(reads)
        read_blocks <- unlist(union(grl, grl))
        rm(reads, grl); gc()

        tx_names <- names(tx_on_chrom)
        n_tx <- length(tx_names)
        chunk_size <- 500
        n_chunks <- ceiling(n_tx / chunk_size)

        for(i in 1:n_chunks) {{
            start_idx <- ((i-1) * chunk_size) + 1
            end_idx <- min(i * chunk_size, n_tx)
            current_tx_chunk <- tx_on_chrom[tx_names[start_idx:end_idx]]

            tx_hits <- mapToTranscripts(read_blocks, current_tx_chunk, ignore.strand = FALSE)
            
            if(length(tx_hits) > 0) {{
                # Set seqlengths so coverage() Rle spans the full transcript length
                # Without this, the Rle is truncated at the last mapped position,
                # causing aggregate() to return NA for bins beyond that point
                hit_tx_names <- levels(seqnames(tx_hits))
                seqlengths(tx_hits) <- tx_lengths[hit_tx_names]
                chunk_cov <- coverage(tx_hits)
                
                for(tn in names(chunk_cov)) {{
                    cov_rle <- chunk_cov[[tn]]
                    full_len <- tx_lengths[tn]
                    
                    # --- HARDENING STEPS ---
                    # 1. Skip if transcript is shorter than bin count
                    if(full_len < {n_bins}) next
                    
                    # 2. Only include if max coverage > 0
                    max_val <- max(cov_rle)
                    if(is.na(max_val) || max_val <= 0) next
                    
                    # 3. Define bins and aggregate
                    bins <- seq(1, full_len + 1, length.out = {n_bins + 1})
                    bin_means <- aggregate(cov_rle, FUN=mean, start=head(bins, -1), end=tail(bins, -1) - 1)
                    
                    # 4. Normalize
                    norm_bins <- as.numeric(bin_means) / max_val
                    
                    # 5. Final check: Only add if there are no NAs in this transcript's profile
                    # 5. Final check: Only add if there are no NAs in this transcript's profile
                    if(!any(is.na(norm_bins))) {{
                        global_profile_sum <- global_profile_sum + norm_bins
                        total_tx_counted <- total_tx_counted + 1
                    }}
                }}
                rm(tx_hits, chunk_cov)
            }}
            if(i %% 10 == 0) gc()
        }}
        rm(read_blocks, tx_on_chrom); gc()
    }}

    if(total_tx_counted > 0) {{
        final_vector <- global_profile_sum / total_tx_counted
        output_data <- data.frame(bin = 1:{n_bins}, relative_coverage = final_vector)

        write.table(output_data, file = "{output_file}",
                    sep = "\\t", quote = FALSE, row.names = FALSE)

        cat("Successfully processed", total_tx_counted, "transcripts.\\n")
    }} else {{
        stop("Zero transcripts were successfully processed. Check chromosome naming or strandMode.")
    }}
    """


def compute_coverage_batch_job(
    batch: hb.Batch,
    args: argparse.Namespace,
    bam_filepath: str,
    gtf_filepath: str,
    sample_id: str,
    prefix: str,
    cds_only: bool = False,
    da_gene_ids: Optional[set[str]] = None,
) -> None:
    """
    Create and configure a Hail Batch job to compute short-read coverage for one sample.

    :param batch: The Hail Batch to add the job to.
    :param args: Parsed command-line arguments containing output directory.
    :param bam_filepath: GCS path to the BAM file.
    :param gtf_filepath: GCS path to the GTF annotation file.
    :param sample_id: Sample identifier.
    :param prefix: Output file prefix (e.g. "sr", "sr_cds", "sr_da").
    :param cds_only: If True, restrict to CDS regions.
    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    output_file = f"{prefix}_{sample_id}_coverage.csv"
    if hfs.is_file(f"{args.output}/{output_file}"):
        return
    cur_job = batch.new_job(f"{prefix}_compute_coverage_{sample_id}")
    cur_job._preemptible = False
    cur_job.cpu(DEFAULT_CPU)
    cur_job.storage("60G")
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

    r_script = compute_coverage(local_gtf_filepath, local_bam_filepath, output_file, cds_only=cds_only, da_gene_ids_file=da_gene_ids_file)
    cur_job.command(f"""cat > compute_coverage.R << 'RSCRIPT'
{r_script}
RSCRIPT
xvfb-run Rscript compute_coverage.R""")

    cur_job.command(f"gcloud storage cp {output_file} {args.output}/{output_file}")



def compute_coverage_sr(
    batch: hb.Batch,
    args: argparse.Namespace,
    da_gene_ids: Optional[set[str]] = None,
) -> None:
    """
    Submit batch jobs for short-read BAM files from Airtable metadata.

    :param batch: The Hail Batch to add jobs to.
    :param args: Parsed command-line arguments.
    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    short_read_dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    short_read_dat = short_read_dat[short_read_dat["imputed_tissue"] == "whole_blood"]
    short_read_dat = short_read_dat[~(short_read_dat["exclude"] == "yes")]
    short_read_dat = short_read_dat[~(short_read_dat["watchmaker"] == "yes")]
    short_read_dat = short_read_dat[short_read_dat["read_length"] == 151]
    short_read_dat = short_read_dat.sample(n=25, random_state=42)
    print(f"Total samples: {short_read_dat.shape[0]}")

    sample_ids = list(short_read_dat["sample_id"])
    bam_filepaths = list(short_read_dat["star_bam"])

    prefix = "sr_da" if args.da_genes else ("sr_cds" if args.cds_only else "sr")
    for cur_sample, cur_bam_filepath in zip(sample_ids, bam_filepaths):
        print(f"Submitting job for {cur_sample}: {cur_bam_filepath}")
        compute_coverage_batch_job(batch, args, cur_bam_filepath, GTF_FILEPATH, cur_sample, prefix, cds_only=args.cds_only, da_gene_ids=da_gene_ids)


def compute_coverage_watchmaker(
    batch: hb.Batch,
    args: argparse.Namespace,
    cds_only: bool = False,
    da_gene_ids: Optional[set[str]] = None,
) -> None:
    """
    Submit batch jobs for Watchmaker short-read BAM files.

    :param batch: The Hail Batch to add jobs to.
    :param args: Parsed command-line arguments.
    :param cds_only: If True, restrict to CDS regions.
    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["watchmaker"] == "yes"]
    dat = dat[~(dat["exclude"] == "yes")]
    dat = dat[["sample_id", "star_bam"]].dropna(subset=["star_bam"])
    prefix = "wm_da" if da_gene_ids else ("wm_cds" if cds_only else "wm")
    print(f"Watchmaker samples: {dat.shape[0]} (CDS only: {cds_only}, DA genes: {da_gene_ids is not None})")

    for _, row in dat.iterrows():
        cur_sample = row["sample_id"]
        cur_bam_filepath = row["star_bam"]
        print(f"Submitting job for {cur_sample}: {cur_bam_filepath}")
        compute_coverage_batch_job(batch, args, cur_bam_filepath, GTF_FILEPATH, cur_sample, prefix, cds_only=cds_only, da_gene_ids=da_gene_ids)


def main(da_gene_ids: Optional[set[str]] = None) -> None:
    """
    Submit short-read coverage batch jobs and run the batch.

    :param da_gene_ids: Set of disease-associated gene IDs for filtering.
    """
    compute_coverage_sr(batch, args, da_gene_ids=da_gene_ids)
    batch.run()


def analyze_results(args: argparse.Namespace) -> None:
    """
    Download and analyze coverage results from GCS.

    :param args: Parsed command-line arguments containing the output GCS directory.
    """
    cur_data_dir = f"{DATA_DIR}/sr_coverage"
    os.makedirs(cur_data_dir, exist_ok=True)
    os.system(f"""gcloud storage cp "{args.output}/sr*_coverage.csv" {cur_data_dir}""")

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
                              title="Transcript Coverage (Short-Read)",
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
    parser.add_argument("--watchmaker", action="store_true",
                        help="Compute coverage for Watchmaker samples instead of whole blood.")
    parser.add_argument("--cds-only", action="store_true",
                        help="Compute coverage over CDS regions only (exclude UTRs).")
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

        batch_name = "wm_compute_coverage" if args.watchmaker else "sr_compute_coverage"
        batch = hb.Batch(backend=backend, name=batch_name,
                         requester_pays_project=args.requester_pays_project,
                         default_image=DOCKER_GCLOUD_R,
                         default_cpu=DEFAULT_CPU,
                         default_memory=DEFAULT_MEMORY)
        if args.watchmaker:
            compute_coverage_watchmaker(batch, args, cds_only=args.cds_only, da_gene_ids=da_ids)
            batch.run()
        else:
            main(da_gene_ids=da_ids)
