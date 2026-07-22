"""Run the FRASER2 aberrant splicing detection pipeline on Hail Batch."""

import os
from typing import Optional

import hailtop.batch as hb
import argparse
import numpy as np
from datetime import date
import hailtop.fs as hfs
import pandas as pd

from rare_disease_lr_rnaseq.splicing.fraser2_rscripts import count_split_reads_single_sample_r, \
    count_reads_all_samples_r, run_fraser_r
from rare_disease_lr_rnaseq.utils import get_long_read_sample_ids, DATA_DIR
from rare_disease_lr_rnaseq.config import DOCKER_FRASER2, DOCKER_FRASER1, GCS_BAM_DIR, GCS_TMP_BUCKET, GCS_GCS_GENE_MODELS_GFF, METADATA_FILEPATH

import logging

logger = logging.getLogger(__name__)

# Default GCS path for long-read BAM files
DEFAULT_LONGREAD_BAM_DIR = GCS_BAM_DIR
DEFAULT_LONGREAD_BAM_SUFFIX = ".aligned.sorted.bam"

REGION = ["us-central1"]
DEFAULT_CPU = 2 ** 4
DEFAULT_MEMORY = "highmem"
PER_BAM_SIZE = 50
DEFAULT_STORAGE = f"{PER_BAM_SIZE}G"
SPLIT_READS_DIR = f"{GCS_TMP_BUCKET}/fraser_split_reads"


DELTA_PSI_THRESHOLD_INIT_FILTERING = 0.05
DELTA_PSI_THRESHOLD_RESULT_TABLE = 0.1
PADJ_THRESHOLD = 0.3
MIN_READS = 2


def get_sample_rows_from_longread_bams(
    bam_dir: str = DEFAULT_LONGREAD_BAM_DIR,
    bam_suffix: str = DEFAULT_LONGREAD_BAM_SUFFIX,
    to_exclude: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Get sample metadata by constructing BAM paths from sample IDs in utils.py.

    :param bam_dir: GCS directory containing BAM files.
    :param bam_suffix: Suffix for BAM files (e.g., '.aligned.sorted.bam').
    :param to_exclude: Set of sample IDs to exclude.
    :return: DataFrame with columns 'sample_id', 'bam_path', 'sex', and 'library'.
    """
    if to_exclude is None:
        to_exclude = set()

    sample_ids = get_long_read_sample_ids()
    logger.info(f"Found {len(sample_ids)} sample IDs from utils.py")

    metadata_path = METADATA_FILEPATH
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    sex_map = dict(zip(metadata_df['entity:Sample_ID'], metadata_df['Gender']))

    rows = []
    for sample_id in sample_ids:
        if sample_id in to_exclude:
            continue

        bam_path = f"{bam_dir}/{sample_id}{bam_suffix}"
        metadata_sample_id = sample_id.replace("_R1", "")
        sex = sex_map.get(metadata_sample_id, "unknown")
        rows.append({
            "sample_id": sample_id,
            "bam_path": bam_path,
            "sex": sex,
            "library": "long-read",
        })

    df = pd.DataFrame(rows)
    logger.info(f"The number of excluded samples: {len(to_exclude)}")
    logger.info(f"Using {len(df)} samples after exclusions")

    return df


def create_symbolic_links(
    job: hb.batch.job.Job, path: str, link_path: str, requester_pays_project: str
) -> None:
    """Copy a file from GCS to a local path within a Hail Batch job.

    :param job: Hail Batch job to add the command to.
    :param path: Source GCS path.
    :param link_path: Destination local path within the job.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    """
    job.command(
        f"gcloud storage cp --billing-project={requester_pays_project} {path} {link_path}")


def load_bam_and_index_file(
    cur_job: hb.batch.job.Job, sample_id: str, bam_path: str, requester_pays_project: str
) -> str:
    """Copy a BAM file and its index into the job working directory.

    :param cur_job: Hail Batch job to add the copy commands to.
    :param sample_id: Sample identifier used for local file naming.
    :param bam_path: GCS path to the BAM file.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :return: Local path to the linked BAM file within the job.
    """
    bam_index_path = f"{bam_path}.bai"
    link_bam_path = f"{sample_id}.bam"
    link_bam_index_path = f"{sample_id}.bam.bai"
    create_symbolic_links(cur_job, bam_path, link_bam_path, requester_pays_project)
    create_symbolic_links(cur_job, bam_index_path, link_bam_index_path, requester_pays_project)
    return link_bam_path


def count_and_cache_split_reads(
    batch: hb.Batch, row: pd.Series, cur_saved_split_reads_path: str, requester_pays_project: str
) -> hb.batch.job.Job:
    """Count split reads for a single sample and cache results to GCS.

    :param batch: Hail Batch object to create the job in.
    :param row: Row from the sample annotation DataFrame with 'sampleID' and 'bamFile'.
    :param cur_saved_split_reads_path: GCS path to save the cached split read counts tarball.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :return: The Hail Batch job that counts split reads.
    """
    sample_id = row["sampleID"]
    bam_path = row["bamFile"]
    cur_count_job = batch.new_job(f"count_{sample_id}")
    bam_size = hfs.ls(bam_path)[0].size
    cur_count_job.storage(bam_size + 4e10)
    cur_count_job.memory("standard")
    cur_count_job.command("cd /io")

    link_bam_path = load_bam_and_index_file(cur_count_job,
                                            sample_id,
                                            bam_path,
                                            requester_pays_project)

    row["bamFile"] = link_bam_path
    cur_anno_dat = pd.DataFrame(row).T
    csv_string = cur_anno_dat.to_csv(index=False, header=True)
    output_filepath = f"{sample_id}.csv"
    cur_count_job.command(f"echo '{csv_string}' > {output_filepath}")

    cur_count_job.command(f"""xvfb-run Rscript -e '
    {count_split_reads_single_sample_r(DEFAULT_CPU, output_filepath)}
    '
    """)

    cached_filename = os.path.basename(cur_saved_split_reads_path)
    cur_count_job.command("ls -lh .")
    cur_count_job.command(f"tar czf {cached_filename} cache")
    cur_count_job.command(f"cp {cached_filename} {cur_count_job.ofile}")
    batch.write_output(cur_count_job.ofile, cur_saved_split_reads_path)

    return cur_count_job


def get_split_reads_path(cur_id: str) -> str:
    """Build the GCS path for a sample's cached split read counts tarball.

    :param cur_id: Sample identifier.
    :return: GCS path to the cached split read counts tarball.
    """
    return f"{SPLIT_READS_DIR}/count_split_reads_long_read_{cur_id}.tar.gz"


def copy_split_read_counts_files(
    job: hb.batch.job.Job, sample_ids: list[str], requester_pays_project: str
) -> None:
    """Copy cached split read count tarballs from GCS into the job.

    :param job: Hail Batch job to add the copy commands to.
    :param sample_ids: List of sample identifiers whose cached files to copy.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    """
    for cur_id in sample_ids:
        path = get_split_reads_path(cur_id)  # cloud
        link_path = f"count_split_reads_long_read_{cur_id}.tar.gz"  # soft link path
        create_symbolic_links(job, path, link_path, requester_pays_project)


def get_split_reads(
    batch: hb.Batch, use_rdg_dat: pd.DataFrame, requester_pays_project: str
) -> list[hb.batch.job.Job]:
    """Create per-sample split read counting jobs and return them.

    :param batch: Hail Batch object to create jobs in.
    :param use_rdg_dat: Sample annotation DataFrame with 'sampleID' and 'bamFile' columns.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :return: List of Hail Batch jobs that count split reads.
    """
    count_jobs = []
    for _, row in use_rdg_dat.iterrows():
        cur_id = row["sampleID"]
        cur_saved_split_reads_path = get_split_reads_path(cur_id)
        cur_count_job = count_and_cache_split_reads(batch,
                                                    row,
                                                    cur_saved_split_reads_path,
                                                    requester_pays_project)
        count_jobs.append(cur_count_job)

    return count_jobs


def get_all_reads(
    batch: hb.Batch,
    cur_job: hb.batch.job.Job,
    use_rdg_dat: pd.DataFrame,
    saved_fds_path: str,
    requester_pays_project: str,
) -> Optional[hb.batch.job.Job]:
    """Count all reads (split and non-split) across samples and cache results.

    :param batch: Hail Batch object to write output to.
    :param cur_job: Hail Batch job to add commands to.
    :param use_rdg_dat: Sample annotation DataFrame with 'sampleID' and 'bamFile' columns.
    :param saved_fds_path: GCS path to save the FraserDataSet savedObjects tarball.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :return: The job if it was configured, or None if cached results already exist.
    """
    if hfs.is_file(saved_fds_path):
        return None

    cur_job.storage(f"{15 * use_rdg_dat.shape[0]}G")
    sample_ids = list(use_rdg_dat["sampleID"])
    bam_paths = list(use_rdg_dat["bamFile"])
    link_bam_paths = load_split_reads_and_bam_files(cur_job, sample_ids,
                                                    bam_paths, requester_pays_project)

    use_rdg_dat["bamFile"] = link_bam_paths
    csv_string = use_rdg_dat.to_csv(index=False, header=True)
    annotation_dat_path = "all_annotation_dat.csv"
    cur_job.command(f"echo '{csv_string}' > {annotation_dat_path}")

    cur_job.command(f"""xvfb-run Rscript -e '
            {count_reads_all_samples_r(DEFAULT_CPU, annotation_dat_path)}
            '
            """)

    cur_job.command("ls -lh .")
    cur_job.command(f"tar czf savedObjects.tar.gz savedObjects")
    cur_job.command(f"cp savedObjects.tar.gz {cur_job.ofile}")
    batch.write_output(cur_job.ofile, saved_fds_path)

    return cur_job


def load_split_reads_and_bam_files(
    cur_job: hb.batch.job.Job,
    sample_ids: list[str],
    bam_paths: list[str],
    requester_pays_project: str,
) -> list[str]:
    """Load cached split read counts and BAM files into the job.

    :param cur_job: Hail Batch job to add commands to.
    :param sample_ids: List of sample identifiers.
    :param bam_paths: List of GCS BAM file paths corresponding to sample_ids.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :return: List of local linked BAM file paths within the job.
    """
    cur_job.command("cd /io")

    copy_split_read_counts_files(cur_job, sample_ids, requester_pays_project)
    cur_job.command("ls -lh")
    cur_job.command("for i in count_split_reads*.tar.gz; do tar xzf $i; done")
    cur_job.command("ls -lh cache")
    link_bam_paths = []

    for i in range(len(sample_ids)):
        cur_id = sample_ids[i]
        cur_bam_path = bam_paths[i]
        cur_link_bam_path = load_bam_and_index_file(cur_job, cur_id,
                                                    cur_bam_path, requester_pays_project)
        link_bam_paths.append(cur_link_bam_path)
    return link_bam_paths


def run_fraser(
    batch: hb.Batch,
    cur_job: hb.batch.job.Job,
    cur_type: str,
    num_samples: int,
    saved_fds_path: str,
    fraser_dir: str,
    today_formatted: str,
    requester_pays_project: str,
    run_fraser1: bool = False,
) -> None:
    """Run the FRASER/FRASER2 pipeline in a Hail Batch job.

    :param batch: Hail Batch object to write output to.
    :param cur_job: Hail Batch job to add commands to.
    :param cur_type: PSI type to analyse (e.g., 'jaccard', 'psi5', 'psi3', 'theta').
    :param num_samples: Number of samples in the analysis.
    :param saved_fds_path: GCS path to the saved FraserDataSet tarball.
    :param fraser_dir: GCS directory for FRASER results.
    :param today_formatted: Date-based subdirectory name for result versioning.
    :param requester_pays_project: GCP billing project for requester-pays buckets.
    :param run_fraser1: If True, run FRASER1 instead of FRASER2.
    """
    create_symbolic_links(cur_job, saved_fds_path, "savedObjects.tar.gz", requester_pays_project)
    cur_job.command(f"tar xzf savedObjects.tar.gz")
    cur_job.command("ls -lh")

    gene_models_gff_path = os.path.basename(GCS_GENE_MODELS_GFF)
    create_symbolic_links(cur_job, GCS_GENE_MODELS_GFF, gene_models_gff_path, requester_pays_project)

    prefix = f"longread_{cur_type}_{num_samples}_samples_" \
             f"pdj_{PADJ_THRESHOLD}_deltapsi_{DELTA_PSI_THRESHOLD_RESULT_TABLE}"
    result_table = f"{prefix}_results.csv"
    heatmap_before_ae = f"{prefix}_before_ae_heatmap.png"
    heatmap_after_ae = f"{prefix}_after_ae_heatmap.png"
    enc_dim_auc = f"{prefix}_enc_dim_auc.png"
    enc_dim_loss = f"{prefix}_enc_dim_loss.png"

    zip_dat = f"{prefix}_zip_dat.tar.gz"

    fraser2 = "False" if run_fraser1 else "True"
    cur_job.command(f"""xvfb-run Rscript -e '
    {run_fraser_r(cur_type, DEFAULT_CPU, result_table, heatmap_before_ae, heatmap_after_ae, enc_dim_auc, enc_dim_loss,
                  DELTA_PSI_THRESHOLD_RESULT_TABLE, DELTA_PSI_THRESHOLD_INIT_FILTERING, PADJ_THRESHOLD, MIN_READS, gene_models_gff_path, fraser2)}
    '
    """)
    cur_job.command("ls -lh .")

    cur_job.command(f"tar czf {zip_dat} {result_table} filtered_p_value"
                    f"_{PADJ_THRESHOLD}_deltapsi_{DELTA_PSI_THRESHOLD_RESULT_TABLE}_{result_table}"
                    f" {heatmap_before_ae} {heatmap_after_ae} volcano_* savedObjects")
    cur_job.command(f"cp {zip_dat} {cur_job.ofile}")
    batch.write_output(cur_job.ofile, f"{fraser_dir}/{today_formatted}/{zip_dat}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FRASER2 splicing analysis on long-read RNA-seq BAM files."
    )
    parser.add_argument("--billing-project", type=str,
                        help="Project to bill under.",
                        default="tgg-rare-disease")
    parser.add_argument("--requester-pays-project", type=str,
                        help="Requester pays project to bill under.",
                        default="cmg-analysis")
    parser.add_argument("--file-dir", type=str,
                        help="The directory to store results table.",
                        default=GCS_TMP_BUCKET)
    parser.add_argument("-e", "--exclude",
                        nargs='+',
                        help="Sample IDs to exclude.")
    parser.add_argument("--fraser1", action="store_true",
                        help="Run FRASER1 instead of FRASER2.")
    parser.add_argument("--flag",
                        help="An extra flag added to the output filename.")
    args = parser.parse_args()

    backend = hb.ServiceBackend(billing_project=args.billing_project,
                                remote_tmpdir=args.file_dir,
                                regions=REGION)

    if args.fraser1:
        logger.info("The pipeline is running FRASER1.")
        prefix = "fraser1"
        current_docker_image = DOCKER_FRASER1
        psi_types = ["psi5", "psi3", "theta"]
    else:
        logger.info("The pipeline is running FRASER2.")
        prefix = "fraser2"
        current_docker_image = DOCKER_FRASER2
        psi_types = ["jaccard"]

    batch_name = f"longread_{prefix}"
    fraser_dir = f"{args.file_dir}/{prefix}"
    batch = hb.Batch(backend=backend, name=batch_name,
                     requester_pays_project=args.requester_pays_project,
                     default_image=current_docker_image,
                     default_cpu=DEFAULT_CPU,
                     default_memory=DEFAULT_MEMORY,
                     default_storage=DEFAULT_STORAGE)

    today = date.today()
    if not args.flag:
        today_formatted = f"{today.month}_{today.year}"
    else:
        today_formatted = f"{today.month}_{today.year}_{args.flag}"

    saved_fds_path = f"{fraser_dir}/{today_formatted}/long_read_savedObjects.tar.gz"

    to_exclude = set(args.exclude) if args.exclude else set()
    use_rdg_dat = get_sample_rows_from_longread_bams(to_exclude=to_exclude)

    use_rdg_dat = use_rdg_dat.rename(columns={"sample_id": "sampleID",
                                              "bam_path": "bamFile"})
    use_rdg_dat["pairedEnd"] = "FALSE"  # long reads are not paired-end
    logger.info(use_rdg_dat)

    logger.info("The number of samples used in this batch run is: ", use_rdg_dat.shape[0])

    input_samples_filename = f"longread_{prefix}_sample_ids_{date.today()}.txt"
    np.savetxt(input_samples_filename,
               use_rdg_dat["sampleID"],
               fmt="%s",
               delimiter='\n')
    os.system(f"gcloud storage cp {input_samples_filename} {fraser_dir}/{today_formatted}/")

    count_jobs = get_split_reads(batch, use_rdg_dat, args.requester_pays_project)

    all_reads_job = batch.new_job(f"get_all_reads_{use_rdg_dat.shape[0]}_longread_samples")
    all_reads_job._preemptible = False
    if len(count_jobs) > 0:
        all_reads_job.depends_on(*count_jobs)
    get_all_reads(batch, all_reads_job, use_rdg_dat, saved_fds_path, args.requester_pays_project)

    for cur_type in psi_types:
        fraser_job = batch.new_job(f"{prefix}_{cur_type}")
        fraser_job._preemptible = False
        fraser_job.depends_on(all_reads_job)
        run_fraser(batch, fraser_job, cur_type,
                   num_samples=use_rdg_dat.shape[0],
                   saved_fds_path=saved_fds_path,
                   fraser_dir=fraser_dir,
                   today_formatted=today_formatted,
                   requester_pays_project=args.requester_pays_project,
                   run_fraser1=args.fraser1)

    batch.run()
