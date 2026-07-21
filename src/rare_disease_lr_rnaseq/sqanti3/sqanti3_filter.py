"""Run SQANTI3 quality filtering on long-read transcript models via Hail Batch."""

from __future__ import annotations

import argparse
import logging

import hailtop.batch as hb
import hailtop.fs as hfs
import pandas as pd

from rare_disease_lr_rnaseq.config import DOCKER_SQANTI3, GCS_TMP_BUCKET
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import \
    switch_to_gmail_account

logger = logging.getLogger(__name__)

REGION = ["us-central1"]
DEFAULT_CPU = 1


def create_symbolic_links(job: hb.batch.job.Job, path: str, link_path: str) -> None:
    """
    Add a gcloud storage cp command to a Hail Batch job to stage a file locally.

    :param job: The Hail Batch job to add the command to.
    :param path: GCS source path to copy from.
    :param link_path: Local destination filename.
    """
    job.command(f"gcloud storage cp --billing-project={args.requester_pays_project} {path} {link_path}")



def run_sqanti3_filter_job(job: hb.batch.job.Job, prefix: str, classification_file: str, filter: str) -> None:
    """
    Add a SQANTI3 filter command to a Hail Batch job.

    :param job: The Hail Batch job to add the command to.
    :param prefix: Output file prefix for the filtered results.
    :param classification_file: Path to the SQANTI3 classification file.
    :param filter: Filter type to apply (e.g. "rules" or "ml").
    """
    job.command(
        f"""conda run -n sqanti3 sqanti3_filter.py {filter} {classification_file} --filter_mono_exonic -o {prefix}""")


def main() -> None:
    """
    Iterate over SQANTI3 QC tar files and submit filter jobs for each sample.
    """
    data_dir = args.file_dir
    for tar_file in hfs.ls(data_dir):
        tar_file = tar_file.path
        if tar_file.endswith("_sqanti3_qc.tar.gz"):
            sample_id = tar_file.split("/")[-1].replace("_sqanti3_qc.tar.gz", "")
            print(f"Processing {sample_id} from {tar_file}")

            prefix = f"{sample_id}_sqanti3_filtered"
            if args.ml_filter:
                prefix = f"{sample_id}_sqanti3_ml_filtered"
            tar_file_filtered = f"{prefix}.tar.gz"
            if hfs.exists(f"{args.file_dir}/{tar_file_filtered}"):
                print(f"Filtered file for {sample_id} already exists. Skipping...")
                continue

            cur_job = batch.new_job(name=f"extract_sqanti3_filter_{sample_id}")
            switch_to_gmail_account(cur_job)

            cur_job.command("cd /io")
            link_path = f"{sample_id}_sqanti3_qc.tar.gz"
            create_symbolic_links(cur_job, tar_file, link_path)
            cur_job.command(f"ls -lh .")

            cur_job.command(f"tar xzf {link_path}")
            cur_job.command(f"ls -lh .")

            classification_file = f"{sample_id}_classification.txt"
            if args.ml_filter:
                run_sqanti3_filter_job(cur_job, sample_id, classification_file, "ml")
            else:
                run_sqanti3_filter_job(cur_job, sample_id, classification_file, "rules")
            cur_job.command(f"ls -lh .")

            cur_job.command(f"tar czf {tar_file_filtered} {prefix}*")
            cur_job.command(f"cp {tar_file_filtered} {cur_job.ofile}")

            batch.write_output(cur_job.ofile,
                               f"{args.file_dir}/{tar_file_filtered}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQANTI3 filter on long read RNA-seq data.")
    parser.add_argument("--billing-project",
                        type=str,
                        help="Project to bill under.",
                        default="tgg-rare-disease")
    parser.add_argument("--requester-pays-project",
                        type=str,
                        help="Requester pays project to bill under.",
                        default="cmg-analysis")
    parser.add_argument("--temp-dir",
                        type=str,
                        help="The temporary directory for Hail.",
                        default=GCS_TMP_BUCKET)
    parser.add_argument("--file-dir",
                        type=str,
                        help="Directory to fetch the SQANTI3 classification files.",
                        default=f"{GCS_TMP_BUCKET}/long_read/sqanti3")
    parser.add_argument("--ml-filter",
                        action="store_true",
                        help="Whether to use default rules filter or ML fileter."
                        )
    args = parser.parse_args()

    batch_name = "generate_sqanti3_filter"
    backend = hb.ServiceBackend(billing_project=args.billing_project,
                                remote_tmpdir=args.temp_dir,
                                regions=REGION)
    batch = hb.Batch(backend=backend, name=batch_name,
                     requester_pays_project=args.requester_pays_project,
                     default_image=DOCKER_SQANTI3,
                     default_cpu=DEFAULT_CPU,
                     default_memory="highmem",
                     default_storage="5G")

    main()
    batch.run()
