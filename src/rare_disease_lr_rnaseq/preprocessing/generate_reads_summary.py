"""Calculate FLNC and mapped read counts per sample from GCS-stored BAM files."""

import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd

from rare_disease_lr_rnaseq.config import GCS_BAM_DIR, GCS_FLNC_DIRS
from rare_disease_lr_rnaseq.utils import get_long_read_sample_ids

logger = logging.getLogger(__name__)


def count_reads(bam_path: str, *, mapped: bool) -> int:
    """Count reads in a GCS BAM file using gcloud storage cat piped to samtools.

    :param bam_path: GCS path to the BAM file.
    :param mapped: If True, exclude secondary/supplementary alignments (flag 2308).
    :return: Number of reads in the BAM file.
    """
    flags = "-c -F 2308 -" if mapped else "-c -"
    result = subprocess.run(
        f"set -o pipefail; gcloud storage cat '{bam_path}' | samtools view {flags}",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Command failed with stderr: %s", result.stderr[:500])
        raise subprocess.CalledProcessError(result.returncode, "count_reads")
    return int(result.stdout.strip())


def get_flnc_path(sample_id: str) -> str | None:
    """Find the FLNC BAM path for a sample, checking each FLNC directory.

    :param sample_id: Sample identifier.
    :return: GCS path to the FLNC BAM, or None if not found.
    """
    for flnc_dir in GCS_FLNC_DIRS:
        path = f"{flnc_dir}/{sample_id}.merged.unaligned.bam"
        result = subprocess.run(
            ["gcloud", "storage", "ls", path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return path
    return None


def get_mapped_path(sample_id: str) -> str:
    """Get the mapped BAM path for a sample.

    :param sample_id: Sample identifier.
    :return: GCS path to the mapped BAM.
    """
    return f"{GCS_BAM_DIR}/{sample_id}.aligned.sorted.bam"


def main(output: Path) -> None:
    """Calculate FLNC and mapped read counts for all samples.

    :param output: Path to write the output TSV file.
    """
    sample_ids = get_long_read_sample_ids()
    logger.info("Found %d samples", len(sample_ids))

    results: list[dict[str, str | int]] = []

    for sample_id in sample_ids:
        logger.info("Processing %s...", sample_id)

        flnc_count = 0
        flnc_path = get_flnc_path(sample_id)
        if flnc_path:
            try:
                flnc_count = count_reads(flnc_path, mapped=False)
                logger.info("  FLNC reads: %d", flnc_count)
            except subprocess.CalledProcessError:
                logger.warning("  Failed to count FLNC reads for %s", sample_id)
        else:
            logger.warning("  FLNC BAM not found for %s", sample_id)

        mapped_count = 0
        mapped_path = get_mapped_path(sample_id)
        try:
            mapped_count = count_reads(mapped_path, mapped=True)
            logger.info("  Mapped reads: %d", mapped_count)
        except subprocess.CalledProcessError:
            logger.warning("  Failed to count mapped reads for %s", sample_id)

        results.append({
            "sample_id": sample_id,
            "flnc_reads": flnc_count,
            "mapped_reads": mapped_count,
        })

    df = pd.DataFrame(results)
    df.to_csv(output, sep="\t", index=False)
    logger.info("Saved read counts for %d samples to %s", len(df), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate FLNC and mapped read counts per sample from GCS BAM files."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("reads_summary.tsv"),
        help="Output TSV file path for read counts summary.",
    )
    args = parser.parse_args()
    main(args.output)
