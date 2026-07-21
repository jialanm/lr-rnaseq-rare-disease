"""
Count junction read support across ALL transcripts on Hail Batch.

For each sample:
  1. Load SQANTI3 junction file (all structural categories: FSM, ISM, NIC, NNC)
  2. Deduplicate junctions by genomic coordinates
  3. Count BAM reads spanning each junction using CIGAR parsing
  4. Join with LRAA expression to get uniq_reads per isoform

Alt 5' classification is done downstream (not in this script).
No GTEx/CHESS filter. No structural category filter.
"""

import logging
import os
import subprocess
from datetime import date
from pathlib import Path

import hail as hl
import hailtop.fs as hfs
import pandas as pd

from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids
from rare_disease_lr_rnaseq.config import DOCKER_PARTIAL_IR, GCS_BAM_DIR, GCS_OUTPUT_BASE
from tgg.batch import batch_utils

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

hl.init(idempotent=True, log="/dev/null")

CPU_NUM = 2
MIN_MAPQ = 10

DEFAULT_GCS_BAM_DIR = GCS_BAM_DIR
DEFAULT_GCS_OUTPUT_DIR = f"{GCS_OUTPUT_BASE}/long_read/outrider/all_junction_counts"

JUNCTION_COUNT_SCRIPT = '''
"""Count junction read support for all junctions across all transcripts."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import pysam

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# CIGAR operation codes
CMATCH = 0
CINS = 1
CDEL = 2
CREF_SKIP = 3
CSOFT_CLIP = 4
CHARD_CLIP = 5
CPAD = 6
CEQUAL = 7
CDIFF = 8


MIN_UNIQ_READS = 2


def get_passing_isoforms(classification_path, lraa_expr_path, sample_id):
    """Get isoform IDs passing QC filters.

    Filters:
      - exons > 1 (no mono-exons)
      - all_canonical == "canonical"
      - RTS_stage == "False" (no RT-switching artifacts)
      - uniq_reads >= 2 (from LRAA expression)
    No structural category filter (keep FSM, ISM, NIC, NNC).
    """
    # Load classification
    cls_df = pd.read_csv(
        classification_path, sep="\\t",
        usecols=["isoform", "exons", "all_canonical", "RTS_stage"],
        dtype={"isoform": str, "exons": int, "all_canonical": str, "RTS_stage": str},
    )
    initial = len(cls_df)

    cls_df = cls_df[cls_df["exons"] > 1]
    cls_df = cls_df[cls_df["all_canonical"].astype(str) == "canonical"]
    cls_df = cls_df[cls_df["RTS_stage"].astype(str).str.upper() == "FALSE"]
    after_qc = len(cls_df)
    logger.info(f"  Classification QC: {initial:,} -> {after_qc:,} isoforms")

    # Load LRAA expression and filter by uniq_reads
    expr_df = pd.read_csv(lraa_expr_path, sep="\\t")
    expr_sample = expr_df[expr_df["sample_id"] == sample_id].copy()
    expr_sample = expr_sample[expr_sample["uniq_reads"] >= MIN_UNIQ_READS]
    logger.info(f"  LRAA expression: {len(expr_sample):,} isoforms with uniq_reads >= {MIN_UNIQ_READS}")

    # Intersect
    passing = set(cls_df["isoform"]) & set(expr_sample["original_tx_id"])
    logger.info(f"  Passing both filters: {len(passing):,} isoforms")

    return passing


def load_unique_junctions(junctions_path, passing_isoforms):
    """Load SQANTI3 junctions, filter to passing isoforms, deduplicate.

    Returns DataFrame with columns:
      chrom, strand, genomic_start_coord, genomic_end_coord,
      start_site_category, end_site_category, junction_category,
      isoform, junction_key
    Deduplicated by genomic coordinates (junction_key = chrom:start-end).
    """
    df = pd.read_csv(junctions_path, sep="\\t")
    logger.info(f"Loaded {len(df)} junction rows from {junctions_path}")

    # Filter to passing isoforms
    df = df[df["isoform"].isin(passing_isoforms)].copy()
    logger.info(f"  After isoform QC filter: {len(df)} junction rows")

    keep_cols = [
        "chrom", "strand", "genomic_start_coord", "genomic_end_coord",
        "start_site_category", "end_site_category", "junction_category",
        "isoform",
    ]
    df = df[keep_cols].copy()
    df["junction_key"] = (
        df["chrom"] + ":" +
        df["genomic_start_coord"].astype(str) + "-" +
        df["genomic_end_coord"].astype(str)
    )

    # Deduplicate by junction_key (same genomic coordinates across transcripts)
    dedup = df.drop_duplicates(subset="junction_key").copy()
    logger.info(f"  Unique junctions: {len(dedup)}")

    return dedup


def count_junction_reads(bam_file, chrom, junction_start, junction_end,
                         window=0, min_mapq=10):
    """Count reads spanning a splice junction using CIGAR operations.

    Returns (total_read_count, unique_read_count).
    """
    total_count = 0
    unique_count = 0

    try:
        reads = bam_file.fetch(chrom, junction_start - 100, junction_end + 100)
    except ValueError:
        logger.debug(f"Chromosome {chrom} not found in BAM")
        return 0, 0

    for read in reads:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue

        ref_pos = read.reference_start
        cigar_tuples = read.cigartuples
        if cigar_tuples is None:
            continue

        for op, length in cigar_tuples:
            if op in (CMATCH, CEQUAL, CDIFF):
                ref_pos += length
            elif op == CDEL:
                ref_pos += length
            elif op == CREF_SKIP:
                intron_start = ref_pos
                intron_end = ref_pos + length

                donor_1based = intron_start + 1
                acceptor_1based = intron_end

                start_match = abs(donor_1based - junction_start) <= window
                end_match = abs(acceptor_1based - junction_end) <= window

                if start_match and end_match:
                    total_count += 1
                    if read.mapping_quality >= min_mapq:
                        unique_count += 1
                    break

                ref_pos += length
            elif op in (CINS, CSOFT_CLIP, CHARD_CLIP):
                pass

    return total_count, unique_count


def main():
    parser = argparse.ArgumentParser(
        description="Count junction read support for all junctions"
    )
    parser.add_argument("--junctions", required=True,
                        help="SQANTI3 junctions file (all transcripts)")
    parser.add_argument("--classification", required=True,
                        help="SQANTI3 classification file (for QC filters)")
    parser.add_argument("--bam", required=True, help="BAM file path")
    parser.add_argument("--lraa-expr", required=True,
                        help="LRAA expression mapping TSV (original_tx_id -> uniq_reads)")
    parser.add_argument("--sample-id", required=True, help="Sample ID")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--min-mapq", type=int, default=10, help="Minimum MAPQ")
    parser.add_argument("--window", type=int, default=0,
                        help="Junction coordinate tolerance")
    args = parser.parse_args()

    # 1. QC filter: get passing isoforms
    logger.info("Applying QC filters...")
    passing_isoforms = get_passing_isoforms(
        args.classification, args.lraa_expr, args.sample_id
    )

    # 2. Load junctions from passing isoforms only
    logger.info(f"Loading SQANTI3 junctions: {args.junctions}")
    junctions_df = load_unique_junctions(args.junctions, passing_isoforms)

    if junctions_df.empty:
        logger.info("No junctions after QC filtering. Writing empty output.")
        pd.DataFrame(columns=[
            "junction_key", "chrom", "strand", "genomic_start_coord",
            "genomic_end_coord", "start_site_category", "end_site_category",
            "junction_category", "junction_read_counts",
            "junction_unique_read_counts",
        ]).to_csv(args.output, sep="\\t", index=False)
        return

    # 3. Count BAM reads for each junction
    logger.info(f"Opening BAM: {args.bam}")
    junction_read_counts = []
    junction_unique_read_counts = []

    with pysam.AlignmentFile(str(args.bam), "rb") as bam_file:
        for i, (_, row) in enumerate(junctions_df.iterrows()):
            total, unique = count_junction_reads(
                bam_file,
                row["chrom"],
                int(row["genomic_start_coord"]),
                int(row["genomic_end_coord"]),
                args.window,
                args.min_mapq,
            )
            junction_read_counts.append(total)
            junction_unique_read_counts.append(unique)

            if (i + 1) % 500 == 0:
                logger.info(f"  Counted {i + 1}/{len(junctions_df)} junctions")

    junctions_df = junctions_df.copy()
    junctions_df["junction_read_counts"] = junction_read_counts
    junctions_df["junction_unique_read_counts"] = junction_unique_read_counts

    logger.info(f"  Counted all {len(junctions_df)} junctions")

    # 4. Add uniq_reads and junction_ratio from LRAA
    logger.info(f"Loading LRAA expression: {args.lraa_expr}")
    expr_df = pd.read_csv(args.lraa_expr, sep="\\t")
    expr_sample = expr_df[expr_df["sample_id"] == args.sample_id].copy()
    if not expr_sample.empty:
        isoform_reads = dict(zip(expr_sample["original_tx_id"], expr_sample["uniq_reads"]))
        junctions_df["uniq_reads"] = junctions_df["isoform"].map(isoform_reads).fillna(0).astype(int)
        junctions_df["junction_ratio"] = junctions_df.apply(
            lambda r: r["junction_unique_read_counts"] / r["uniq_reads"]
            if r["uniq_reads"] > 0 else 0.0,
            axis=1,
        )
    else:
        logger.warning(f"No LRAA expression data for sample {args.sample_id}")
        junctions_df["uniq_reads"] = 0
        junctions_df["junction_ratio"] = 0.0

    # 5. Save output
    out_cols = [
        "junction_key", "chrom", "strand", "genomic_start_coord",
        "genomic_end_coord", "start_site_category", "end_site_category",
        "junction_category", "isoform", "junction_read_counts",
        "junction_unique_read_counts", "uniq_reads", "junction_ratio",
    ]
    junctions_df[out_cols].to_csv(args.output, sep="\\t", index=False)
    logger.info(f"Saved {len(junctions_df)} junctions to {args.output}")
    logger.info(
        f"  Junctions with unique reads > 0: "
        f"{(junctions_df['junction_unique_read_counts'] > 0).sum()}"
    )


if __name__ == "__main__":
    main()
'''


def _get_gcloud_env() -> dict[str, str]:
    """
    Get environment dictionary with CLOUDSDK_PYTHON set to a compatible Python.

    :return: Copy of the current environment with CLOUDSDK_PYTHON set if the conda hail environment Python exists.
    """
    env = os.environ.copy()
    conda_python = os.path.expanduser("~/anaconda3/envs/hail/bin/python")
    if os.path.exists(conda_python):
        env["CLOUDSDK_PYTHON"] = conda_python
    return env


def gcs_upload(local_path: str, gcs_path: str) -> None:
    """
    Upload a local file to GCS using gcloud storage cp.

    :param local_path: Local filesystem path to the file to upload.
    :param gcs_path: Destination GCS path (e.g. "gs://bucket/path/file.tsv").
    :raises RuntimeError: If the gcloud storage cp command fails.
    """
    env = _get_gcloud_env()

    result = subprocess.run(
        ["gcloud", "storage", "cp", local_path, gcs_path],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to upload {local_path} to {gcs_path}: {result.stderr}"
        )


def main() -> None:
    """
    Parse arguments, upload input files, and submit Hail Batch jobs for junction read counting.
    """
    p = batch_utils.init_arg_parser(
        default_cpu=CPU_NUM,
        default_temp_bucket="jialan-tmp-7day",
        gsa_key_file=os.path.expanduser(
            "~/.config/gcloud/misc-270914-cb9992ec9b25.json"
        ),
    )
    p.add_argument(
        "--sqanti3-junctions-dir",
        type=Path,
        default=Path(DATA_DIR) / "sqanti3_junctions",
        help="Local directory with per-sample SQANTI3 junction files",
    )
    p.add_argument(
        "--sqanti3-classification-dir",
        type=Path,
        default=Path(DATA_DIR) / "sqanti3",
        help="Local directory with per-sample SQANTI3 classification files",
    )
    p.add_argument(
        "--lraa-expr-path",
        type=Path,
        default=Path(DATA_DIR) / "outrider" / "lraa_tx_id_mapping.tsv",
        help="LRAA expression mapping file (original_tx_id -> uniq_reads)",
    )
    p.add_argument(
        "--gcs-bam-dir",
        type=str,
        default=DEFAULT_GCS_BAM_DIR,
        help="GCS directory containing BAM files",
    )
    p.add_argument(
        "--gcs-output-dir",
        type=str,
        default=DEFAULT_GCS_OUTPUT_DIR,
        help="GCS directory for output files",
    )
    p.add_argument(
        "--bam-suffix",
        type=str,
        default=".aligned.sorted.bam",
        help="BAM file suffix",
    )
    p.add_argument(
        "--min-mapq",
        type=int,
        default=MIN_MAPQ,
        help=f"Minimum MAPQ for unique reads (default: {MIN_MAPQ})",
    )
    p.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="File with sample IDs (one per line). Default: all samples.",
    )
    p.add_argument(
        "--date-suffix",
        type=str,
        default=None,
        help="Date suffix for output directory (default: {month}_{year})",
    )
    p.add_argument(
        "--use-filtered-junctions",
        action="store_true",
        help="Use *_filtered_junctions.txt instead of *_junctions.txt",
    )
    args = p.parse_args()

    sqanti3_dir = args.sqanti3_junctions_dir
    sqanti3_cls_dir = args.sqanti3_classification_dir
    lraa_expr_path = args.lraa_expr_path

    junc_suffix = (
        "_filtered_junctions.txt" if args.use_filtered_junctions
        else "_junctions.txt"
    )

    if args.sample_ids_file:
        with open(args.sample_ids_file) as f:
            sample_ids = [line.strip() for line in f if line.strip()]
    else:
        junc_files = list(sqanti3_dir.glob(f"*{junc_suffix}"))
        # Exclude *_filtered_junctions.txt when using unfiltered suffix
        if junc_suffix == "_junctions.txt":
            junc_files = [f for f in junc_files if "_filtered_junctions.txt" not in f.name]
        sample_ids = sorted(
            f.name.replace(junc_suffix, "") for f in junc_files
        )

    logger.info(f"Found {len(sample_ids)} samples to process")

    today = date.today()
    date_suffix = args.date_suffix or f"{today.month}_{today.year}"
    gcs_output_dir = f"{args.gcs_output_dir.rstrip('/')}/{date_suffix}"

    gcs_staging_dir = f"{gcs_output_dir}/staging"

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(
            f"Samples: {sample_ids[:5]}..."
            if len(sample_ids) > 5
            else f"Samples: {sample_ids}"
        )
        logger.info(f"SQANTI3 junctions dir: {sqanti3_dir}")
        logger.info(f"SQANTI3 classification dir: {sqanti3_cls_dir}")
        logger.info(f"LRAA expression: {lraa_expr_path}")
        logger.info(f"BAM dir: {args.gcs_bam_dir}")
        logger.info(f"Output dir: {gcs_output_dir}")
        logger.info(f"Junction suffix: {junc_suffix}")
        logger.info(f"Min MAPQ: {args.min_mapq}")
        return

    logger.info("Uploading input files to GCS...")
    gcs_lraa = f"{gcs_staging_dir}/lraa_tx_id_mapping.tsv"
    if not hfs.is_file(gcs_lraa) or args.force:
        gcs_upload(str(lraa_expr_path), gcs_lraa)

    for sample_id in sample_ids:
        local_junc = sqanti3_dir / f"{sample_id}{junc_suffix}"
        if not local_junc.exists():
            logger.warning(f"Junction file not found for {sample_id}: {local_junc}")
            continue
        gcs_junc = f"{gcs_staging_dir}/{sample_id}{junc_suffix}"
        if not args.force and hfs.is_file(gcs_junc):
            logger.info(f"  Skipping junction upload for {sample_id}: already exists")
        else:
            gcs_upload(str(local_junc), gcs_junc)

        local_cls = sqanti3_cls_dir / f"{sample_id}_classification.txt"
        if not local_cls.exists():
            logger.warning(f"Classification file not found for {sample_id}: {local_cls}")
            continue
        gcs_cls = f"{gcs_staging_dir}/{sample_id}_classification.txt"
        if not args.force and hfs.is_file(gcs_cls):
            logger.info(f"  Skipping classification upload for {sample_id}: already exists")
        else:
            gcs_upload(str(local_cls), gcs_cls)

    batch_label = f"Junction Read Counting: {len(sample_ids)} samples"
    logger.info(f"Submitting {len(sample_ids)} jobs")

    with batch_utils.run_batch(args, batch_label) as batch:
        jobs_created = 0

        for sample_id in sample_ids:
            local_junc = sqanti3_dir / f"{sample_id}{junc_suffix}"
            if not local_junc.exists():
                continue

            gcs_junc = f"{gcs_staging_dir}/{sample_id}{junc_suffix}"
            gcs_cls = f"{gcs_staging_dir}/{sample_id}_classification.txt"
            bam_path = f"{args.gcs_bam_dir}/{sample_id}{args.bam_suffix}"
            bai_path = f"{bam_path}.bai"
            output_path = f"{gcs_output_dir}/{sample_id}_all_junctions.tsv"

            if not args.force and hfs.is_file(output_path):
                logger.info(
                    f"Skipping {sample_id}: output exists at {output_path}"
                )
                continue

            jobs_created += 1
            logger.info(f"Creating job {jobs_created} for sample: {sample_id}")

            j = batch_utils.init_job(
                batch,
                f"junc_count_{sample_id}",
                image=DOCKER_PARTIAL_IR if not args.raw else None,
                cpu=CPU_NUM,
            )
            j._preemptible = True

            j.command("cd /io")
            j.command(
                "gcloud auth activate-service-account --key-file /gsa-key/key.json"
            )

            j.command(f"gcloud storage cp {gcs_junc} junctions.txt")
            j.command(f"gcloud storage cp {gcs_cls} classification.txt")
            j.command(f"gcloud storage cp {bam_path} sample.bam")
            j.command(f"gcloud storage cp {bai_path} sample.bam.bai")
            j.command(f"gcloud storage cp {gcs_lraa} lraa_expr.tsv")

            j.command(f"""cat > count_junctions.py << 'SCRIPT_EOF'
{JUNCTION_COUNT_SCRIPT}
SCRIPT_EOF
""")

            j.command(f"""python count_junctions.py \\
                --junctions junctions.txt \\
                --classification classification.txt \\
                --bam sample.bam \\
                --lraa-expr lraa_expr.tsv \\
                --sample-id {sample_id} \\
                --output output.tsv \\
                --min-mapq {args.min_mapq}
""")

            j.command(f"gcloud storage cp output.tsv {output_path}")

        logger.info(f"Created {jobs_created} jobs")

    logger.info(f"Results will be saved to: {gcs_output_dir}")

    local_output = Path(DATA_DIR) / "outrider" / "all_junction_counts"
    logger.info("To download results after batch completes:")
    logger.info(
        f"  mkdir -p {local_output} && "
        f"gcloud storage cp '{gcs_output_dir}/*_all_junctions.tsv' {local_output}/"
    )


if __name__ == "__main__":
    main()
