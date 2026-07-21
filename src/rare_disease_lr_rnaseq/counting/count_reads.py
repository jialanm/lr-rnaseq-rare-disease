"""
Run BAM-based transcript read counting on Hail Batch.

Parallelizes counting across samples, with one job per sample.
Each job downloads the BAM and GTF files, runs intron chain matching,
and uploads per-sample counts to GCS. A final merge job combines all
per-sample counts into a single count matrix.
"""

import logging
import os
import subprocess
from datetime import date
from pathlib import Path

import hail as hl
import hailtop.fs as hfs

from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids
from rare_disease_lr_rnaseq.config import DOCKER_PARTIAL_IR, GCS_BAM_DIR, GCS_OUTPUT_BASE
from tgg.batch import batch_utils


def _get_gcloud_env() -> dict[str, str]:
    """
    Get environment dictionary with CLOUDSDK_PYTHON set to a compatible Python.

    :return: Copy of the current environment with CLOUDSDK_PYTHON set if the conda hail environment Python exists.
    """
    env = os.environ.copy()
    # Use Python from conda env for gcloud compatibility
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
        raise RuntimeError(f"Failed to upload {local_path} to {gcs_path}: {result.stderr}")



logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

hl.init(idempotent=True, log="/dev/null")

CPU_NUM = 2
MIN_MAPQ = 20


DEFAULT_GCS_BAM_DIR = GCS_BAM_DIR
DEFAULT_GCS_OUTPUT_DIR = f"{GCS_OUTPUT_BASE}/long_read/outrider/bam_counts"

COUNT_READS_SCRIPT = '''
"""Count BAM reads per transcript using exact intron chain matching."""

import argparse
import logging
from collections import defaultdict
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


def extract_introns_from_cigar(read):
    """Extract intron positions from CIGAR N operations.

    Returns coordinates matching GTF-based intron signatures which use
    (exon_end, next_exon_start) format:
    - intron_start = ref_pos (0-based value equals 1-based exon_end)
    - intron_end = ref_pos + length + 1 (1-based next_exon_start)
    """
    if read.cigartuples is None:
        return ()
    introns = []
    ref_pos = read.reference_start  # 0-based
    for op, length in read.cigartuples:
        if op in (CMATCH, CEQUAL, CDIFF, CDEL):
            ref_pos += length
        elif op == CREF_SKIP:
            # Match GTF intron signature format: (exon_end, next_exon_start)
            intron_start = ref_pos  # 0-based value = 1-based exon_end
            intron_end = ref_pos + length + 1  # 1-based next_exon_start
            introns.append((intron_start, intron_end))
            ref_pos = ref_pos + length
    return tuple(introns)


def read_gtf_from_path(gtf_path):
    """Read GTF file and extract relevant columns."""
    cur_gtf = pd.read_csv(
        gtf_path, sep="\\t", header=None, comment="#",
        dtype={0: "category", 2: "category", 6: "category"},
    )
    cur_gtf.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
    extracted = cur_gtf["attributes"].str.extract(
        r'gene_id "(?P<gene_id>[^"]+)".*?transcript_id "(?P<transcript_id>[^"]+)"'
    )
    cur_gtf = pd.concat([cur_gtf[["chrom", "feature", "start", "end", "strand"]], extracted], axis=1)
    return cur_gtf


def compute_transcript_features(gtf_df):
    """Compute intron signature from GTF exon data."""
    import numpy as np

    exon_gtf = gtf_df[gtf_df["feature"] == "exon"].copy()
    if exon_gtf.empty:
        return pd.DataFrame(columns=["transcript_id", "chrom", "strand", "intron_signature"])

    exon_gtf = exon_gtf.sort_values(["transcript_id", "start"]).reset_index(drop=True)

    # Filter to multi-exon only
    exon_counts = exon_gtf.groupby("transcript_id", sort=False).size()
    multi_exon_tx = exon_counts[exon_counts >= 2].index
    exon_gtf = exon_gtf[exon_gtf["transcript_id"].isin(multi_exon_tx)].reset_index(drop=True)

    if exon_gtf.empty:
        return pd.DataFrame(columns=["transcript_id", "chrom", "strand", "intron_signature"])

    # Compute introns
    tx_ids = exon_gtf["transcript_id"].values
    is_last = np.concatenate([tx_ids[:-1] != tx_ids[1:], [True]])
    exon_ends = exon_gtf["end"].values
    exon_starts = exon_gtf["start"].values
    next_starts = np.roll(exon_starts, -1)

    intron_mask = ~is_last
    intron_tx_ids = tx_ids[intron_mask]
    intron_starts_arr = exon_ends[intron_mask]
    intron_ends_arr = next_starts[intron_mask]

    if len(intron_tx_ids) > 0:
        intron_pairs = [f"{s},{e}" for s, e in zip(intron_starts_arr, intron_ends_arr)]
        intron_df = pd.DataFrame({"transcript_id": intron_tx_ids, "intron_pair": intron_pairs})
        intron_sig_strs = intron_df.groupby("transcript_id", sort=False)["intron_pair"].agg("|".join)

        def parse_intron_sig(s):
            return tuple(tuple(map(int, p.split(","))) for p in s.split("|"))

        intron_sigs = intron_sig_strs.apply(parse_intron_sig)
        intron_sigs.name = "intron_signature"
    else:
        intron_sigs = pd.Series(name="intron_signature", dtype=object)

    agg_df = (
        exon_gtf.groupby("transcript_id", sort=False)
        .agg(chrom=("chrom", "first"), strand=("strand", "first"))
        .reset_index()
    )
    features_df = agg_df.merge(intron_sigs, on="transcript_id", how="left")
    return features_df[["transcript_id", "chrom", "strand", "intron_signature"]]


def count_reads_for_sample(bam_path, gtf_path, cluster_ids, id_mapping_df, sample_id, min_mapq):
    """Count BAM reads for each cluster_id based on exact intron chain matching."""
    sample_mapping = id_mapping_df[id_mapping_df["sample_id"] == sample_id].copy()
    sample_mapping = sample_mapping[sample_mapping["cluster_id"].isin(cluster_ids)]

    if sample_mapping.empty:
        logger.warning(f"No cluster mappings found for sample {sample_id}")
        return {cid: 0 for cid in cluster_ids}

    gtf_df = read_gtf_from_path(gtf_path)
    tx_features = compute_transcript_features(gtf_df)

    if tx_features.empty:
        logger.warning(f"No transcript features computed from GTF for {sample_id}")
        return {cid: 0 for cid in cluster_ids}

    tx_features = tx_features.set_index("transcript_id")

    intron_to_cluster = {}
    cluster_to_intron_key = {}

    for _, row in sample_mapping.iterrows():
        original_tx_id = row["original_tx_id"]
        cluster_id = row["cluster_id"]
        if original_tx_id not in tx_features.index:
            continue
        tx_row = tx_features.loc[original_tx_id]
        intron_sig = tx_row["intron_signature"]
        chrom = tx_row["chrom"]
        strand = tx_row["strand"]
        key = (chrom, strand, intron_sig)
        intron_to_cluster[key] = cluster_id
        cluster_to_intron_key[cluster_id] = key

    logger.info(f"Built {len(intron_to_cluster)} intron signature -> cluster mappings")

    intron_counts = defaultdict(int)

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        total_reads = 0
        matched_reads = 0
        for read in bam.fetch():
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue
            total_reads += 1
            chrom = read.reference_name
            strand = "-" if read.is_reverse else "+"
            read_introns = extract_introns_from_cigar(read)
            key = (chrom, strand, read_introns)
            if key in intron_to_cluster:
                intron_counts[key] += 1
                matched_reads += 1

        if total_reads > 0:
            logger.info(f"Processed {total_reads} reads, {matched_reads} matched ({100*matched_reads/total_reads:.1f}%)")

    cluster_counts = {}
    for cluster_id in cluster_ids:
        if cluster_id in cluster_to_intron_key:
            key = cluster_to_intron_key[cluster_id]
            cluster_counts[cluster_id] = intron_counts.get(key, 0)
        else:
            cluster_counts[cluster_id] = 0

    return cluster_counts


def main():
    parser = argparse.ArgumentParser(description="Count BAM reads per transcript")
    parser.add_argument("--bam", required=True, help="Path to BAM file")
    parser.add_argument("--gtf", required=True, help="Path to GTF file")
    parser.add_argument("--counts-matrix", required=True, help="Path to count matrix TSV")
    parser.add_argument("--id-mapping", required=True, help="Path to ID mapping TSV")
    parser.add_argument("--sample-id", required=True, help="Sample identifier")
    parser.add_argument("--output", required=True, help="Output TSV path")
    parser.add_argument("--min-mapq", type=int, default=20, help="Minimum MAPQ")
    args = parser.parse_args()

    logger.info(f"Processing sample: {args.sample_id}")

    counts_df = pd.read_csv(args.counts_matrix, sep="\\t")
    if args.sample_id in counts_df.columns:
        sample_counts = counts_df[["transcript_id", args.sample_id]].copy()
        sample_counts = sample_counts[sample_counts[args.sample_id] > 0]
        cluster_ids = sample_counts["transcript_id"].tolist()
        logger.info(f"Found {len(cluster_ids)} clusters with counts for {args.sample_id}")
    else:
        logger.warning(f"Sample {args.sample_id} not found in count matrix")
        cluster_ids = []

    id_mapping_df = pd.read_csv(args.id_mapping, sep="\\t")

    cluster_counts = count_reads_for_sample(
        bam_path=args.bam,
        gtf_path=args.gtf,
        cluster_ids=cluster_ids,
        id_mapping_df=id_mapping_df,
        sample_id=args.sample_id,
        min_mapq=args.min_mapq,
    )

    output_df = pd.DataFrame([
        {"cluster_id": cid, "bam_read_count": count}
        for cid, count in cluster_counts.items()
    ])
    output_df.to_csv(args.output, sep="\\t", index=False)
    logger.info(f"Saved {len(output_df)} cluster counts to {args.output}")


if __name__ == "__main__":
    main()
'''

MERGE_COUNTS_SCRIPT = '''
"""Merge per-sample BAM counts into a single count matrix."""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge per-sample BAM counts")
    parser.add_argument("--input-dir", required=True, help="Directory with per-sample count files")
    parser.add_argument("--sample-ids", required=True, help="Comma-separated sample IDs")
    parser.add_argument("--output", required=True, help="Output merged count matrix")
    args = parser.parse_args()

    sample_ids = args.sample_ids.split(",")
    logger.info(f"Merging counts for {len(sample_ids)} samples")

    all_counts = {}
    for sample_id in sample_ids:
        count_file = Path(args.input_dir) / f"{sample_id}_bam_counts.tsv"
        if count_file.exists():
            df = pd.read_csv(count_file, sep="\\t")
            counts_dict = dict(zip(df["cluster_id"], df["bam_read_count"]))
            all_counts[sample_id] = counts_dict
            logger.info(f"  Loaded {len(counts_dict)} counts for {sample_id}")
        else:
            logger.warning(f"  Count file not found for {sample_id}")
            all_counts[sample_id] = {}

    # Get all cluster IDs
    all_cluster_ids = set()
    for counts_dict in all_counts.values():
        all_cluster_ids.update(counts_dict.keys())

    logger.info(f"Total unique cluster IDs: {len(all_cluster_ids)}")

    # Build matrix
    rows = []
    for cluster_id in sorted(all_cluster_ids):
        row = {"transcript_id": cluster_id}
        for sample_id in sample_ids:
            row[sample_id] = all_counts[sample_id].get(cluster_id, 0)
        rows.append(row)

    merged_df = pd.DataFrame(rows)
    merged_df.to_csv(args.output, sep="\\t", index=False)
    logger.info(f"Saved merged count matrix to {args.output}")
    logger.info(f"  Shape: {len(merged_df)} transcripts x {len(sample_ids)} samples")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    """
    Parse arguments, upload input files, and submit Hail Batch jobs for BAM read counting.
    """
    p = batch_utils.init_arg_parser(
        default_cpu=CPU_NUM,
        default_temp_bucket="jialan-tmp-7day",
        gsa_key_file=os.path.expanduser(
            "~/.config/gcloud/misc-270914-cb9992ec9b25.json"
        ),
    )
    p.add_argument(
        "--counts-matrix",
        type=str,
        required=True,
        help="GCS or local path to lraa_tx_counts_full.tsv",
    )
    p.add_argument(
        "--id-mapping",
        type=str,
        required=True,
        help="GCS or local path to lraa_tx_id_mapping.tsv",
    )
    p.add_argument(
        "--gcs-gtf-dir",
        type=str,
        default=None,
        help="GCS directory containing sample GTF files. If not provided, will upload from local DATA_DIR/tx_gtf/",
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
        help=f"Minimum MAPQ for read filtering (default: {MIN_MAPQ})",
    )
    p.add_argument(
        "--local-output-path",
        type=Path,
        default=None,
        help="Local path to download the final merged count matrix",
    )
    p.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="File with sample IDs (one per line). If not provided, uses all samples.",
    )
    p.add_argument(
        "--date-suffix",
        type=str,
        default=None,
        help="Date suffix for output directory (default: {month}_{year})",
    )
    args = p.parse_args()

    if args.sample_ids_file:
        with open(args.sample_ids_file) as f:
            sample_ids = [line.strip() for line in f if line.strip()]
    else:
        sample_ids = get_long_read_sample_ids()

    logger.info(f"Found {len(sample_ids)} samples to process")

    today = date.today()
    date_suffix = args.date_suffix or f"{today.month}_{today.year}"
    gcs_output_dir = f"{args.gcs_output_dir.rstrip('/')}/{date_suffix}"

    counts_matrix_gcs = args.counts_matrix
    id_mapping_gcs = args.id_mapping
    gcs_gtf_dir = args.gcs_gtf_dir

    if not args.counts_matrix.startswith("gs://"):
        counts_matrix_gcs = f"{gcs_output_dir}/lraa_tx_counts_full.tsv"

    if not args.id_mapping.startswith("gs://"):
        id_mapping_gcs = f"{gcs_output_dir}/lraa_tx_id_mapping.tsv"

    if gcs_gtf_dir is None:
        gcs_gtf_dir = f"{gcs_output_dir}/gtf"

    per_sample_dir = f"{gcs_output_dir}/per_sample"
    merged_output = f"{gcs_output_dir}/lraa_tx_bam_counts.tsv"

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info(
            f"Samples: {sample_ids[:5]}..."
            if len(sample_ids) > 5
            else f"Samples: {sample_ids}"
        )
        logger.info(f"Count matrix: {counts_matrix_gcs}")
        logger.info(f"ID mapping: {id_mapping_gcs}")
        logger.info(f"GTF dir: {gcs_gtf_dir}")
        logger.info(f"BAM dir: {args.gcs_bam_dir}")
        logger.info(f"Output dir: {gcs_output_dir}")
        logger.info(f"Min MAPQ: {args.min_mapq}")
        return

    if not args.counts_matrix.startswith("gs://"):
        logger.info(f"Uploading count matrix to {counts_matrix_gcs}")
        gcs_upload(args.counts_matrix, counts_matrix_gcs)

    if not args.id_mapping.startswith("gs://"):
        logger.info(f"Uploading ID mapping to {id_mapping_gcs}")
        gcs_upload(args.id_mapping, id_mapping_gcs)

    if args.gcs_gtf_dir is None:
        local_gtf_dir = Path(DATA_DIR) / "tx_gtf"
        logger.info(f"Uploading GTF files from {local_gtf_dir} to {gcs_gtf_dir}")

        gtf_files_to_upload = []
        for sample_id in sample_ids:
            local_gtf = local_gtf_dir / f"{sample_id}.LRAA.gtf"
            if local_gtf.exists():
                gcs_gtf = f"{gcs_gtf_dir}/{sample_id}.LRAA.gtf"
                if not hfs.is_file(gcs_gtf):
                    gtf_files_to_upload.append(str(local_gtf))

        if gtf_files_to_upload:
            logger.info(f"Uploading {len(gtf_files_to_upload)} GTF files in parallel")
            result = subprocess.run(
                ["gcloud", "storage", "cp"] + gtf_files_to_upload + [gcs_gtf_dir + "/"],
                capture_output=True,
                text=True,
                env=_get_gcloud_env(),
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to upload GTF files: {result.stderr}")

    batch_label = f"BAM Read Counting: {len(sample_ids)} samples"
    logger.info(f"Submitting {len(sample_ids)} per-sample jobs + 1 merge job")

    with batch_utils.run_batch(args, batch_label) as batch:
        per_sample_jobs = []
        jobs_created = 0

        for sample_id in sample_ids:
            bam_path = f"{args.gcs_bam_dir}/{sample_id}{args.bam_suffix}"
            bai_path = f"{bam_path}.bai"
            gtf_path = f"{gcs_gtf_dir}/{sample_id}.LRAA.gtf"
            output_path = f"{per_sample_dir}/{sample_id}_bam_counts.tsv"

            if not args.force and hfs.is_file(output_path):
                logger.info(f"Skipping {sample_id}: output exists at {output_path}")
                continue

            jobs_created += 1
            logger.info(f"Creating job {jobs_created} for sample: {sample_id}")

            j = batch_utils.init_job(
                batch,
                f"count_{sample_id}",
                image=DOCKER_PARTIAL_IR if not args.raw else None,
                cpu=CPU_NUM,
            )
            j._preemptible = True

            j.command("cd /io")
            j.command(
                "gcloud auth activate-service-account --key-file /gsa-key/key.json"
            )

            j.command(f"gcloud storage cp {bam_path} sample.bam")
            j.command(f"gcloud storage cp {bai_path} sample.bam.bai")
            j.command(f"gcloud storage cp {gtf_path} sample.gtf")
            j.command(f"gcloud storage cp {counts_matrix_gcs} counts_matrix.tsv")
            j.command(f"gcloud storage cp {id_mapping_gcs} id_mapping.tsv")

            j.command(f"""cat > count_reads.py << 'SCRIPT_EOF'
{COUNT_READS_SCRIPT}
SCRIPT_EOF
""")

            j.command(f"""python count_reads.py \\
                --bam sample.bam \\
                --gtf sample.gtf \\
                --counts-matrix counts_matrix.tsv \\
                --id-mapping id_mapping.tsv \\
                --sample-id {sample_id} \\
                --output output.tsv \\
                --min-mapq {args.min_mapq}
""")

            j.command(f"gcloud storage cp output.tsv {output_path}")

            per_sample_jobs.append(j)

        logger.info(f"Created {jobs_created} per-sample jobs")

        if jobs_created > 0:
            merge_job = batch_utils.init_job(
                batch,
                "merge_counts",
                image=DOCKER_PARTIAL_IR if not args.raw else None,
                cpu=1,
            )
            merge_job._preemptible = True

            for j in per_sample_jobs:
                merge_job.depends_on(j)

            merge_job.command("cd /io")
            merge_job.command(
                "gcloud auth activate-service-account --key-file /gsa-key/key.json"
            )

            merge_job.command("mkdir -p per_sample")
            merge_job.command(
                f"gcloud storage cp '{per_sample_dir}/*_bam_counts.tsv' per_sample/"
            )

            merge_job.command(f"""cat > merge_counts.py << 'SCRIPT_EOF'
{MERGE_COUNTS_SCRIPT}
SCRIPT_EOF
""")

            sample_ids_str = ",".join(sample_ids)
            merge_job.command(f"""python merge_counts.py \\
                --input-dir per_sample \\
                --sample-ids "{sample_ids_str}" \\
                --output merged_counts.tsv
""")

            merge_job.command(f"gcloud storage cp merged_counts.tsv {merged_output}")

    logger.info(f"Results will be saved to: {gcs_output_dir}")
    logger.info(f"Merged count matrix: {merged_output}")

    if args.local_output_path and jobs_created > 0:
        logger.info("Note: To download results after batch completes, run:")
        logger.info(f"  gcloud storage cp {merged_output} {args.local_output_path}")


if __name__ == "__main__":
    main()
