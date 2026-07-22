"""Centralized configuration for the long-read RNA-seq analysis pipeline."""

import os
from pathlib import Path

DATA_DIR = os.environ.get("LRRNASEQ_DATA_DIR")
if DATA_DIR is None:
    raise EnvironmentError(
        "Set LRRNASEQ_DATA_DIR environment variable to point to the local data directory. "
        "Example: export LRRNASEQ_DATA_DIR=/path/to/data"
    )
DATA_DIR = str(Path(DATA_DIR).resolve())

GCS_REF_GTF = "gs://<your-bucket>/ref/gencode.v47.annotation.gtf"
GCS_BAM_DIR = "gs://<your-bucket>/"
GCS_FLNC_DIRS = [
    "gs://<your-bucket>/tag2167/merge",
    "gs://<your-bucket>/tag2195/merge",
]
GCS_TMP_BUCKET = "gs://<your-tmp-bucket>"
GCS_OUTPUT_BASE = "gs://<your-output-bucket>"

DOCKER_GCLOUD_R = (
    "gcr.io/cmg-analysis/gcloud-r"
    "@sha256:462b4ff1f4a533902831e45ac7aa818c3694bbc0d62933636055ad3b9f878709"
)
DOCKER_FRASER2 = (
    "gcr.io/cmg-analysis/fraser2"
    "@sha256:38e7e777a08886b5d4789b4c06f5433af60953542774134165281b7c39d35eeb"
)
DOCKER_FRASER1 = (
    "weisburd/gagneurlab"
    "@sha256:e2c0195ff95cb01c9a3619f281a5328d75af1994e5144fc1072e39b607d97edd"
)
DOCKER_SQANTI3 = (
    "gcr.io/cmg-analysis/sqanti3"
    "@sha256:fec424384c8687a1207c05f3bcb53b691dba4624a60aa5fa96727ec10750c26d"
)
DOCKER_PARTIAL_IR = "gcr.io/cmg-analysis/partial-ir:latest"
DOCKER_GCLOUD_SAMTOOLS = (
    "gcr.io/cmg-analysis/gcloud-samtools"
    "@sha256:37fd014fd6a4fbfeba9befad6b5661e5887781d33d171be3fd613c99922ff67c"
)

DEFAULT_BILLING_PROJECT = "<your-gcp-project>"
DEFAULT_REQUESTER_PAYS_PROJECT = "<your-requester-pays-project>"
HAIL_BATCH_REGIONS = ["us-central1"]

# User-input file paths (local)
METADATA_FILEPATH = "<your-metadata-file>"
LR_SAMPLE_IDS_FILEPATH = "<your-lr-sample-ids-file>"
GENCODE_GTF_FILEPATH = "<your-gencode-gtf-file>"
MENDELIAN_GENE_DISEASE_TABLE_FILEPATH = "<your-mendelian-gene-disease-table>"
FRASER_JACCARD_FILEPATH = "<your-fraser-jaccard-results-csv>"
FRASER_PSI3_FILEPATH = "<your-fraser-psi3-results-csv>"

# GCS reference paths
GCS_GENE_MODELS_GFF = "<your-gene-models-gff>"
