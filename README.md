# Long-Read RNA-seq Transcriptome Characterization for Rare Disease

Analysis code for the manuscript describing long-read RNA-seq transcriptome characterization in a rare disease cohort.

## Setup

### 1. Install the package

```bash
pip install -e .

# For Hail Batch support (cloud execution):
pip install -e ".[batch]"

# For development (linting):
pip install -e ".[dev]"
```

### 2. Configure data directory

Set the `LRRNASEQ_DATA_DIR` environment variable to point to your local data directory:

```bash
export LRRNASEQ_DATA_DIR=/path/to/data
```
It is the directory where output files are.

### 3. Configure `config.py`

Edit `src/rare_disease_lr_rnaseq/config.py` to set all placeholder values (marked with `<your-...>`).

**Cloud paths:**
- `GCS_REF_GTF` — GCS path to GENCODE v47 annotation GTF
- `GCS_BAM_DIR` — GCS path to aligned BAM files
- `GCS_FLNC_DIRS` — GCS paths to FLNC BAM directories
- `GCS_TMP_BUCKET` — GCS temporary bucket
- `GCS_OUTPUT_BASE` — GCS output bucket
- `GCS_GENE_MODELS_GFF` — GCS path to GENCODE GFF3 annotation (version numbers stripped, used by FRASER2)
- `DEFAULT_BILLING_PROJECT` / `DEFAULT_REQUESTER_PAYS_PROJECT` — GCP project IDs

**User-input file paths (local):**
- `METADATA_FILEPATH` — sample metadata TSV (columns: `entity:Sample_ID`, `Gender`, `RIN`, `Total_readcount`, etc.)
- `LR_SAMPLE_IDS_FILEPATH` — text file with one long-read sample ID per line
- `GENCODE_GTF_FILEPATH` — local GENCODE GTF annotation (gzipped)
- `MENDELIAN_GENE_DISEASE_TABLE_FILEPATH` (optional) — Mendelian gene-disease association table with ClinGen classifications
- `FRASER_JACCARD_FILEPATH`(optional) — FRASER2 Jaccard results CSV (output of the splicing pipeline)
- `FRASER_PSI3_FILEPATH` (optional) — FRASER1 PSI3 results CSV (output of the splicing pipeline)

**Docker images** are pre-configured; only update if rebuilding containers.

## Pipeline Execution Order

### Upstream computation (produces data consumed by figures)

1. **SQANTI3 filtering** — `src/.../sqanti3/sqanti3_filter.py`
2. **Preprocessing** — annotate, generate sample/fusion tables, read counts, read length distributions
   - `src/rare_disease_lr_rnaseq/preprocessing/annotate_gene_and_expr.py`
   - `src/rare_disease_lr_rnaseq/preprocessing/generate_sample_table.py`
   - `src/rare_disease_lr_rnaseq/preprocessing/generate_fusion_tx_table.py`
   - `src/rare_disease_lr_rnaseq/preprocessing/generate_reads_summary.py`
   - `src/rare_disease_lr_rnaseq/preprocessing/generate_rl_dist.py`
3. **Coverage** — short-read and long-read coverage computation
   - `src/rare_disease_lr_rnaseq/coverage/get_sr_coverage.py`
   - `src/rare_disease_lr_rnaseq/coverage/get_lr_coverage.py`
4. **Counting** — junction support and transcript read counts
   - `src/rare_disease_lr_rnaseq/counting/count_alt5_junction_support.py`
   - `src/rare_disease_lr_rnaseq/counting/count_reads.py`
5. **Splicing** — FRASER2 and alt 5' shift analyses
   - `src/rare_disease_lr_rnaseq/splicing/fraser2_pipeline.py`
   - `src/rare_disease_lr_rnaseq/splicing/alt5_shift_analysis.py`
   - `src/rare_disease_lr_rnaseq/splicing/alt5_shift_analysis_sr.py`

### Figure generation

```bash
python -m rare_disease_lr_rnaseq.figure_helpers.generate_manuscript_figures \
    --gtex-junctions /path/to/GTEX_junctions.bed.gz

python -m rare_disease_lr_rnaseq.figure_helpers.plot_qc_metrics \
    --reads-summary /path/to/reads_summary.tsv
```

## Data Requirements

Files configured via `config.py` (see step 3 above):
- Sample metadata TSV
- Long-read sample IDs list
- GENCODE v47 annotation GTF
- Mendelian gene-disease association table
- FRASER2 Jaccard and PSI3 results CSVs

Files expected under `$LRRNASEQ_DATA_DIR`:
- Per-sample SQANTI3 classification and junction files
- Per-sample LRAA expression quantification files
- Short-read STAR SJ.out.tab files (for alt 5' SR analysis)

Files passed as CLI arguments:
- GTEx junctions BED file (tabix-indexed, for sashimi plots)

## Repository Structure

```
src/rare_disease_lr_rnaseq/
    config.py                  # Centralized GCS paths, Docker images, project IDs
    utils.py                   # Shared utilities (GTF parsing, sample IDs, etc.)
    figure_helpers/            # Figure generation and helpers
        generate_manuscript_figures.py
        plot_qc_metrics.py
        get_num_of_isoforms_per_gene.py
        plot_all_tx.py
        generate_reads_cum_dist.py
        get_cdf_by_tx_length.py
    preprocessing/             # Data annotation, sample tables, read length dist
    splicing/                  # FRASER2, alt 5' shift analysis
    coverage/                  # Short-read and long-read coverage
    counting/                  # Junction support and transcript read counts
    sqanti3/                   # SQANTI3 filtering on Hail Batch

docker/                        # Dockerfiles for batch execution
    sqanti3/
    fraser2/
    coverage/
    cdf_plot/
    outrider/
    read_length/
```

## Cloud Execution

Batch scripts use [Hail Batch](https://hail.is/docs/batch/index.html) to parallelize computation on GCP. Install the `batch` extra and configure `gcloud auth` before running.

## License

MIT LICENSE.
