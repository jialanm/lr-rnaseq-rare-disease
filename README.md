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

### 3. Configure cloud paths

Edit `src/rare_disease_lr_rnaseq/config.py` to set:
- `GCS_REF_GTF` — GCS path to GENCODE v47 annotation GTF
- `GCS_BAM_DIR` — GCS path to aligned BAM files
- `GCS_FLNC_DIRS` — GCS paths to FLNC BAM directories
- `GCS_TMP_BUCKET` — GCS temporary bucket
- `GCS_OUTPUT_BASE` — GCS output bucket
- `DEFAULT_BILLING_PROJECT` / `DEFAULT_REQUESTER_PAYS_PROJECT` — GCP project IDs
- Docker image SHAs (pre-configured, only update if rebuilding)

## Pipeline Execution Order

### Upstream computation (produces data consumed by figures)

1. **SQANTI3 filtering** — `src/.../sqanti3/sqanti3_filter.py`
2. **Preprocessing** — annotate, generate sample/fusion tables, read length distributions
   - `src/.../preprocessing/annotate_gene_and_expr.py`
   - `src/.../preprocessing/generate_sample_table.py`
   - `src/.../preprocessing/generate_fusion_tx_table.py`
   - `src/.../preprocessing/generate_rl_dist.py`
3. **Coverage** — short-read and long-read coverage computation
   - `src/.../coverage/get_sr_coverage.py`
   - `src/.../coverage/get_lr_coverage.py`
4. **Counting** — junction support and transcript read counts
   - `src/.../counting/count_alt5_junction_support.py`
   - `src/.../counting/count_reads.py`
5. **Splicing** — FRASER2 and alt 5' shift analyses
   - `src/.../splicing/fraser2_pipeline.py`
   - `src/.../splicing/alt5_shift_analysis.py`
   - `src/.../splicing/alt5_shift_analysis_sr.py`

### Figure generation

```bash
python -m rare_disease_lr_rnaseq.figure_helpers.generate_manuscript_figures
python -m rare_disease_lr_rnaseq.figure_helpers.plot_qc_metrics
```

## Data Requirements

The following data files are expected under `$LRRNASEQ_DATA_DIR`:

- GENCODE v47 annotation GTF (`gencode.v47.annotation.gtf.gz`)
- Per-sample SQANTI3 classification and junction files
- Per-sample LRAA expression quantification files
- Mendelian gene-disease association table
- ClinGen gene validity classifications
- FRASER2 output tables
- Short-read STAR SJ.out.tab files (for alt 5' SR analysis)

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

See LICENSE file.
