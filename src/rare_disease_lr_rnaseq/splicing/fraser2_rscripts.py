"""R script generators for FRASER2 split-read counting and outlier detection."""

import logging

logger = logging.getLogger(__name__)

IMPLEMENTATION = "PCA"
COVARIATES = """c("sex", "stranded", "read_length", "batch", "library")"""


def count_split_reads_single_sample_r(num_of_cpu: int, annotation_dat_path: str) -> str:
    """Generate R script string to count split reads for a single sample.

    :param num_of_cpu: Number of CPUs to use for parallel processing.
    :param annotation_dat_path: Path to the CSV annotation file with sample metadata.
    :return: R script string for FRASER split read counting.
    """
    return f"""
    library(data.table)
    library(FRASER)
    library(BiocParallel)

    if(.Platform$OS.type == "unix") {{
        register(MulticoreParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }} else {{
        register(SnowParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }}

    # create sample dataset for FRASER with columns of sampleID and bamFile
    annotation_dat <- fread("{annotation_dat_path}")
    print(annotation_dat)

    # fds <- FraserDataSet(colData=annotation_dat, workingDir=".")
    # 
    # # count split reads
    # split_counts <- getSplitReadCountsForAllSamples(fds, BPPARAM=bpparam())

    settings <- FraserDataSet(colData=annotation_dat, workingDir=".")
    # count the split and non-split reads
    fds <- countRNAData(settings, BPPARAM=bpparam())
    """


def count_reads_all_samples_r(num_of_cpu: int, annotation_dat_path: str) -> str:
    """Generate R script string to count reads across all samples.

    :param num_of_cpu: Number of CPUs to use for parallel processing.
    :param annotation_dat_path: Path to the CSV annotation file with sample metadata.
    :return: R script string for FRASER read counting and PSI calculation.
    """
    return f"""
    library(data.table)
    library(FRASER)
    library(BiocParallel)

    # create sample dataset for FRASER with columns of sampleID and bamFile
    annotation_dat <- fread("{annotation_dat_path}")
    print(annotation_dat)

    settings <- FraserDataSet(colData=annotation_dat, workingDir=".")

    if(.Platform$OS.type == "unix") {{
        register(MulticoreParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }} else {{
        register(SnowParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }}

    # count the split and non-split reads
    fds <- countRNAData(settings, BPPARAM = bpparam())

    # calculate Jaccard intron index and other metrics
    fds <- calculatePSIValues(fds, BPPARAM=bpparam())

    saveFraserDataSet(fds, dir=".")
    """


def run_fraser_r(
    psitype: str,
    num_of_cpu: int,
    result_table_filename: str,
    heatmap_before_ae: str,
    heatmap_after_ae: str,
    enc_dim_auc: str,
    enc_dim_loss: str,
    delta_psi_threshold: float,
    delta_psi_threshold_init_filter: float,
    padj_threshold: float,
    min_reads: int,
    gene_models_gff_path: str,
    fraser2: str,
) -> str:
    """Generate R script string to run the FRASER/FRASER2 analysis pipeline.

    :param psitype: PSI type to analyse (e.g., 'jaccard', 'psi5', 'psi3', 'theta').
    :param num_of_cpu: Number of CPUs to use for parallel processing.
    :param result_table_filename: Output filename for the results CSV table.
    :param heatmap_before_ae: Output filename for the pre-autoencoder heatmap PNG.
    :param heatmap_after_ae: Output filename for the post-autoencoder heatmap PNG.
    :param enc_dim_auc: Output filename for the encoding dimension AUC plot PNG.
    :param enc_dim_loss: Output filename for the encoding dimension loss plot PNG.
    :param delta_psi_threshold: Delta PSI threshold for the result table filtering.
    :param delta_psi_threshold_init_filter: Delta PSI threshold for initial expression/variability filtering.
    :param padj_threshold: Adjusted p-value threshold for result filtering.
    :param min_reads: Minimum read count for expression filtering.
    :param gene_models_gff_path: Path to the GENCODE GFF3 gene model file.
    :param fraser2: 'True' to run FRASER2, 'False' to run FRASER1.
    :return: R script string for the full FRASER analysis pipeline.
    """
    return f"""
    library(data.table)
    library(FRASER)
    library(ggplot2)
    library(BiocParallel)
    library(GenomicFeatures)
    library(org.Hs.eg.db)

    if(.Platform$OS.type == "unix") {{
        register(MulticoreParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }} else {{
        register(SnowParam(workers=min({num_of_cpu}, multicoreWorkers())))
    }}

    fds = loadFraserDataSet(".")
    sample_ids <- fds$sampleID
    print(fds)

    if ("{fraser2}" == "True") {{
        # change splice metrics
        fitMetrics(fds) <- "{psitype}"  # not available in FRASER1
    }} else {{
        currentType(fds) <- "{psitype}"
    }}

    setwd(".")
    curDir <- getwd()
    print(curDir)
    
    # plot color heatmap for samples before autoencoder correction for sample covariance
    print({COVARIATES})
    before_ae <- plotCountCorHeatmap(fds, type="{psitype}", logit=TRUE, plotType="sampleCorrelation", annotation_col={COVARIATES})
    ggsave(filename = "{heatmap_before_ae}", plot = before_ae, device = "png", type="cairo")

    # filter junctions with low expressions
    print("Filterirng based on:")
    print(paste({min_reads}, "reads"))
    print(paste({delta_psi_threshold_init_filter}, "delta psi"))
    fds <- filterExpressionAndVariability(fds,  minExpressionInOneSample={min_reads}, minDeltaPsi={delta_psi_threshold_init_filter}, filter=TRUE, BPPARAM=bpparam())
    saveFraserDataSet(fds, dir=".")
    
    # get the optimal dimension of the latent space
    fds <- optimHyperParams(fds, type="{psitype}", implementation="{IMPLEMENTATION}", BPPARAM=bpparam())
    best_q = bestQ(fds, type="{psitype}")
    print("Best Q is: ")
    print(best_q)

    # # plot the encoding dimension search auc and loss
    # # enc_auc = plotEncDimSearch(fds, type="{psitype}", plotType="auc") 
    # # ggsave(filename = "{enc_dim_auc}", plot = enc_auc, device="png")
    # # enc_loss = plotEncDimSearch(fds, type="{psitype}", plotType="loss") 
    # # ggsave(filename = "{enc_dim_loss}", plot = enc_loss, device="png")


    # run FRASER pipeline
    print("Run FRASER pipeline... ")
    print(fds)

    if ("{fraser2}" == "True") {{
        # Limit FDR correction to a list of known disease genes
        # gene_vector <- c("PYROXD1", "NEB", "RYR1", "TTN", "COL6A3", "POMGNT1", "DMD", "COL6A1", "CAPN3", "LARGE1", "SRPK3", "MTM1", "LAMA2", "COL6A2")
        # gene_list <- setNames(lapply(sample_ids, function(x) gene_vector), sample_ids)
        # print(gene_list)

        # fds <- FRASER(fds, q=best_q, type="{psitype}", implementation="{IMPLEMENTATION}", BPPARAM=bpparam(), subsets=list("exampleSubset"=gene_list))
        fds <- FRASER(fds, q=best_q, type="{psitype}", implementation="{IMPLEMENTATION}", BPPARAM=bpparam())
        fds <- calculateZscore(fds, type="{psitype}")

        # fds <- fit(fds, implementation="{IMPLEMENTATION}", q=best_q, type="{psitype}", BPPARAM=bpparam())
        # fds <- calculatePvalues(fds, type="{psitype}")
        # fds <- calculatePadjValues(fds, type="{psitype}", subsets=list("exampleSubset"=gene_list))
        # fds <- calculateZscore(fds, type="{psitype}")
    }} else {{
        fds = fit(fds, implementation="PCA", q=best_q, type="{psitype}", BPPARAM=bpparam())
        fds = calculatePvalues(fds, type="{psitype}")
        fds = calculatePadjValues(fds, type="{psitype}")
        fds = calculateZscore(fds, type = "{psitype}")
    }}
    print(fds)

    # plot heatmap after confounder correction
    after_ae <- plotCountCorHeatmap(fds, type="{psitype}", logit=TRUE, normalized=TRUE, plotType="sampleCorrelation", annotation_col={COVARIATES})
    ggsave(filename = "{heatmap_after_ae}", plot = after_ae, device = "png", type="cairo")

    # annotate with GENCODE gene IDs
    txdb_obj <- makeTxDbFromGFF("./{gene_models_gff_path}")
    fds <- annotateRangesWithTxDb(fds, txdb=txdb_obj, feature="ENSEMBL", keytype="ENSEMBL")
    print(fds)

    if ("{fraser2}" == "True") {{
        res_filtered <- as.data.table(results(fds, padjCutoff={padj_threshold}, deltaPsiCutoff={delta_psi_threshold}))
        res <- as.data.table(results(fds, padjCutoff=1, deltaPsiCutoff=0))
        print(res)
    }} else {{
        res_filtered <- as.data.table(results(fds, psiType="{psitype}", padjCutoff={padj_threshold}, deltaPsiCutoff={delta_psi_threshold}))
        res <- as.data.table(results(fds, psiType="{psitype}", padjCutoff=1, deltaPsiCutoff=0))
        print(res)
    }}

    write.table(res_filtered, file="filtered_p_value_{padj_threshold}_deltapsi_{delta_psi_threshold}_{result_table_filename}", quote=FALSE, row.names=FALSE, sep = ",")
    write.table(res, file="{result_table_filename}", quote=FALSE, row.names=FALSE, sep = ",")

    # annotate with gene names
    fds <- annotateRangesWithTxDb(fds, txdb=txdb_obj, keytype="ENSEMBL")

    # result visualization
    for(sample_id in sample_ids) {{
        cur_plot_name <- paste("volcano_", sample_id, ".png", sep="")
        p <- plotVolcano(fds, sampleID=sample_id, label="aberrant", type="{psitype}", deltaPsiCutoff = {delta_psi_threshold}, padjCutoff = {padj_threshold})
        ggsave(filename = cur_plot_name, plot = p, device = "png", type="cairo")
    }}
    """
