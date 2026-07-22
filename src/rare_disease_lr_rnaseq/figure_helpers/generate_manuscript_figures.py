#!/usr/bin/env python3
"""Generate manuscript figures at 1x1 inch, 300 DPI, viridis palette, no titles."""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pysam
import seaborn as sns

from rare_disease_lr_rnaseq.config import FRASER_JACCARD_FILEPATH, FRASER_PSI3_FILEPATH
from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids, read_sqanti3_annotated, get_unique_tx
from tgg_rnaseq_pipelines.rnaseq_sample_metadata.metadata_utils import (
    read_from_airtable, RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID,
)
from rare_disease_lr_rnaseq.figure_helpers.get_num_of_isoforms_per_gene import (
    get_isoform_counts_per_sample, categorize_isoform_count,
)
from rare_disease_lr_rnaseq.figure_helpers.plot_all_tx import (
    get_all_tx, get_disease_associated_genes, get_expressed_da_genes, EVIDENCE_COLORS,
)
from rare_disease_lr_rnaseq.figure_helpers.generate_reads_cum_dist import (
    GENES_TO_REMOVE, intersect, get_gene_read_counts_dict, add_to_total_uniq_reads, convert_to_df,
)
from rare_disease_lr_rnaseq.figure_helpers.get_cdf_by_tx_length import (
    compute_cdf_at_lengths, load_sr_sample_cdfs, load_lr_sample_cdfs,
)
from scipy import stats

from rare_disease_lr_rnaseq.figure_helpers.plot_qc_metrics import load_read_length_data_both, load_reads_summary, load_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path(DATA_DIR) / "manuscript_figures"

VIR_A = "#440154"
VIR_B = "#21918c"
VIR_C = "#fde725"


def setup_style() -> None:
    """
    Configure matplotlib for manuscript-quality 1x1 inch figures at 300 DPI.
    """
    mpl.rcParams.update({
        "figure.figsize": (1, 1),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.labelsize": 4.5,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.5,
        "ytick.major.size": 1.5,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "legend.fontsize": 3.5,
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.handlelength": 0.5,
        "legend.handletextpad": 0.3,
        "legend.borderpad": 0.2,
        "legend.borderaxespad": 0.2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 4,
        "lines.linewidth": 0.8,
    })


def plot_read_length_distribution() -> None:
    """
    Plot read length distribution for pre- and post-hemoglobin depletion samples.
    """
    setup_style()
    pre_matrix, post_matrix, bins = load_read_length_data_both()
    x = np.array(bins)

    pre_mean, pre_std = np.mean(pre_matrix, axis=0), np.std(pre_matrix, axis=0)
    post_mean, post_std = np.mean(post_matrix, axis=0), np.std(post_matrix, axis=0)

    cumsum_pre = np.cumsum(pre_mean)
    median_rl_pre = x[np.searchsorted(cumsum_pre, cumsum_pre[-1] / 2)]
    cumsum_post = np.cumsum(post_mean)
    median_rl_post = x[np.searchsorted(cumsum_post, cumsum_post[-1] / 2)]

    fig, ax = plt.subplots()
    c_with = cm.viridis(0.3)
    c_without = cm.viridis(0.8)
    ax.fill_between(x, pre_mean - pre_std, pre_mean + pre_std, alpha=0.25, color=c_with, linewidth=0)
    l1, = ax.plot(x, pre_mean, color=c_with)
    ax.fill_between(x, post_mean - post_std, post_mean + post_std, alpha=0.25, color=c_without, linewidth=0)
    l2, = ax.plot(x, post_mean, color=c_without)

    ax.set_xlabel("Read length (bp)")
    ax.set_ylabel(r"Mean read count ($10^6$)")
    ax.set_xlim(0, 3000)
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="plain")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t / 1e6:.1f}" for t in yticks])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend([l1, l2],
              ["w/ hemoglobin",
               "w/o hemoglobin"],
              loc="upper right")

    out = OUTPUT_DIR / "read_length_distribution.png"
    fig.savefig(out)

    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_cumulative_reads_by_gene() -> None:
    """
    Plot cumulative fraction of reads by gene rank for top 10 genes.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    total_with, total_without = {}, {}

    for sample_id in samples:
        if not sample_id.endswith("_3_R1"):
            continue
        ann = read_sqanti3_annotated(sample_id, rules_filter=False)
        ann = ann[~(ann["gene_type"] == "novel")]

        ann["interval"] = ann.apply(lambda r: f"{r['chrom']}:{r['start']}-{r['end']}", axis=1)
        ann["is_hemo"] = ann["interval"].map(
            lambda x: any(intersect(x, h) for h in GENES_TO_REMOVE))
        ann_no_hemo = ann[~ann["is_hemo"]]

        total_with = add_to_total_uniq_reads(total_with, get_gene_read_counts_dict(ann))
        total_without = add_to_total_uniq_reads(total_without, get_gene_read_counts_dict(ann_no_hemo))

    df_with, _ = convert_to_df(total_with)
    df_without, _ = convert_to_df(total_without)

    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    c_with = cm.viridis(0.3)
    c_without = cm.viridis(0.8)
    ax.plot(np.arange(1, len(df_with) + 1), df_with["cum_dist"].values,
            color=c_with, marker="o", markersize=1.5, zorder=3)
    ax.plot(np.arange(1, len(df_without) + 1), df_without["cum_dist"].values,
            color=c_without, marker="^", markersize=1.5, zorder=3)

    ax.set_xlabel("Gene rank")
    ax.set_ylabel("Cumulative fraction of reads")
    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(["w/ hemoglobin", "w/o hemoglobin"], loc="right")

    fig.subplots_adjust(left=0.28, bottom=0.20, right=0.95, top=0.95)
    out = OUTPUT_DIR / "cumulative_reads_distribution_by_genes.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_cdf_by_transcript_length() -> None:
    """
    Plot CDF of reads by transcript length comparing short-read and long-read data.
    """
    setup_style()
    length_bins = np.linspace(0, 350_000, 1000)

    sr_cdfs = load_sr_sample_cdfs(f"{DATA_DIR}/sr_transcript_read_counts", length_bins)
    lr_cdfs = load_lr_sample_cdfs(f"{DATA_DIR}/lr_transcript_read_counts", length_bins)
    sr_mean = np.mean(sr_cdfs, axis=0)
    lr_mean = np.mean(lr_cdfs, axis=0)

    length_bins_kb = length_bins / 1000

    fig, ax = plt.subplots()
    l_sr, = ax.plot(length_bins_kb, sr_mean, color=VIR_A, linestyle="--")
    l_lr, = ax.plot(length_bins_kb, lr_mean, color=VIR_B)

    sr_99_length = length_bins[min(np.searchsorted(sr_mean, 0.99), len(length_bins) - 1)]
    sr_99_kb = sr_99_length / 1000
    lr_cdf_at_sr99 = lr_mean[min(np.searchsorted(length_bins, sr_99_length), len(lr_mean) - 1)]
    sr_cdf_at_sr99 = sr_mean[min(np.searchsorted(length_bins, sr_99_length), len(sr_mean) - 1)]
    ax.axvline(x=sr_99_kb, color="black", linestyle="--", linewidth=0.6)

    ax.set_xlabel("Transcript length (kb)")
    ax.set_ylabel("Cumulative fraction of reads")
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend([l_lr, l_sr], ["Long read", "Short read"], loc="upper right",
              bbox_to_anchor=(1.0, 0.88))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax, width="40%", height="40%", loc="lower left",
                        borderpad=2.5)
    ax_ins.plot(length_bins_kb, sr_mean, color=VIR_A, linestyle="--")
    ax_ins.plot(length_bins_kb, lr_mean, color=VIR_B)
    ax_ins.axvline(x=sr_99_kb, color="black", linestyle="--", linewidth=0.4)
    ax_ins.axhline(y=sr_cdf_at_sr99, color="black", linestyle="--", linewidth=0.4)
    ax_ins.axhline(y=lr_cdf_at_sr99, color="black", linestyle="--", linewidth=0.4)
    ax_ins.set_xlim(0, 10)
    ax_ins.set_ylim(0, 1.05)
    ax_ins.set_xticks([0, 5, 10])
    ax_ins.set_yticks([0, 0.5, 0.8, 1.0])
    ax_ins.tick_params(labelsize=2.5, length=1, width=0.4)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.28, bottom=0.20, right=0.95, top=0.95)
    out = OUTPUT_DIR / "cdf_by_tx_length.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_rin_distribution() -> None:
    """
    Plot RIN distribution as a histogram with KDE overlay.
    """
    setup_style()
    metadata_df = load_metadata()
    rin_values = metadata_df["RIN"].dropna().values

    fig, ax = plt.subplots()

    ax.hist(rin_values, bins=12, density=True, alpha=0.7, color=VIR_B,
            edgecolor="white", linewidth=0.3)

    kde = stats.gaussian_kde(rin_values)
    x_kde = np.linspace(rin_values.min() - 0.5, rin_values.max() + 0.5, 200)
    ax.plot(x_kde, kde(x_kde), color=VIR_A)

    mean_rin = rin_values.mean()
    ax.axvline(mean_rin, color="black", linestyle="--", linewidth=0.6,
               label=f"Mean: {mean_rin:.1f}")

    ax.set_xlabel("RIN")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    out = OUTPUT_DIR / "rin_distribution.png"
    fig.savefig(out)

    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


READS_SUMMARY_PATH = Path(DATA_DIR) / "reads_summary.tsv"


def plot_mapping_rate() -> None:
    """
    Plot mapping rate distribution as a KDE density plot with individual sample dots.
    """
    setup_style()
    reads_df = load_reads_summary(READS_SUMMARY_PATH)
    mapping_rates = reads_df["mapping_rate"].values

    fig, ax = plt.subplots()

    kde = stats.gaussian_kde(mapping_rates)
    x_kde = np.linspace(mapping_rates.min() - 0.5, mapping_rates.max() + 0.5, 200)
    y_kde = kde(x_kde)

    ax.fill_between(x_kde, y_kde, alpha=0.3, color=VIR_B, linewidth=0)
    ax.plot(x_kde, y_kde, color=VIR_A)

    jitter = np.random.default_rng(42).uniform(-0.02, 0.02, len(mapping_rates)) * y_kde.max()
    ax.scatter(mapping_rates, jitter + y_kde.max() * 0.05,
               s=3, alpha=0.6, color=VIR_B, edgecolor="white", linewidth=0.2, zorder=3)

    mean_rate = mapping_rates.mean()
    median_rate = np.median(mapping_rates)
    ax.axvline(mean_rate, color="black", linestyle="--", linewidth=0.6,
               label=f"Mean: {mean_rate:.1f}%")
    ax.axvline(median_rate, color="black", linestyle=":", linewidth=0.6,
               label=f"Median: {median_rate:.1f}%")

    ax.set_xlabel("Mapping rate (%)")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    out = OUTPUT_DIR / "mapping_rate.png"
    fig.savefig(out)

    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def load_coverage_arrays(cov_dir: Path, prefix: str) -> np.ndarray:
    """
    Load all coverage CSV files from a directory into a matrix.

    :param cov_dir: Directory containing coverage CSV files.
    :param prefix: Filename prefix to filter coverage files.
    :return: Matrix of relative coverage values with shape (n_samples, n_positions).
    """
    arrays = []
    for f in sorted(cov_dir.glob(f"{prefix}*_coverage.csv")):
        df = pd.read_csv(f, sep="\t")
        arrays.append(df["relative_coverage"].values)
    return np.array(arrays)


def plot_transcript_coverage() -> None:
    """
    Plot normalized transcript coverage profiles for SR, LR, and Watchmaker samples.
    """
    setup_style()
    data_dir = Path(DATA_DIR)

    sr_cov = load_coverage_arrays(data_dir / "sr_coverage", "sr_")
    lr_cov = load_coverage_arrays(data_dir / "lr_coverage", "lr_")
    wm_cov = load_coverage_arrays(data_dir / "wm_coverage", "wm_")
    log.info("Coverage samples — SR: %d, LR: %d, WM: %d",
             len(sr_cov), len(lr_cov), len(wm_cov))

    sr_mean = np.mean(sr_cov, axis=0)
    sr_mean = sr_mean / np.max(sr_mean)
    lr_mean = np.mean(lr_cov, axis=0)
    lr_mean = lr_mean / np.max(lr_mean)
    wm_mean = np.mean(wm_cov, axis=0)
    wm_mean = wm_mean / np.max(wm_mean)

    sr_std = np.std(sr_cov, axis=0) / np.max(np.mean(sr_cov, axis=0))
    lr_std = np.std(lr_cov, axis=0) / np.max(np.mean(lr_cov, axis=0))
    wm_std = np.std(wm_cov, axis=0) / np.max(np.mean(wm_cov, axis=0))

    x = np.arange(len(sr_mean))

    fig, ax = plt.subplots()

    ax.fill_between(x, sr_mean - sr_std, sr_mean + sr_std, alpha=0.2, color=VIR_A, linewidth=0)
    ax.fill_between(x, wm_mean - wm_std, wm_mean + wm_std, alpha=0.3, color=VIR_C, linewidth=0)
    ax.fill_between(x, lr_mean - lr_std, lr_mean + lr_std, alpha=0.2, color=VIR_B, linewidth=0)

    l_sr, = ax.plot(x, sr_mean, color=VIR_A, linestyle="--")
    l_wm, = ax.plot(x, wm_mean, color=VIR_C, linestyle=":")
    l_lr, = ax.plot(x, lr_mean, color=VIR_B)

    ax.set_ylabel("Normalized coverage")
    ax.set_xlabel("Transcript position")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(sr_mean) - 1)
    ax.set_xticks([0, len(sr_mean) - 1])
    ax.set_xticklabels(["5'", "3'"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend([l_sr, l_wm, l_lr],
              ["Illumina srRNA-seq", "Illumina srRNA-seq (globin-depletion)", "PacBio lrRNA-seq"],
              loc="lower center")

    out = OUTPUT_DIR / "coverage_profile.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_isoforms_per_gene() -> None:
    """
    Plot average number of genes by isoform count categories across all samples.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    category_order = ["1 isoform", "2-3 isoforms", "4-5 isoforms", ">=6 isoforms"]

    all_counts = []
    for sample_id in samples:
        isoforms = get_isoform_counts_per_sample(sample_id)
        cats = isoforms.apply(categorize_isoform_count)
        counts = cats.value_counts()
        for cat in category_order:
            if cat not in counts:
                counts[cat] = 0
        all_counts.append(counts[category_order])

    counts_df = pd.DataFrame(all_counts, columns=category_order)
    avg = counts_df.mean()
    std = counts_df.std()

    colors = [cm.viridis(v) for v in [0.15, 0.4, 0.65, 0.95]]

    fig, ax = plt.subplots()
    x = np.arange(len(category_order))
    bars = ax.bar(x, avg, yerr=std, capsize=1.5, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.3, error_kw={"linewidth": 0.5})

    for bar, a, s in zip(bars, avg, std):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 30,
                f"{a:.0f}", ha="center", va="bottom", fontsize=3.5)

    ax.set_xlabel("Isoforms per gene")
    ax.set_ylabel("Avg. number of genes")
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2-3", "4-5", "\u22656"])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "isoforms_per_gene.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


CATEGORY_ORDER = [
    "full-splice_match", "incomplete-splice_match",
    "novel_in_catalog", "novel_not_in_catalog",
    "fusion", "genic_intron", "antisense", "genic", "intergenic",
]
CATEGORY_LABELS = ["FSM", "ISM", "NIC", "NNC", "Fusion", "Genic\nintron", "Antisense", "Genic", "Inter-\ngenic"]


def _violin_by_category(data_matrix: np.ndarray, all_categories: list[str], ylabel: str, outname: str) -> None:
    """
    Shared violin plot logic for total and trio-unique transcripts.

    :param data_matrix: Matrix of counts with shape (n_samples, n_categories).
    :param all_categories: List of structural category names.
    :param ylabel: Label for the y-axis.
    :param outname: Output filename for the saved figure.
    """
    setup_style()

    records = []
    for i in range(data_matrix.shape[0]):
        for j, cat in enumerate(all_categories):
            records.append({"category": cat, "count": data_matrix[i, j]})
    df_long = pd.DataFrame(records)

    order = [c for c in CATEGORY_ORDER if c in all_categories]
    labels = [CATEGORY_LABELS[CATEGORY_ORDER.index(c)] for c in order]

    n_cats = len(order)
    colors = [cm.viridis(v) for v in np.linspace(0.1, 0.9, n_cats)]

    fig, ax = plt.subplots(figsize=(2, 1))
    parts = ax.violinplot(
        [df_long.loc[df_long["category"] == cat, "count"].values for cat in order],
        positions=range(n_cats), showmedians=True, showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("black")
        body.set_linewidth(0.3)
        body.set_alpha(0.8)
    parts["cmedians"].set_linewidth(0.5)
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Structural category")
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / outname
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_all_tx_violin() -> None:
    """
    Plot violin distributions of all transcript counts by structural category.
    """
    samples = get_long_read_sample_ids()
    all_tx = get_all_tx(samples)

    all_categories = set()
    for df in all_tx.values():
        all_categories.update(df["structural_category"].unique())
    all_categories = sorted(all_categories)

    data_matrix = np.zeros((len(all_tx), len(all_categories)))
    for i, (_, df) in enumerate(all_tx.items()):
        counts = df["structural_category"].value_counts().to_dict()
        for j, cat in enumerate(all_categories):
            data_matrix[i, j] = counts.get(cat, 0)

    _violin_by_category(data_matrix, all_categories, "Number of transcripts", "all_tx_violin.png")


def plot_unique_tx_violin() -> None:
    """
    Plot violin distributions of trio-unique transcript counts by structural category.
    """
    samples = get_long_read_sample_ids()
    unique_tx = get_unique_tx(samples, rule_filter=False)

    all_categories = set()
    for df in unique_tx.values():
        all_categories.update(df["structural_category"].unique())
    all_categories = sorted(all_categories)

    data_matrix = np.zeros((len(unique_tx), len(all_categories)))
    for i, (_, df) in enumerate(unique_tx.items()):
        counts = df["structural_category"].value_counts().to_dict()
        for j, cat in enumerate(all_categories):
            data_matrix[i, j] = counts.get(cat, 0)

    _violin_by_category(data_matrix, all_categories, "Number of unique transcripts", "unique_tx_violin.png")


def plot_combined_tx_violin() -> None:
    """
    Side-by-side violins for all vs trio-unique transcripts per SQANTI3 category.
    """
    setup_style()
    samples = get_long_read_sample_ids()

    all_tx = get_all_tx(samples)
    all_categories = set()
    for df in all_tx.values():
        all_categories.update(df["structural_category"].unique())
    for df in get_unique_tx(samples, rule_filter=False).values():
        all_categories.update(df["structural_category"].unique())
    all_categories = sorted(all_categories)

    all_matrix = np.zeros((len(all_tx), len(all_categories)))
    for i, (_, df) in enumerate(all_tx.items()):
        counts = df["structural_category"].value_counts().to_dict()
        for j, cat in enumerate(all_categories):
            all_matrix[i, j] = counts.get(cat, 0)

    unique_tx = get_unique_tx(samples, rule_filter=False)
    uniq_matrix = np.zeros((len(unique_tx), len(all_categories)))
    for i, (_, df) in enumerate(unique_tx.items()):
        counts = df["structural_category"].value_counts().to_dict()
        for j, cat in enumerate(all_categories):
            uniq_matrix[i, j] = counts.get(cat, 0)

    order = [c for c in CATEGORY_ORDER if c in all_categories]
    labels = [CATEGORY_LABELS[CATEGORY_ORDER.index(c)] for c in order]
    n_cats = len(order)

    width = 0.35
    fig, ax = plt.subplots(figsize=(2.5, 1.2))

    for k, cat in enumerate(order):
        j = all_categories.index(cat)
        all_vals = all_matrix[:, j]
        uniq_vals = uniq_matrix[:, j]

        pos_all = k - width / 2
        pos_uniq = k + width / 2

        vp1 = ax.violinplot([all_vals], positions=[pos_all], widths=width,
                            showmedians=True, showextrema=False)
        for body in vp1["bodies"]:
            body.set_facecolor(VIR_A)
            body.set_edgecolor("black")
            body.set_linewidth(0.3)
            body.set_alpha(0.7)
        vp1["cmedians"].set_linewidth(0.5)
        vp1["cmedians"].set_color("black")

        vp2 = ax.violinplot([uniq_vals], positions=[pos_uniq], widths=width,
                            showmedians=True, showextrema=False)
        for body in vp2["bodies"]:
            body.set_facecolor(VIR_B)
            body.set_edgecolor("black")
            body.set_linewidth(0.3)
            body.set_alpha(0.7)
        vp2["cmedians"].set_linewidth(0.5)
        vp2["cmedians"].set_color("black")

        ax.errorbar(pos_all, all_vals.mean(), yerr=all_vals.std(),
                     fmt="o", color="black", markersize=1, capsize=1.5,
                     capthick=0.4, elinewidth=0.4, zorder=5)
        ax.errorbar(pos_uniq, uniq_vals.mean(), yerr=uniq_vals.std(),
                     fmt="o", color="black", markersize=1, capsize=1.5,
                     capthick=0.4, elinewidth=0.4, zorder=5)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of transcripts")
    ax.set_xlabel("Structural category")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=VIR_A, edgecolor="black", linewidth=0.3, alpha=0.7, label="All"),
                 Patch(facecolor=VIR_B, edgecolor="black", linewidth=0.3, alpha=0.7, label="Trio-unique")],
        loc="upper right",
    )

    out = OUTPUT_DIR / "supplemental_2A_combined_tx_violin.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_2c_novel_enrichment() -> None:
    """
    Paired comparison of novel isoform fraction between all and trio-unique transcripts per sample.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    all_tx = get_all_tx(samples)
    unique_tx = get_unique_tx(samples, rule_filter=False)

    all_fracs: list[float] = []
    uniq_fracs: list[float] = []
    for sid in samples:
        for src, fracs in [(all_tx, all_fracs), (unique_tx, uniq_fracs)]:
            df = src.get(sid)
            if df is not None and len(df) > 0:
                vc = df["structural_category"].value_counts().to_dict()
                ref = vc.get("full-splice_match", 0) + vc.get("incomplete-splice_match", 0)
                nov = vc.get("novel_in_catalog", 0) + vc.get("novel_not_in_catalog", 0)
                fracs.append(nov / (ref + nov) * 100 if (ref + nov) > 0 else 0)
            else:
                fracs.append(0)

    all_arr = np.array(all_fracs)
    uniq_arr = np.array(uniq_fracs)
    _, p_val = stats.wilcoxon(all_arr, uniq_arr)

    fig, ax = plt.subplots(figsize=(1, 1.2))

    for a, u in zip(all_arr, uniq_arr):
        ax.plot([0, 1], [a, u], color="#999999", linewidth=0.3, alpha=0.5)

    ax.scatter([0] * len(all_arr), all_arr, c=VIR_A, s=6, zorder=5,
               edgecolor="white", linewidth=0.2)
    ax.scatter([1] * len(uniq_arr), uniq_arr, c=VIR_B, s=6, zorder=5,
               edgecolor="white", linewidth=0.2)

    bp = ax.boxplot([all_arr, uniq_arr], positions=[0, 1], widths=0.3,
                    patch_artist=True, showfliers=False, zorder=4,
                    medianprops=dict(color="black", linewidth=0.8),
                    whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5))
    bp["boxes"][0].set(facecolor=VIR_A, alpha=0.3, linewidth=0.5)
    bp["boxes"][1].set(facecolor=VIR_B, alpha=0.3, linewidth=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["All", "Trio-unique"])
    ax.set_ylabel("Novel isoform fraction (%)")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_max = max(all_arr.max(), uniq_arr.max())
    ax.plot([0, 0, 1, 1], [y_max + 1, y_max + 2, y_max + 2, y_max + 1],
            color="black", linewidth=0.5)
    ax.text(0.5, y_max + 2.5, f"*** p = {p_val:.2e}", ha="center", va="bottom", fontsize=3.5)

    out = OUTPUT_DIR / "supplemental_2E_novel_enrichment.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_da_genes_expressed() -> None:
    """
    Plot bar chart of expressed vs not-expressed disease-associated genes.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    all_tx = get_all_tx(samples)
    da_genes_df = get_disease_associated_genes()
    _, summary = get_expressed_da_genes(all_tx, da_genes_df)

    categories = ["Expressed", "Not\nexpressed"]
    counts = [summary["expressed_da_genes"], summary["not_expressed_da_genes"]]
    colors = [cm.viridis(0.3), cm.viridis(0.75)]

    fig, ax = plt.subplots()
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.3, width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f"{count:,}", ha="center", va="bottom", fontsize=4, fontweight="bold")

    pct = summary["pct_da_genes_expressed"]
    ax.set_ylabel("Number of DA genes")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.98, 0.97, f"{pct:.1f}% of {summary['total_da_genes']:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=3.5, color="gray")

    out = OUTPUT_DIR / "da_genes_expressed.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_da_genes_by_evidence(summary: dict | None = None) -> None:
    """
    Plot expressed disease-associated genes grouped by ClinGen evidence level.

    :param summary: Summary dictionary from get_expressed_da_genes. If None, data is loaded fresh.
    """
    setup_style()

    if summary is None:
        samples = get_long_read_sample_ids()
        all_tx = get_all_tx(samples)
        da_genes_df = get_disease_associated_genes()
        _, summary = get_expressed_da_genes(all_tx, da_genes_df)

    evidence_order = ["Definitive", "Strong", "Moderate", "Limited"]
    evidence_counts = [summary["evidence_counts"].get(e, 0) for e in evidence_order]
    colors = [cm.viridis(v) for v in [0.1, 0.35, 0.6, 0.85]]

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(evidence_order)), evidence_counts, color=colors,
                  edgecolor="black", linewidth=0.3, width=0.6)

    for bar, count in zip(bars, evidence_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(evidence_counts) * 0.02,
                f"{count:,}", ha="center", va="bottom", fontsize=4, fontweight="bold")

    ax.set_xticks(range(len(evidence_order)))
    ax.set_xticklabels(evidence_order, rotation=45, ha="right")
    ax.set_xlabel("ClinGen evidence level")
    ax.set_ylabel("Number of expressed DA genes")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "da_genes_by_evidence.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_fig2b_da_genes_combined() -> None:
    """
    Stacked bar: expressed DA genes coloured by evidence level, not-expressed grey.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    all_tx = get_all_tx(samples)
    da_genes_df = get_disease_associated_genes()
    _, summary = get_expressed_da_genes(all_tx, da_genes_df)

    evidence_order = ["Definitive", "Strong", "Moderate"]
    evidence_counts = [summary["evidence_counts"].get(e, 0) for e in evidence_order]
    not_expressed = summary["not_expressed_da_genes"]
    colors = [cm.viridis(v) for v in [0.1, 0.35, 0.6]]

    fig, ax = plt.subplots(figsize=(1, 1))

    bottom = 0
    for i, (_, cnt) in enumerate(zip(evidence_order, evidence_counts)):
        ax.bar(0, cnt, bottom=bottom, color=colors[i], edgecolor="black",
               linewidth=0.3, width=0.5)
        bottom += cnt

    ax.bar(1, not_expressed, color="#BBBBBB", edgecolor="black", linewidth=0.3, width=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Expressed", "Not\nexpressed"])
    ax.set_ylabel("Number of DA genes")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3,
              label=f"{ev} (n={cnt:,})")
        for i, (ev, cnt) in enumerate(zip(evidence_order, evidence_counts))
    ]
    legend_handles.append(Patch(facecolor="#BBBBBB", edgecolor="black", linewidth=0.3,
                                label=f"Not expressed (n={not_expressed:,})"))
    ax.legend(handles=legend_handles, loc="center right",
              fontsize=2.5, handlelength=0.5, handleheight=0.5,
              borderpad=0.3, labelspacing=0.2)

    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.18)
    out = OUTPUT_DIR / "Fig2B_da_genes_combined.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


DISTANCE_ORDER = ["2 kb", "20 kb", ">20 kb", "interchromosomal"]
DISTANCE_LABELS = ["2 kb", "20 kb", ">20 kb", "Inter-\nchrom."]


def plot_unique_fusion_by_distance() -> None:
    """
    Plot average unique fusion transcript counts grouped by fusion distance category.
    """
    setup_style()

    fusion_chars = pd.read_csv(
        Path(DATA_DIR) / "unique_fusion_tx" / "fusion_characteristics_ml_filtered.csv"
    )

    counts = (
        fusion_chars.groupby(["sample_id", "distance_category"])
        .size()
        .reset_index(name="count")
    )

    sample_ids = counts["sample_id"].unique()
    full_index = pd.MultiIndex.from_product(
        [sample_ids, DISTANCE_ORDER], names=["sample_id", "distance_category"]
    )
    counts = (
        counts.set_index(["sample_id", "distance_category"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    n_cats = len(DISTANCE_ORDER)
    colors = [cm.viridis(v) for v in np.linspace(0.1, 0.9, n_cats)]

    avg = [counts.loc[counts["distance_category"] == cat, "count"].mean() for cat in DISTANCE_ORDER]
    std = [counts.loc[counts["distance_category"] == cat, "count"].std() for cat in DISTANCE_ORDER]

    fig, ax = plt.subplots(figsize=(1, 1))
    x = np.arange(n_cats)
    bars = ax.bar(x, avg, yerr=std, capsize=1.5, color=colors,
                  edgecolor="black", linewidth=0.3, error_kw={"linewidth": 0.5})

    for bar, a, s in zip(bars, avg, std):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + max(avg) * 0.03,
                f"{a:.1f}", ha="center", va="bottom", fontsize=3.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DISTANCE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Avg. unique fusion tx")
    ax.set_xlabel("Fusion distance")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / "unique_fusion_by_distance.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def _plot_alt5_shift_vs_novel(metrics_path: Path, outname: str, has_parents: bool) -> None:
    """
    Shared scatter plot for alt 5' shift vs novel burden, manuscript style.

    :param metrics_path: Path to the sample shift metrics TSV file.
    :param outname: Output filename for the saved figure.
    :param has_parents: Whether to color parent samples differently.
    """
    setup_style()

    df = pd.read_csv(metrics_path, sep="\t")
    df = df.dropna(subset=["gene_enrichment_z"])
    if df.empty:
        log.warning("No valid data in %s — skipping", metrics_path)
        return

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()

    is_target = df["sample_id"].str.contains("<target_sample>")
    df_other = df[~is_target]
    df_target = df[is_target]

    if has_parents:
        colors_other = [VIR_A if "_3_R1" in s else VIR_B for s in df_other["sample_id"]]
    else:
        colors_other = [VIR_A] * len(df_other)

    fig, ax = plt.subplots(figsize=(1, 1))

    ax.scatter(
        df_other["gene_enrichment_z"].values,
        df_other["n_psi_outliers"].values,
        c=colors_other, edgecolors="black", linewidth=0.2, s=4, zorder=3,
        marker="o",
    )

    if not df_target.empty:
        ax.scatter(
            df_target["gene_enrichment_z"].values,
            df_target["n_psi_outliers"].values,
            c=[VIR_A], edgecolors="black", linewidth=0.2, s=12, zorder=4,
            marker="*",
        )

    ax.set_xlabel("Novel alt 5\u2032 ss\nburden z-score")
    ax.set_ylabel("Alt 5\u2032 usage shift\nin existing isoforms")

    if outlier_std > 0:
        ax.axhline(outlier_mean + 2 * outlier_std, color="red", linestyle="--", linewidth=0.4)
    ax.axvline(3, color="blue", linestyle=":", linewidth=0.4)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_A,
               markeredgecolor="black", markeredgewidth=0.3, markersize=2.5, label="Proband"),
    ]
    if has_parents:
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_B,
                   markeredgecolor="black", markeredgewidth=0.3, markersize=2.5, label="Parent"),
        )
    handles += [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=VIR_A,
               markeredgecolor="black", markeredgewidth=0.3, markersize=3.5, label="<target_sample>"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=0.4, label="+2 SD"),
        Line2D([0], [0], color="blue", linestyle=":", linewidth=0.4, label="+3 SD"),
    ]
    ax.legend(handles=handles, loc="lower right")

    out = OUTPUT_DIR / outname
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_alt5_shift_vs_novel_bam() -> None:
    """
    Plot alt 5' shift vs novel burden scatter for BAM-based analysis.
    """
    _plot_alt5_shift_vs_novel(
        Path(DATA_DIR) / "outrider" / "alt5_shift_analysis_bam" / "sample_shift_metrics.tsv",
        "alt5_shift_vs_novel_bam.png",
        has_parents=True,
    )


def plot_alt5_shift_vs_novel_sr() -> None:
    """
    Plot alt 5' shift vs novel burden scatter for short-read analysis.
    """
    _plot_alt5_shift_vs_novel(
        Path(DATA_DIR) / "outrider" / "alt5_shift_analysis_sr" / "sample_shift_metrics.tsv",
        "alt5_shift_vs_novel_sr.png",
        has_parents=False,
    )


def plot_supplemental_3b() -> None:
    """
    Two-panel strip+box: PSI shift outliers and novel alt 5' counts per sample.
    """
    setup_style()

    metrics_path = (
        Path(DATA_DIR) / "outrider" / "alt5_shift_analysis_bam" / "sample_shift_metrics.tsv"
    )
    df = pd.read_csv(metrics_path, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(1.2, 1))

    for ax, col, ylabel in zip(
        axes,
        ["n_psi_outliers", "n_truly_novel_alt5"],
        ["Elevated PSI\njunctions (z \u2265 2)", "Novel alt 5\u2032\njunctions"],
    ):
        vals = df[col]
        ax.boxplot(
            vals, positions=[0], widths=0.5, showfliers=False,
            boxprops={"linewidth": 0.4, "color": "black"},
            whiskerprops={"linewidth": 0.4},
            capprops={"linewidth": 0.4},
            medianprops={"linewidth": 0.6, "color": "black"},
        )
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            jitter, vals,
            c=VIR_A, s=3, edgecolors="black", linewidth=0.15, zorder=3,
        )
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=0.8)
    out = OUTPUT_DIR / "supplemental_3B_alt5_metrics.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_3c() -> None:
    """
    Scatter: alt 5' PSI shift outliers vs truly novel alt 5' count (raw).
    """
    setup_style()

    metrics_path = (
        Path(DATA_DIR) / "outrider" / "alt5_shift_analysis_bam" / "sample_shift_metrics.tsv"
    )
    df = pd.read_csv(metrics_path, sep="\t")
    df = df.dropna(subset=["n_truly_novel_alt5"])
    if df.empty:
        log.warning("No valid data in %s — skipping", metrics_path)
        return

    outlier_mean = df["n_psi_outliers"].mean()
    outlier_std = df["n_psi_outliers"].std()
    novel_mean = df["n_truly_novel_alt5"].mean()
    novel_std = df["n_truly_novel_alt5"].std()

    is_target = df["sample_id"].str.contains("<target_sample>")
    df_other = df[~is_target]
    df_target = df[is_target]

    colors_other = [VIR_A if "_3_R1" in s else VIR_B for s in df_other["sample_id"]]

    fig, ax = plt.subplots(figsize=(1, 1))

    ax.scatter(
        df_other["n_truly_novel_alt5"].values,
        df_other["n_psi_outliers"].values,
        c=colors_other, edgecolors="black", linewidth=0.2, s=4, zorder=3,
        marker="o",
    )
    if not df_target.empty:
        ax.scatter(
            df_target["n_truly_novel_alt5"].values,
            df_target["n_psi_outliers"].values,
            c=[VIR_A], edgecolors="black", linewidth=0.2, s=12, zorder=4,
            marker="*",
        )

    ax.set_xlabel("Novel\nalt 5\u2032 junctions")
    ax.set_ylabel("Alt 5\u2032 usage shift\nin existing isoforms")

    if outlier_std > 0:
        ax.axhline(outlier_mean + 2 * outlier_std, color="red", linestyle="--", linewidth=0.4)
    if novel_std > 0:
        ax.axvline(novel_mean + 3 * novel_std, color="blue", linestyle=":", linewidth=0.4)

    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_A,
               markeredgecolor="black", markeredgewidth=0.3, markersize=2.5, label="Proband"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_B,
               markeredgecolor="black", markeredgewidth=0.3, markersize=2.5, label="Parent"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=VIR_A,
               markeredgecolor="black", markeredgewidth=0.3, markersize=3.5, label="<target_sample>"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=0.4, label="+2 SD"),
        Line2D([0], [0], color="blue", linestyle=":", linewidth=0.4, label="+3 SD"),
    ]
    ax.legend(handles=handles, loc="lower right")

    out = OUTPUT_DIR / "supplemental_3C_alt5_shift_vs_novel_raw.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)



FRASER_TARGET = {"<sample_id_1>", "<sample_id_2>"}

JACCARD_PATH = FRASER_JACCARD_FILEPATH
PSI3_PATH = FRASER_PSI3_FILEPATH


def _plot_fraser_ranked_bars(csv_path: Path, ylabel: str, outname: str) -> None:
    """
    Ranked bar chart of splicing outlier counts per sample, manuscript style.

    :param csv_path: Path to the FRASER results CSV file.
    :param ylabel: Label for the y-axis.
    :param outname: Output filename for the saved figure.
    """
    setup_style()

    df = pd.read_csv(csv_path)
    df = df[df["padjust"] <= 0.05]
    counts = df["sampleID"].value_counts().sort_values(ascending=False)

    colors = [VIR_A if str(sid) in FRASER_TARGET else VIR_B for sid in counts.index]

    fig, ax = plt.subplots(figsize=(1, 1))
    x = np.arange(len(counts))
    ax.bar(x, counts.values, color=colors, edgecolor="none", width=1.0)

    labels = counts.index.astype(str).tolist()
    for i, sid in enumerate(labels):
        if sid in FRASER_TARGET:
            ax.annotate(
                sid.replace("_R1", ""),
                xy=(i, counts.values[i]),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=3, fontweight="bold",
                color=VIR_A,
            )

    ax.set_xlabel("Sample rank")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUTPUT_DIR / outname
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_fraser_jaccard() -> None:
    """
    Plot ranked bar chart of FRASER Jaccard splicing outlier counts per sample.
    """
    _plot_fraser_ranked_bars(JACCARD_PATH, "Splicing outliers\n(Jaccard)", "fraser_jaccard_ranked.png")


def plot_fraser_psi3() -> None:
    """
    Plot ranked bar chart of FRASER PSI3 splicing outlier counts per sample.
    """
    _plot_fraser_ranked_bars(PSI3_PATH, "Splicing outliers\n(PSI3)", "fraser_psi3_ranked.png")


JUNC_DIR = Path(DATA_DIR) / "outrider" / "all_junction_counts"

SASHIMI_TARGETS = [
    ("<sample_id_3>", "GRK3", "CRYBB2P1", "chr22", 25520734, 25604376, "> 20 kb"),
    ("<sample_id_4>", "TRBC1", "TRBC2", "chr7", 142792081, 142801943, "2\u201320 kb"),
    ("<sample_id_5>", "TRAPPC5", "MCEMP1", "chr19", 7678984, 7682241, "< 2 kb"),
    ("<sample_id_6>", "CIMAP1B", "SCO2", "chr22", 50524425, 50529903, "2\u201320 kb"),
]


def _load_shared_junctions(
    sample_id: str, chrom: str, donor: int, acceptor: int,
    allowed_isoforms: set[str] | None = None,
) -> pd.DataFrame:
    """
    Load junctions sharing a donor or acceptor site from a sample junction file.

    :param sample_id: Identifier for the sample.
    :param chrom: Chromosome name to filter on.
    :param donor: Genomic start coordinate of the fusion junction.
    :param acceptor: Genomic end coordinate of the fusion junction.
    :param allowed_isoforms: If provided, restrict to junctions from these isoforms.
    :return: DataFrame of junctions matching the donor or acceptor site.
    """
    fpath = JUNC_DIR / f"{sample_id}_all_junctions.tsv"
    cols = ["chrom", "genomic_start_coord", "genomic_end_coord", "junction_unique_read_counts", "isoform"]
    df = pd.read_csv(fpath, sep="\t", usecols=cols)
    if allowed_isoforms is not None:
        df = df[df["isoform"].isin(allowed_isoforms)]
    df = df[df["chrom"] == chrom]
    return df[(df["genomic_start_coord"] == donor) | (df["genomic_end_coord"] == acceptor)]


def _get_gtex_shared(chrom: str, donor: int, acceptor: int,
                     gtex_junc_path: Path) -> pd.DataFrame:
    """
    Query GTEx junction BED file for junctions sharing a donor or acceptor site.

    :param chrom: Chromosome name.
    :param donor: Genomic donor coordinate.
    :param acceptor: Genomic acceptor coordinate.
    :param gtex_junc_path: Path to the tabix-indexed GTEx junctions BED file.
    :return: DataFrame with start, end, and reads columns for matching junctions.
    """
    tbx = pysam.TabixFile(str(gtex_junc_path))
    records: list[tuple[int, int, int]] = []
    lo = min(donor, acceptor) - 1000
    hi = max(donor, acceptor) + 1000
    try:
        for row in tbx.fetch(chrom, max(0, lo), hi):
            fields = row.split("\t")
            s, e = int(fields[1]) + 1, int(fields[2])  # BED 0-based start → 1-based
            if s == donor or e == acceptor:
                uniq_mapped = 0
                for part in fields[3].split(";"):
                    if part.startswith("uniquely_mapped="):
                        uniq_mapped = int(part.split("=")[1])
                records.append((s, e, uniq_mapped))
    except ValueError:
        pass
    tbx.close()
    if records:
        return pd.DataFrame(records, columns=["start", "end", "reads"])
    return pd.DataFrame(columns=["start", "end", "reads"])


def _draw_arc(ax: plt.Axes, x1: int, x2: int, height: float, color: str,
              lw: float = 0.5) -> None:
    """
    Draw a semicircular arc between two genomic positions on an axes.

    :param ax: Matplotlib axes to draw on.
    :param x1: Left genomic coordinate.
    :param x2: Right genomic coordinate.
    :param height: Height of the arc.
    :param color: Color of the arc.
    :param lw: Line width of the arc.
    """
    mid = (x1 + x2) / 2
    width = abs(x2 - x1)
    arc = mpatches.Arc((mid, 0), width, height * 2, angle=0, theta1=0, theta2=180,
                        color=color, lw=lw, alpha=0.8)
    ax.add_patch(arc)


def _plot_sashimi(sample_id: str, gene1: str, gene2: str, chrom: str,
                  junc_start: int, junc_end: int, dist_cat: str = "",
                  *, gtex_junc_path: Path) -> None:
    """
    Generate a sashimi-style arc plot for a fusion junction across three tracks.

    :param sample_id: Identifier for the sample containing the fusion.
    :param gene1: Name of the first gene in the fusion.
    :param gene2: Name of the second gene in the fusion.
    :param chrom: Chromosome of the fusion junction.
    :param junc_start: Genomic start coordinate of the fusion junction.
    :param junc_end: Genomic end coordinate of the fusion junction.
    :param dist_cat: Distance category label for the title.
    :param gtex_junc_path: Path to the tabix-indexed GTEx junctions BED file.
    """
    setup_style()
    donor, acceptor = junc_start, junc_end

    all_sample_ids = sorted(
        f.stem.replace("_all_junctions", "") for f in JUNC_DIR.glob("*_all_junctions.tsv")
    )

    family_prefix = "_".join(sample_id.split("_")[:2])

    sample_juncs = _load_shared_junctions(sample_id, chrom, donor, acceptor)

    other_frames: list[pd.DataFrame] = []
    for sid in all_sample_ids:
        if sid.startswith(family_prefix):
            continue
        ml_isoforms = set(read_sqanti3_annotated(sid, rules_filter=False)["isoform"])
        oj = _load_shared_junctions(sid, chrom, donor, acceptor, allowed_isoforms=ml_isoforms)
        if not oj.empty:
            other_frames.append(oj)
    if other_frames:
        all_other = pd.concat(other_frames)
        avg_juncs = (
            all_other.groupby(["genomic_start_coord", "genomic_end_coord"])["junction_unique_read_counts"]
            .mean().reset_index()
        )
        avg_juncs.columns = ["start", "end", "reads"]
    else:
        avg_juncs = pd.DataFrame(columns=["start", "end", "reads"])

    gtex_juncs = _get_gtex_shared(chrom, donor, acceptor, gtex_junc_path)

    all_coords: list[int] = [donor, acceptor]
    for df_check in [sample_juncs, avg_juncs, gtex_juncs]:
        if not df_check.empty:
            scol = "genomic_start_coord" if "genomic_start_coord" in df_check.columns else "start"
            ecol = "genomic_end_coord" if "genomic_end_coord" in df_check.columns else "end"
            all_coords.extend(df_check[scol].tolist() + df_check[ecol].tolist())
    xmin, xmax = min(all_coords), max(all_coords)
    pad = max(1000, int((xmax - xmin) * 0.08))
    xmin -= pad
    xmax += pad

    FUSION_COLOR = "#d62728"

    fig, axes = plt.subplots(3, 1, figsize=(2.5, 1.8), sharex=True)
    track_labels = [
        "Family " + "_".join(sample_id.split("_")[:2]),
        "Rest of the\nlrRNA-seq cohort\n(n=57)",
        "GTEx blood\n(n=755)",
    ]
    track_colors = [VIR_A, VIR_B, VIR_C]

    datasets: list[list[tuple]] = []
    if not sample_juncs.empty:
        datasets.append([(r["genomic_start_coord"], r["genomic_end_coord"], r["junction_unique_read_counts"]) for _, r in sample_juncs.iterrows()])
    else:
        datasets.append([])
    if not avg_juncs.empty:
        datasets.append([(r["start"], r["end"], r["reads"]) for _, r in avg_juncs.iterrows()])
    else:
        datasets.append([])
    if not gtex_juncs.empty:
        datasets.append([(r["start"], r["end"], r["reads"]) for _, r in gtex_juncs.iterrows()])
    else:
        datasets.append([])

    for idx, (ax, label, _, data) in enumerate(zip(axes, track_labels, track_colors, datasets)):
        ax.set_ylabel(label, fontsize=3.5, rotation=0, ha="right", va="center", labelpad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.axhline(0, color="grey", lw=0.2)

        if not data:
            ax.set_ylim(-0.02, 0.5)
            continue

        max_reads = max(d[2] for d in data)
        if max_reads == 0:
            max_reads = 1
        max_h = 0.0

        fusion_label_h = 0.0
        for s, e, reads in data:
            is_fusion = s == donor and e == acceptor
            h = max(0.15, reads / max_reads)
            c = FUSION_COLOR if is_fusion else "grey"
            _draw_arc(ax, s, e, h, c, lw=0.8 if is_fusion else 0.4)
            if is_fusion:
                mid = (s + e) / 2
                label_text = (
                    str(int(reads))
                    if isinstance(reads, (int, np.integer)) or reads == int(reads)
                    else f"{reads:.1f}"
                )
                fusion_label_h = h * 1.05
                ax.text(mid, fusion_label_h, label_text, ha="center", va="bottom",
                        fontsize=2.5, color=FUSION_COLOR, fontweight="bold")
            max_h = max(max_h, h)

        ax.set_ylim(-0.02, max_h * 1.4 + 0.1)
        ax.set_xlim(xmin, xmax)

    for ax in axes:
        ax.axvline(donor, color="grey", lw=0.2, ls="-", alpha=0.4)
        ax.axvline(acceptor, color="grey", lw=0.2, ls="-", alpha=0.4)

    axes[-1].set_xticks([donor, acceptor])
    axes[-1].set_xticklabels([gene1, gene2], fontsize=4, fontstyle="italic")
    axes[-1].tick_params(axis="x", length=0, pad=2)
    title = f"{gene1}\u2013{gene2}"
    if dist_cat:
        title += f" ({dist_cat})"
    axes[0].set_title(title, fontsize=5, fontstyle="italic")
    fig.subplots_adjust(hspace=0.1)

    outname = f"sashimi_{sample_id}_{gene1}_{gene2}.png"
    out = OUTPUT_DIR / outname
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_sashimi_all(gtex_junc_path: Path) -> None:
    """
    Generate sashimi plots for all predefined fusion junction targets.

    :param gtex_junc_path: Path to the tabix-indexed GTEx junctions BED file.
    """
    for sid, g1, g2, chrom, js, je, dc in SASHIMI_TARGETS:
        _plot_sashimi(sid, g1, g2, chrom, js, je, dc, gtex_junc_path=gtex_junc_path)


def plot_supplemental_1a() -> None:
    """
    Bar chart of total transcript isoforms per sample (ML-filtered).

    Samples sorted by group (Probands first, then Parents), descending
    within each group. A dashed median line is drawn across the panel.
    """
    setup_style()
    samples = get_long_read_sample_ids()

    records: list[dict] = []
    for sid in samples:
        df = read_sqanti3_annotated(sid, rules_filter=False)
        records.append({"sample_id": sid, "n_isoforms": len(df)})
    counts_df = pd.DataFrame(records)

    counts_df["group"] = counts_df["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )
    group_rank = {"Proband": 0, "Parent": 1}
    counts_df["group_rank"] = counts_df["group"].map(group_rank)
    counts_df = counts_df.sort_values(
        ["group_rank", "n_isoforms"], ascending=[True, False]
    ).reset_index(drop=True)

    n_probands = (counts_df["group"] == "Proband").sum()
    colors = [VIR_A if g == "Proband" else VIR_B for g in counts_df["group"]]

    fig, ax = plt.subplots(figsize=(2.5, 1))
    x = np.arange(len(counts_df))
    ax.bar(x, counts_df["n_isoforms"], color=colors, edgecolor="white", linewidth=0.2, width=1.0)

    median_val = counts_df["n_isoforms"].median()
    ax.axhline(median_val, color="#d62728", linestyle="--", linewidth=0.5,
               label=f"Median: {median_val:,.0f}")

    ax.axvline(n_probands - 0.5, color="black", linestyle="--", linewidth=0.4)

    proband_mid = (n_probands - 1) / 2
    parent_mid = n_probands + (len(counts_df) - n_probands - 1) / 2
    y_top = counts_df["n_isoforms"].max()
    ax.text(proband_mid, y_top * 1.06, "Probands", ha="center", fontsize=4, fontweight="bold")
    ax.text(parent_mid, y_top * 1.06, "Parents", ha="center", fontsize=4, fontweight="bold")

    ax.set_xlabel("Sample")
    ax.set_ylabel("Transcript isoforms")
    ax.set_xticks([])
    ax.set_ylim(top=y_top * 1.15)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")

    out = OUTPUT_DIR / "supplemental_2A_isoforms_per_sample.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_1b() -> None:
    """
    Bar chart of trio-unique transcript isoforms per sample.

    Samples sorted by group (Probands first, then Parents), descending
    within each group. A dashed median line is drawn across the panel.
    """
    setup_style()
    samples = get_long_read_sample_ids()
    unique_tx = get_unique_tx(samples, rule_filter=False)

    records: list[dict] = []
    for sid in samples:
        df = unique_tx.get(sid)
        records.append({"sample_id": sid, "n_unique": len(df) if df is not None else 0})
    counts_df = pd.DataFrame(records)

    counts_df["group"] = counts_df["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )
    group_rank = {"Proband": 0, "Parent": 1}
    counts_df["group_rank"] = counts_df["group"].map(group_rank)
    counts_df = counts_df.sort_values(
        ["group_rank", "n_unique"], ascending=[True, False]
    ).reset_index(drop=True)

    n_probands = (counts_df["group"] == "Proband").sum()
    colors = [VIR_A if g == "Proband" else VIR_B for g in counts_df["group"]]

    fig, ax = plt.subplots(figsize=(2.5, 1))
    x = np.arange(len(counts_df))
    ax.bar(x, counts_df["n_unique"], color=colors, edgecolor="white", linewidth=0.2, width=1.0)

    median_val = counts_df["n_unique"].median()
    ax.axhline(median_val, color="#d62728", linestyle="--", linewidth=0.5,
               label=f"Median: {median_val:,.0f}")

    ax.axvline(n_probands - 0.5, color="black", linestyle="--", linewidth=0.4)

    proband_mid = (n_probands - 1) / 2
    parent_mid = n_probands + (len(counts_df) - n_probands - 1) / 2
    y_top = counts_df["n_unique"].max()
    ax.text(proband_mid, y_top * 1.06, "Probands", ha="center", fontsize=4, fontweight="bold")
    ax.text(parent_mid, y_top * 1.06, "Parents", ha="center", fontsize=4, fontweight="bold")

    ax.set_xlabel("Sample")
    ax.set_ylabel("Trio-unique isoforms")
    ax.set_xticks([])
    ax.set_ylim(top=y_top * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")

    out = OUTPUT_DIR / "supplemental_2B_trio_unique_per_sample.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_1c() -> None:
    """
    Grouped dot/strip plot of three QC metrics per sample (ML-filtered).

    Metrics: multi-exonic rate, canonical splice site rate, RTS-free rate.
    One column per metric, individual sample dots colored by role.
    """
    setup_style()
    samples = get_long_read_sample_ids()

    records: list[dict] = []
    for sid in samples:
        df = read_sqanti3_annotated(sid, rules_filter=False)
        n = len(df)
        records.append({
            "sample_id": sid,
            "Multi-exonic": (df["exons"] >= 2).sum() / n * 100,
            "Canonical\nsplice sites": (df["all_canonical"] == "canonical").sum() / n * 100,
            "RTS-free": (df["RTS_stage"] != "True").sum() / n * 100,
        })
    qc_df = pd.DataFrame(records)
    qc_df["group"] = qc_df["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )

    metrics = ["Multi-exonic", "Canonical\nsplice sites", "RTS-free"]

    fig, ax = plt.subplots(figsize=(1.2, 1))
    rng = np.random.default_rng(42)

    group_offset = 0.18
    for i, metric in enumerate(metrics):
        for g, offset, color in [("Proband", -group_offset, VIR_A),
                                  ("Parent", group_offset, VIR_B)]:
            mask = qc_df["group"] == g
            vals = qc_df.loc[mask, metric].values
            jitter = rng.uniform(-0.07, 0.07, len(vals))
            for j, v in enumerate(vals):
                ax.scatter(i + offset + jitter[j], v, s=3, alpha=0.7, zorder=3,
                           color=color, edgecolors="white", linewidth=0.15)
            ax.plot(i + offset, vals.mean(), marker="_", color="black",
                    markersize=4, markeredgewidth=0.8, zorder=4)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(99.0, 100.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_A,
               markeredgecolor="white", markeredgewidth=0.2, markersize=2.5,
               label="Proband"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=VIR_B,
               markeredgecolor="white", markeredgewidth=0.2, markersize=2.5,
               label="Parent"),
        Line2D([0], [0], marker="_", color="black", markersize=3,
               markeredgewidth=0.6, label="Mean", linestyle="None"),
    ]
    ax.legend(handles=handles, loc="lower left")

    out = OUTPUT_DIR / "supplemental_1C_qc_metrics.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_1d() -> None:
    """
    Two-panel bar chart of median and mean unique reads per transcript.

    Left: per-sample median (robust centre).
    Right: per-sample mean (sensitive to high-coverage transcripts).
    Samples sorted Probands-first, descending within each group.
    """
    setup_style()
    samples = get_long_read_sample_ids()

    records: list[dict] = []
    for sid in samples:
        df = read_sqanti3_annotated(sid, rules_filter=False)
        df["interval"] = df.apply(lambda r: f"{r['chrom']}:{r['start']}-{r['end']}", axis=1)
        df["is_hemo"] = df["interval"].map(
            lambda x: any(intersect(x, h) for h in GENES_TO_REMOVE))
        df = df[~df["is_hemo"]]
        records.append({
            "sample_id": sid,
            "median_reads": df["uniq_reads"].median(),
            "mean_reads": df["uniq_reads"].mean(),
        })
    stats_df = pd.DataFrame(records)

    stats_df["group"] = stats_df["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )
    group_rank = {"Proband": 0, "Parent": 1}
    stats_df["group_rank"] = stats_df["group"].map(group_rank)

    panels = [
        ("median_reads", "Median unique reads\nper transcript"),
        ("mean_reads", "Mean unique reads\nper transcript"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(5, 1))

    for ax, (col, ylabel) in zip(axes, panels):
        sorted_df = stats_df.sort_values(
            ["group_rank", col], ascending=[True, False]
        ).reset_index(drop=True)

        n_probands = (sorted_df["group"] == "Proband").sum()
        colors = [VIR_A if g == "Proband" else VIR_B for g in sorted_df["group"]]

        x = np.arange(len(sorted_df))
        ax.bar(x, sorted_df[col], color=colors, edgecolor="white",
               linewidth=0.2, width=1.0)

        if "median" in col:
            ref = sorted_df[col].median()
            label = f"Median: {ref:.1f}"
        else:
            ref = sorted_df[col].mean()
            label = f"Mean: {ref:.1f}"
        ax.axhline(ref, color="#d62728", linestyle="--", linewidth=0.5, label=label)

        ax.axvline(n_probands - 0.5, color="black", linestyle="--", linewidth=0.4)

        proband_mid = (n_probands - 1) / 2
        parent_mid = n_probands + (len(sorted_df) - n_probands - 1) / 2
        y_top = sorted_df[col].max()
        ax.text(proband_mid, y_top * 1.06, "Probands",
                ha="center", fontsize=4, fontweight="bold")
        ax.text(parent_mid, y_top * 1.06, "Parents",
                ha="center", fontsize=4, fontweight="bold")

        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.set_ylim(top=y_top * 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right")

    fig.subplots_adjust(wspace=0.4)

    out = OUTPUT_DIR / "supplemental_2D_reads_per_transcript.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_2e() -> None:
    """
    Bar chart of trio-unique fusion transcript counts per sample.

    Samples sorted by group (Probands first, then Parents), descending
    within each group. A dashed median line is drawn across the panel.
    """
    setup_style()

    fusion_df = pd.read_csv(
        Path(DATA_DIR) / "unique_fusion_tx" / "fusion_characteristics_ml_filtered.csv"
    )
    per_sample = fusion_df.groupby("sample_id").size().reset_index(name="n_fusions")

    all_samples = get_long_read_sample_ids()
    counts_df = pd.DataFrame({"sample_id": all_samples})
    counts_df = counts_df.merge(per_sample, on="sample_id", how="left")
    counts_df["n_fusions"] = counts_df["n_fusions"].fillna(0).astype(int)

    counts_df["group"] = counts_df["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )
    group_rank = {"Proband": 0, "Parent": 1}
    counts_df["group_rank"] = counts_df["group"].map(group_rank)
    counts_df = counts_df.sort_values(
        ["group_rank", "n_fusions"], ascending=[True, False]
    ).reset_index(drop=True)

    n_probands = (counts_df["group"] == "Proband").sum()
    colors = [VIR_A if g == "Proband" else VIR_B for g in counts_df["group"]]

    fig, ax = plt.subplots(figsize=(2.5, 1))
    x = np.arange(len(counts_df))
    ax.bar(x, counts_df["n_fusions"], color=colors, edgecolor="white",
           linewidth=0.2, width=1.0)

    median_val = counts_df["n_fusions"].median()
    ax.axhline(median_val, color="#d62728", linestyle="--", linewidth=0.5,
               label=f"Median: {median_val:.0f}")

    ax.axvline(n_probands - 0.5, color="black", linestyle="--", linewidth=0.4)

    proband_mid = (n_probands - 1) / 2
    parent_mid = n_probands + (len(counts_df) - n_probands - 1) / 2
    y_top = counts_df["n_fusions"].max()
    ax.text(proband_mid, y_top * 1.06, "Probands",
            ha="center", fontsize=4, fontweight="bold")
    ax.text(parent_mid, y_top * 1.06, "Parents",
            ha="center", fontsize=4, fontweight="bold")

    ax.set_xlabel("Sample")
    ax.set_ylabel("Trio-unique fusion\ntranscripts")
    ax.set_xticks([])
    ax.set_ylim(top=y_top * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")

    out = OUTPUT_DIR / "supplemental_2E_fusion_per_sample.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_2f() -> None:
    """
    Paired slope plot of fusion transcript counts: proband vs parent mean.

    Each line connects a trio's parent mean to the proband count.
    Annotated with Wilcoxon p-value and proband-excess fraction.
    """
    setup_style()

    fusion_df = pd.read_csv(
        Path(DATA_DIR) / "unique_fusion_tx" / "fusion_characteristics_ml_filtered.csv"
    )
    member = fusion_df["sample_id"].str.split("_").str[2].astype(int)
    fusion_df["role"] = member.apply(lambda x: "Proband" if x == 3 else "Parent")
    fusion_df["trio_id"] = fusion_df["sample_id"].str.extract(r"(RGP_\d+)")

    sample_counts = (
        fusion_df.groupby(["sample_id", "role", "trio_id"])
        .size()
        .reset_index(name="fusion_count")
    )

    paired_data: list[dict] = []
    for trio in sample_counts["trio_id"].unique():
        trio_samples = sample_counts[sample_counts["trio_id"] == trio]
        proband_row = trio_samples[trio_samples["role"] == "Proband"]
        parent_rows = trio_samples[trio_samples["role"] == "Parent"]
        if len(proband_row) == 1 and len(parent_rows) >= 1:
            paired_data.append({
                "trio_id": trio,
                "proband": proband_row["fusion_count"].values[0],
                "parent_mean": parent_rows["fusion_count"].mean(),
            })
    paired_df = pd.DataFrame(paired_data)

    _, p_value = stats.wilcoxon(paired_df["proband"], paired_df["parent_mean"])
    n_excess = (paired_df["proband"] > paired_df["parent_mean"]).sum()

    fig, ax = plt.subplots(figsize=(1, 1))

    for _, row in paired_df.iterrows():
        color = VIR_A if row["proband"] > row["parent_mean"] else VIR_B
        ax.plot([0, 1], [row["parent_mean"], row["proband"]],
                color=color, alpha=0.5, linewidth=0.6)

    ax.scatter([0] * len(paired_df), paired_df["parent_mean"],
               c=VIR_B, s=6, zorder=5, edgecolor="white", linewidth=0.2)
    ax.scatter([1] * len(paired_df), paired_df["proband"],
               c=VIR_A, s=6, zorder=5, edgecolor="white", linewidth=0.2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Parent\nmean", "Proband"])
    ax.set_ylabel("Fusion transcripts")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.5, 0.97, f"p = {p_value:.2f}\n{n_excess}/{len(paired_df)} proband excess",
            transform=ax.transAxes, ha="center", va="top", fontsize=3.5)

    out = OUTPUT_DIR / "supplemental_2F_proband_vs_parent_fusion.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_supplemental_2h() -> None:
    """
    Venn diagram of genes involved in fusion transcripts: proband vs parent.
    """
    setup_style()
    from matplotlib_venn import venn2

    fusion_df = pd.read_csv(
        Path(DATA_DIR) / "unique_fusion_tx" / "fusion_characteristics_ml_filtered.csv"
    )
    member = fusion_df["sample_id"].str.split("_").str[2].astype(int)
    fusion_df["role"] = member.apply(lambda x: "Proband" if x == 3 else "Parent")

    proband_genes = (
        set(fusion_df.loc[fusion_df["role"] == "Proband", "gene1"])
        | set(fusion_df.loc[fusion_df["role"] == "Proband", "gene2"])
    )
    parent_genes = (
        set(fusion_df.loc[fusion_df["role"] == "Parent", "gene1"])
        | set(fusion_df.loc[fusion_df["role"] == "Parent", "gene2"])
    )

    fig, ax = plt.subplots(figsize=(1, 1))
    v = venn2(
        [proband_genes, parent_genes],
        set_labels=("Proband", "Parent"),
        set_colors=(VIR_A, VIR_B),
        alpha=0.6,
        ax=ax,
    )
    for text in v.set_labels:
        text.set_fontsize(4)
    for text in v.subset_labels:
        if text:
            text.set_fontsize(3.5)

    shared = len(proband_genes & parent_genes)
    total = len(proband_genes | parent_genes)
    ax.text(0.5, -0.05, f"{shared / total * 100:.1f}% shared",
            transform=ax.transAxes, ha="center", fontsize=3.5, color="gray")

    fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1)
    out = OUTPUT_DIR / "supplemental_2H_fusion_gene_venn.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def generate_supplemental_table_1() -> None:
    """
    Generate per-sample summary statistics table.
    """
    samples = get_long_read_sample_ids()

    summary = pd.read_csv(Path(DATA_DIR) / "summary_data_table.tsv")
    reads_df = load_reads_summary(READS_SUMMARY_PATH)
    meta = load_metadata()

    table = summary[["sample_id", "RIN", "total_readcount",
                      "total_transcripts_count", "unique_transcripts_count"]].copy()

    table["family_id"] = table["sample_id"].str.extract(r"(RGP_\d+)")
    table["role"] = table["sample_id"].apply(
        lambda s: "Proband" if s.endswith("_3_R1") else "Parent"
    )

    table = table.merge(
        reads_df[["sample_id", "flnc_reads", "mapping_rate"]],
        on="sample_id", how="left",
    )

    table = table.merge(
        meta[["sample_id", "Gender", "Affected_Status"]],
        on="sample_id", how="left",
    )

    qc_records: list[dict] = []
    for sid in samples:
        df = read_sqanti3_annotated(sid, rules_filter=False)
        n = len(df)
        qc_records.append({
            "sample_id": sid,
            "multi_exonic_pct": (df["exons"] >= 2).sum() / n * 100,
            "canonical_splice_pct": (df["all_canonical"] == "canonical").sum() / n * 100,
            "rts_free_pct": (df["RTS_stage"] != "True").sum() / n * 100,
        })
    qc_df = pd.DataFrame(qc_records)
    table = table.merge(qc_df, on="sample_id", how="left")

    table = table[[
        "sample_id", "family_id", "role", "Gender", "Affected_Status",
        "RIN", "flnc_reads", "mapping_rate",
        "total_transcripts_count", "unique_transcripts_count",
        "multi_exonic_pct", "canonical_splice_pct", "rts_free_pct",
    ]].rename(columns={
        "sample_id": "Sample ID",
        "family_id": "Family",
        "role": "Role",
        "Gender": "Sex",
        "Affected_Status": "Affected",
        "RIN": "RIN",
        "flnc_reads": "FLNC reads",
        "mapping_rate": "Mapping rate (%)",
        "total_transcripts_count": "Total isoforms",
        "unique_transcripts_count": "Trio-unique isoforms",
        "multi_exonic_pct": "Multi-exonic (%)",
        "canonical_splice_pct": "Canonical splice (%)",
        "rts_free_pct": "RTS-free (%)",
    })

    role_order = {"Proband": 0, "Parent": 1}
    table["_role_rank"] = table["Role"].map(role_order)
    table = table.sort_values(["Family", "_role_rank"]).drop(columns=["_role_rank"])

    table["Mapping rate (%)"] = table["Mapping rate (%)"].round(2)
    table["Multi-exonic (%)"] = table["Multi-exonic (%)"].round(2)
    table["Canonical splice (%)"] = table["Canonical splice (%)"].round(2)
    table["RTS-free (%)"] = table["RTS-free (%)"].round(2)

    out = OUTPUT_DIR / "supplemental_table_1.csv"
    table.to_csv(out, index=False)
    log.info("Saved %s", out)


def plot_supplemental_3a() -> None:
    """
    Donut chart of alt 5' splice junctions: shared vs novel.
    """
    setup_style()

    psi = pd.read_csv(
        Path(DATA_DIR) / "outrider" / "alt5_shift_analysis_bam" / "junction_psi_matrix.tsv",
        sep="\t", index_col=0,
    )
    n_observable = psi.notna().sum(axis=1)
    n_shared = int((n_observable >= 3).sum())
    n_novel = int((n_observable <= 2).sum())
    total = n_shared + n_novel

    fig, ax = plt.subplots(figsize=(1, 1))
    wedges, _ = ax.pie(
        [n_shared, n_novel],
        colors=[VIR_A, VIR_B],
        startangle=90,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 0.5},
    )

    ax.text(0, 0, f"{total:,}", ha="center", va="center", fontsize=4,
            fontweight="bold")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=VIR_A, label="Shared ({:,}, {:.1f}%)".format(
            n_shared, n_shared / total * 100)),
        Patch(facecolor=VIR_B, label="Novel ({:,}, {:.1f}%)".format(
            n_novel, n_novel / total * 100)),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.1),
              fontsize=2.5, handlelength=0.5, handleheight=0.5,
              borderpad=0.3, labelspacing=0.2, frameon=False)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.15)
    out = OUTPUT_DIR / "supplemental_3A_alt5_junction_donut.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def _get_expressed_genes_from_gcs(tpm_paths: pd.DataFrame) -> set[str]:
    """
    Download GCS TPM files and return union of genes with TPM >= 1.

    :param tpm_paths: DataFrame with sample_id and rnaseqc_gene_tpm columns.
    :return: Set of gene symbols with TPM >= 1 across all samples.
    """
    import gzip, io, re
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    all_expressed: set[str] = set()
    for _, row in tpm_paths.iterrows():
        gcs_path = row["rnaseqc_gene_tpm"]
        match = re.match(r"gs://([^/]+)/(.+)", gcs_path)
        bucket = client.bucket(match.group(1))
        blob = bucket.blob(match.group(2))
        data = blob.download_as_bytes()
        with gzip.open(io.BytesIO(data), "rt") as fh:
            df = pd.read_csv(fh, sep="\t", skiprows=2)
        expressed = df.loc[df.iloc[:, 2] >= 1, "Description"].dropna().unique()
        all_expressed.update(expressed)
        log.info("    %s: %d genes with TPM >= 1", row["sample_id"], len(expressed))
    return all_expressed


def _plot_da_genes_sr(all_expressed_genes: set[str], out_path: Path) -> None:
    """
    Plot stacked bar of DA genes by evidence level.

    :param all_expressed_genes: Set of expressed gene symbols.
    :param out_path: Path to save the output figure.
    """
    setup_style()
    da_genes_df = get_disease_associated_genes()
    da_gene_symbols = set(da_genes_df["gene_symbol"].dropna().unique())
    expressed_da = all_expressed_genes & da_gene_symbols

    expressed_da_df = da_genes_df[da_genes_df["gene_symbol"].isin(expressed_da)].copy()
    evidence_order = ["Definitive", "Strong", "Moderate"]
    evidence_counts = []
    for ev in evidence_order:
        cnt = expressed_da_df[
            expressed_da_df["CLINGEN_classification"].str.contains(ev, na=False)
        ].shape[0]
        evidence_counts.append(cnt)
    not_expressed = len(da_gene_symbols) - len(expressed_da)
    colors = [cm.viridis(v) for v in [0.1, 0.35, 0.6]]

    log.info("  DA genes: %d total, %d expressed, %d not expressed",
             len(da_gene_symbols), len(expressed_da), not_expressed)
    for ev, cnt in zip(evidence_order, evidence_counts):
        log.info("    %s: %d", ev, cnt)

    fig, ax = plt.subplots(figsize=(1, 1))
    bottom = 0
    for i, cnt in enumerate(evidence_counts):
        ax.bar(0, cnt, bottom=bottom, color=colors[i], edgecolor="black",
               linewidth=0.3, width=0.5)
        bottom += cnt
    ax.bar(1, not_expressed, color="#BBBBBB", edgecolor="black", linewidth=0.3, width=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Expressed", "Not\nexpressed"])
    ax.set_ylabel("Number of DA genes")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3,
                       label=f"{ev} (n={cnt:,})")
        for i, (ev, cnt) in enumerate(zip(evidence_order, evidence_counts))
    ]
    legend_handles.append(mpatches.Patch(facecolor="#BBBBBB", edgecolor="black",
                                         linewidth=0.3,
                                         label=f"Not expressed (n={not_expressed:,})"))
    ax.legend(handles=legend_handles, loc="center right",
              fontsize=2.5, handlelength=0.5, handleheight=0.5,
              borderpad=0.3, labelspacing=0.2)

    fig.subplots_adjust(left=0.28, right=0.98, top=0.95, bottom=0.18)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out_path)
        fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_fig2b_da_genes_combined_watchmaker() -> None:
    """
    Stacked bar: expressed DA genes in Watchmaker SR samples (TPM >= 1).
    """
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["watchmaker"] == "yes"]
    tpm_paths = dat[["sample_id", "rnaseqc_gene_tpm"]].dropna(subset=["rnaseqc_gene_tpm"])
    log.info("  %d Watchmaker samples with TPM paths", len(tpm_paths))

    all_expressed = _get_expressed_genes_from_gcs(tpm_paths)
    _plot_da_genes_sr(all_expressed, OUTPUT_DIR / "Fig2B_da_genes_combined_watchmaker.png")


def plot_fig2b_da_genes_combined_whole_blood() -> None:
    """
    Stacked bar: expressed DA genes in non-Watchmaker whole blood SR samples (TPM >= 1).
    """
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["imputed_tissue"] == "whole_blood"]
    dat = dat[~(dat["exclude"] == "yes")]
    dat = dat[~(dat["watchmaker"] == "yes")]
    dat = dat[~dat["sample_id"].isin(["MAN_2788-01_R1", "MAN_2788-02_R1"])]
    tpm_paths = dat[["sample_id", "rnaseqc_gene_tpm"]].dropna(subset=["rnaseqc_gene_tpm"])
    log.info("  %d whole blood (non-Watchmaker) samples with TPM paths", len(tpm_paths))

    all_expressed = _get_expressed_genes_from_gcs(tpm_paths)
    _plot_da_genes_sr(all_expressed, OUTPUT_DIR / "Fig2B_da_genes_combined_whole_blood.png")


def _compute_da_gene_counts(
    all_expressed_genes: set[str],
    evidence_levels: list[str] | None = None,
) -> dict:
    """
    Return evidence counts and not-expressed count for a set of expressed genes.

    :param all_expressed_genes: Set of expressed gene symbols.
    :param evidence_levels: Evidence levels to include. If None, uses all levels.
    :return: Dictionary with evidence counts, not-expressed count, and gene sets.
    """
    da_genes_df = get_disease_associated_genes()
    if evidence_levels is not None:
        da_genes_df = da_genes_df[
            da_genes_df["CLINGEN_classification"].apply(
                lambda x: all(
                    val.strip() in evidence_levels for val in str(x).split(";")
                )
            )
        ]
    da_gene_symbols = set(da_genes_df["gene_symbol"].dropna().unique())
    expressed_da = all_expressed_genes & da_gene_symbols
    expressed_da_df = da_genes_df[da_genes_df["gene_symbol"].isin(expressed_da)].copy()
    ev_order = evidence_levels if evidence_levels is not None else ["Definitive", "Strong", "Moderate", "Limited"]
    evidence_counts = []
    for ev in ev_order:
        cnt = expressed_da_df[
            expressed_da_df["CLINGEN_classification"].str.contains(ev, na=False)
        ].shape[0]
        evidence_counts.append(cnt)
    not_expressed = len(da_gene_symbols) - len(expressed_da)
    return {
        "evidence_counts": evidence_counts,
        "not_expressed": not_expressed,
        "total_da": len(da_gene_symbols),
        "expressed_da_symbols": expressed_da,
        "da_gene_symbols": da_gene_symbols,
    }


def _mcnemar_pvalue(set_a: set[str], set_b: set[str], universe: set[str]) -> float:
    """
    McNemar's test p-value for two sets of expressed genes over a shared universe.

    :param set_a: First set of expressed gene symbols.
    :param set_b: Second set of expressed gene symbols.
    :param universe: Universe of all gene symbols to consider.
    :return: McNemar's test p-value.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    b = len(set_a - set_b)
    c = len(set_b - set_a)
    a = len(set_a & set_b)
    d = len(universe - set_a - set_b)
    table = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=True)
    return result.pvalue


def _format_pval(p: float) -> str:
    """
    Format a p-value into significance stars.

    :param p: P-value to format.
    :return: Significance string ('***', '**', '*', or 'n.s.').
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def plot_fig2b_da_genes_combined_all(
    evidence_levels: list[str] | None = None,
    output_suffix: str = "",
) -> None:
    """
    Three-column stacked bar: long-read, Watchmaker SR, whole blood SR.

    :param evidence_levels: ClinGen evidence levels to include. If None, uses all.
    :param output_suffix: Suffix appended to output filename (e.g. "_dsm_only").
    """
    setup_style()
    ev_order = evidence_levels if evidence_levels is not None else [
        "Definitive", "Strong", "Moderate", "Limited",
    ]
    viridis_vals = np.linspace(0.1, 0.85, len(ev_order))

    samples = get_long_read_sample_ids()
    all_tx = get_all_tx(samples)
    all_tx = {sid: df[df["uniq_reads"] >= 5] for sid, df in all_tx.items()}
    lr_expressed_genes: set[str] = set()
    for df in all_tx.values():
        if "gene_name" in df.columns:
            lr_expressed_genes.update(df["gene_name"].dropna().unique())
    n_lr = len(samples)
    lr_counts = _compute_da_gene_counts(lr_expressed_genes, evidence_levels=evidence_levels)
    log.info("  Long-read: %d samples, %d/%d expressed",
             n_lr, len(lr_counts["expressed_da_symbols"]), lr_counts["total_da"])

    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    wm = dat[dat["watchmaker"] == "yes"]
    wm_paths = wm[["sample_id", "rnaseqc_gene_tpm"]].dropna(subset=["rnaseqc_gene_tpm"])
    n_wm = len(wm_paths)
    wm_expressed = _get_expressed_genes_from_gcs(wm_paths)
    wm_counts = _compute_da_gene_counts(wm_expressed, evidence_levels=evidence_levels)
    log.info("  Watchmaker: %d samples, %d/%d expressed",
             n_wm, len(wm_counts["expressed_da_symbols"]), wm_counts["total_da"])

    wb = dat[dat["imputed_tissue"] == "whole_blood"]
    wb = wb[~(wb["exclude"] == "yes")]
    wb = wb[~(wb["watchmaker"] == "yes")]
    wb = wb[~wb["sample_id"].isin(["MAN_2788-01_R1", "MAN_2788-02_R1"])]
    wb_paths = wb[["sample_id", "rnaseqc_gene_tpm"]].dropna(subset=["rnaseqc_gene_tpm"])
    n_wb = len(wb_paths)
    wb_expressed = _get_expressed_genes_from_gcs(wb_paths)
    wb_counts = _compute_da_gene_counts(wb_expressed, evidence_levels=evidence_levels)
    log.info("  Whole blood: %d samples, %d/%d expressed",
             n_wb, len(wb_counts["expressed_da_symbols"]), wb_counts["total_da"])

    da_universe = (lr_counts["da_gene_symbols"]
                   | wm_counts["da_gene_symbols"]
                   | wb_counts["da_gene_symbols"])
    pairs = [
        (0, 1, lr_counts["expressed_da_symbols"], wm_counts["expressed_da_symbols"]),
        (1, 2, wm_counts["expressed_da_symbols"], wb_counts["expressed_da_symbols"]),
        (0, 2, lr_counts["expressed_da_symbols"], wb_counts["expressed_da_symbols"]),
    ]
    pvalues = {}
    for i, j, sa, sb in pairs:
        p = _mcnemar_pvalue(sa, sb, da_universe)
        pvalues[(i, j)] = p
        log.info("  McNemar %d vs %d: p=%.2e (%s)", i, j, p, _format_pval(p))

    colors = [cm.viridis(v) for v in viridis_vals]
    grey = "#BBBBBB"

    datasets = [
        ("Long-read\n(LR)", n_lr, lr_counts["evidence_counts"], lr_counts["not_expressed"]),
        ("Whole blood\n(SR globin-\ndepletion)", n_wm, wm_counts["evidence_counts"], wm_counts["not_expressed"]),
        ("Whole blood\n(SR)", n_wb, wb_counts["evidence_counts"], wb_counts["not_expressed"]),
    ]

    fig, ax = plt.subplots(figsize=(1.4, 2.2))
    x_positions = np.array([0, 0.9, 1.8])
    bar_width = 0.45

    bar_tops = []
    for xi, (_, n_samp, ev_counts, not_expr) in zip(x_positions, datasets):
        bottom = 0
        for ci, cnt in enumerate(ev_counts):
            ax.bar(xi, cnt, bottom=bottom, color=colors[ci], edgecolor="black",
                   linewidth=0.3, width=bar_width)
            bottom += cnt
        ax.bar(xi, not_expr, bottom=bottom, color=grey, edgecolor="black",
               linewidth=0.3, width=bar_width)
        total_height = bottom + not_expr
        bar_tops.append(total_height)
        ax.text(xi, total_height + 15, f"n={n_samp}", ha="center",
                va="bottom", fontsize=3.5)

    max_top = max(bar_tops)
    bracket_y = max_top + 250
    bracket_gap = 500
    tick_h = 60
    ns_inset = 0.15
    bracket_order = [(1, 2), (0, 1), (0, 2)]
    for idx, (i, j) in enumerate(bracket_order):
        y = bracket_y + idx * bracket_gap
        x1, x2 = x_positions[i], x_positions[j]
        if (i, j) == (1, 2):
            x1 += ns_inset
            x2 -= ns_inset
            y += 200  # raise n.s. bracket above the sample labels
        ax.plot([x1, x1, x2, x2], [y - tick_h, y, y, y - tick_h],
                color="black", linewidth=0.4, clip_on=False)
        label = _format_pval(pvalues[(i, j)])
        if pvalues[(i, j)] < 0.05:
            label += f"\np={pvalues[(i, j)]:.1e}"
        ax.text((x1 + x2) / 2, y - 5, label,
                ha="center", va="top", fontsize=3, clip_on=False)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([d[0] for d in datasets])
    ax.set_xlim(-0.5, 2.4)
    ax.set_ylabel("Number of DA genes")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3,
                       label=ev_order[i])
        for i in range(len(ev_order))
    ]
    legend_handles.append(mpatches.Patch(facecolor=grey, edgecolor="black",
                                         linewidth=0.3, label="Not expressed"))
    ncol = 3 if len(ev_order) >= 3 else 2
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.35), ncol=ncol,
              fontsize=3, handlelength=0.6, handleheight=0.6,
              borderpad=0.3, labelspacing=0.2, columnspacing=0.5,
              frameon=False)

    fig.subplots_adjust(left=0.24, right=0.98, top=0.68, bottom=0.38)
    out = OUTPUT_DIR / f"Fig2B_da_genes_combined_all{output_suffix}.png"
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(out)
        fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


def plot_rqs_distribution() -> None:
    """
    Plot RQS distribution for Watchmaker samples as a histogram with KDE overlay.
    """
    setup_style()
    dat = read_from_airtable(RNA_SEQ_BASE_ID, DATA_PATHS_TABLE_ID, DATA_PATHS_VIEW_ID)
    dat = dat[dat["watchmaker"] == "yes"]
    rqs_values = dat["RQS"].dropna().astype(float).values
    log.info("  %d Watchmaker samples with RQS values", len(rqs_values))

    fig, ax = plt.subplots()

    ax.hist(rqs_values, bins=12, density=True, alpha=0.7, color=VIR_B,
            edgecolor="white", linewidth=0.3)

    kde = stats.gaussian_kde(rqs_values)
    x_kde = np.linspace(rqs_values.min() - 0.5, rqs_values.max() + 0.5, 200)
    ax.plot(x_kde, kde(x_kde), color=VIR_A)

    mean_rqs = rqs_values.mean()
    ax.axvline(mean_rqs, color="black", linestyle="--", linewidth=0.6,
               label=f"Mean: {mean_rqs:.1f}")

    ax.set_xlabel("RQS")
    ax.set_ylabel("Density")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    out = OUTPUT_DIR / "rqs_distribution.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    log.info("Saved %s", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate manuscript figures.")
    parser.add_argument(
        "--gtex-junctions", type=Path, required=True,
        help="Path to tabix-indexed GTEx junctions BED file.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Figure 1: Read length distribution ===")
    plot_read_length_distribution()

    log.info("=== Figure 2: Cumulative reads by top 10 genes ===")
    plot_cumulative_reads_by_gene()

    log.info("=== Figure 3: CDF by transcript length ===")
    plot_cdf_by_transcript_length()

    log.info("=== Figure 4: RIN distribution ===")
    plot_rin_distribution()

    log.info("=== Figure 5: Mapping rate ===")
    plot_mapping_rate()

    log.info("=== Figure 6: Transcript coverage ===")
    plot_transcript_coverage()

    log.info("=== Figure 7: Isoforms per gene ===")
    plot_isoforms_per_gene()

    log.info("=== Figure 8: Total transcripts violin ===")
    plot_all_tx_violin()

    log.info("=== Figure 9: Trio-unique transcripts violin ===")
    plot_unique_tx_violin()

    log.info("=== Figure 10: Combined violin by SQANTI3 category ===")
    plot_combined_tx_violin()

    log.info("=== Figure 11: Novel isoform enrichment ===")
    plot_supplemental_2c_novel_enrichment()

    log.info("=== Figure 12: DA genes expressed ===")
    plot_da_genes_expressed()

    log.info("=== Figure 13: DA genes by evidence level ===")
    plot_da_genes_by_evidence()

    log.info("=== Figure 14: DA genes combined ===")
    plot_fig2b_da_genes_combined()

    log.info("=== Figure 15: Unique fusion transcripts by distance ===")
    plot_unique_fusion_by_distance()

    log.info("=== Figure 16: Alt 5' shift vs novel (BAM) ===")
    plot_alt5_shift_vs_novel_bam()

    log.info("=== Figure 17: Alt 5' shift vs novel (SR) ===")
    plot_alt5_shift_vs_novel_sr()

    log.info("=== Figure 18: FRASER2 Jaccard ranked ===")
    plot_fraser_jaccard()

    log.info("=== Figure 19: FRASER1 PSI3 ranked ===")
    plot_fraser_psi3()

    log.info("=== Figures 20-23: Sashimi plots ===")
    plot_sashimi_all(args.gtex_junctions)

    log.info("=== Figure 24: Transcript isoforms per sample ===")
    plot_supplemental_1a()

    log.info("=== Figure 25: Trio-unique isoforms per sample ===")
    plot_supplemental_1b()

    log.info("=== Figure 26: Canonical splice site rate ===")
    plot_supplemental_1c()

    log.info("=== Figure 27: Reads per transcript ===")
    plot_supplemental_1d()

    log.info("=== Figure 28: Fusion transcripts per sample ===")
    plot_supplemental_2e()

    log.info("=== Figure 29: Proband vs parent fusion counts ===")
    plot_supplemental_2f()

    log.info("=== Figure 30: Fusion gene Venn diagram ===")
    plot_supplemental_2h()

    log.info("=== Figure 31: Alt 5' junction donut ===")
    plot_supplemental_3a()

    log.info("=== Figure 32: Alt 5' per-sample metrics ===")
    plot_supplemental_3b()

    log.info("=== Figure 33: Alt 5' shift vs novel (raw counts) ===")
    plot_supplemental_3c()

    log.info("=== Figure 34: Per-sample summary table ===")
    generate_supplemental_table_1()

    log.info("=== Figure 35: DA genes combined (Watchmaker) ===")
    plot_fig2b_da_genes_combined_watchmaker()

    log.info("=== Figure 36: DA genes combined (Whole blood) ===")
    plot_fig2b_da_genes_combined_whole_blood()

    log.info("=== Figure 37: DA genes combined (all three, all evidence) ===")
    plot_fig2b_da_genes_combined_all()

    log.info("=== Figure 38: DA genes combined (all three, DSM only) ===")
    plot_fig2b_da_genes_combined_all(
        evidence_levels=["Definitive", "Strong", "Moderate"],
        output_suffix="_dsm_only",
    )

    log.info("=== Figure 39: RQS distribution (Watchmaker) ===")
    plot_rqs_distribution()

    log.info("All figures saved to %s", OUTPUT_DIR)
