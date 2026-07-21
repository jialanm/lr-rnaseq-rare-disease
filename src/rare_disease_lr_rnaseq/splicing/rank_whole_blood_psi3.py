"""Rank samples by FRASER2 PSI3 splicing outlier counts in whole blood."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rare_disease_lr_rnaseq.utils import DATA_DIR

import logging

logger = logging.getLogger(__name__)

FILEPATH = f"{DATA_DIR}/filtered_p_value_0.3_deltapsi_0.1_whole_blood_psi3_105_samples_pdj_0.3_deltapsi_0.1_results.csv"
TARGET = {"<sample_id_1>", "<sample_id_2>"}

if __name__ == "__main__":
    psi3 = pd.read_csv(FILEPATH)
    psi3 = psi3[psi3["padjust"] <= 0.05]
    psi3 = psi3.value_counts("sampleID")
    logger.info(psi3.head())

    colors = ['orange' if str(sid) in TARGET else 'steelblue' for sid in psi3.index]

    plt.figure(figsize=(15, 6))
    bars = plt.bar(psi3.index.astype(str), psi3.values, color=colors)
    plt.xlabel("Sample ID")
    plt.ylabel("Count")
    ax = plt.gca()
    labels = psi3.index.astype(str).tolist()
    for i, label in enumerate(ax.get_xticklabels()):
        if i % 5 != 0 and labels[i] not in TARGET:
            label.set_visible(False)

    for i, sid in enumerate(labels):
        if sid in TARGET:
            ax.annotate(
                sid,
                xy=(i, psi3.values[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color="darkorange",
            )

    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.title("Alternative 5' splice site splicing outliers")
    plt.savefig(f"{DATA_DIR}/alternative_5_splice_site_splicing_outliers_FRASER1.png")
    plt.show()
    