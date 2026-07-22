"""Identify, annotate, and visualize fusion transcripts across long-read samples."""

import json
from typing import Any, Optional

import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pycirclize import Circos
import seaborn as sns
from rare_disease_lr_rnaseq.config import GENCODE_GTF_FILEPATH, MENDELIAN_GENE_DISEASE_TABLE_FILEPATH
from rare_disease_lr_rnaseq.utils import DATA_DIR, get_long_read_sample_ids, read_gtf, read_sqanti3_annotated

import logging

logger = logging.getLogger(__name__)
CLINGEN_EVIDENCE = {"Definitive", "Strong", "Moderate", "Limited"}

# Okabe-Ito colorblind-safe palette for scientific figures
OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']


def map_tx_to_intron_signature(transcript_id: str, gtf: pd.DataFrame) -> str:
    """
    Compute the intron signature for a transcript.

    The intron signature is a string representation of a tuple of
    (intron_start, intron_end) pairs derived from sorted exon boundaries.

    :param transcript_id: Transcript ID to compute the intron signature for.
    :param gtf: GTF DataFrame containing exon features with 'transcript_id', 'feature', 'start', and 'end' columns.
    :return: String representation of the intron signature tuple, or ``"mono_exon"`` for single-exon transcripts, or ``"no_exons"`` if no exons are found.
    """
    cur_gtf = gtf[gtf["transcript_id"] == transcript_id]
    exon_df = cur_gtf[cur_gtf["feature"] == "exon"].copy()

    if exon_df.empty:
        return "no_exons"

    exon_df = exon_df.sort_values("start").reset_index(drop=True)

    if len(exon_df) < 2:
        return "mono_exon"

    introns = []
    for i in range(len(exon_df) - 1):
        intron_start = exon_df.iloc[i]["end"]
        intron_end = exon_df.iloc[i + 1]["start"]
        introns.append((intron_start, intron_end))

    return str(tuple(introns))

# Human chromosome sizes (GRCh38)
CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415
}


def load_gene_coordinates() -> dict[str, tuple[str, int, int, str]]:
    """
    Load gene coordinates and names from the GENCODE GTF file.

    Parses the compressed GTF file and extracts chromosome, start, end,
    and gene name for each gene entry.

    :return: Dictionary mapping gene_id to a tuple of (chrom, start, end, gene_name).
    """
    import gzip
    gene_coords = {}
    with gzip.open(GENCODE_GTF_FILEPATH, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[2] != 'gene':
                continue
            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            attrs = fields[8]
            gene_id = None
            gene_name = None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_id'):
                    gene_id = attr.split('"')[1]
                elif attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
            if gene_id:
                gene_coords[gene_id] = (chrom, start, end, gene_name)
    return gene_coords


def get_disease_associated_genes() -> set[str]:
    """
    Load disease-associated gene symbols from the Mendelian gene disease table.

    Filters genes to those where all ClinGen classifications are among
    the accepted evidence levels (Definitive, Strong, Moderate, Limited).

    :return: Set of gene symbols with accepted ClinGen evidence.
    """
    mendelian_da_genes_df = pd.read_csv(
        MENDELIAN_GENE_DISEASE_TABLE_FILEPATH,
        sep='\t',
        usecols=["gene_symbol", "CLINGEN_classification"]
    )
    mendelian_da_genes = mendelian_da_genes_df[
        mendelian_da_genes_df["CLINGEN_classification"].apply(
            lambda x: all(val.strip() in CLINGEN_EVIDENCE for val in str(x).split(';'))
        )
    ]
    da_gene_symbols = set(mendelian_da_genes["gene_symbol"].tolist())
    logger.info(f"Loaded {len(da_gene_symbols)} disease-associated genes with ClinGen evidence.")
    return da_gene_symbols


def get_distance_category(
    chrom1: str, start1: int, end1: int, chrom2: str, start2: int, end2: int
) -> tuple[Optional[int], str, bool]:
    """
    Calculate fusion distance and assign a distance category.

    Distance is computed as ``min(|end1 - start2|, |start1 - end2|)``.
    Interchromosomal fusions return ``None`` for distance.

    :param chrom1: Chromosome of the first gene.
    :param start1: Start coordinate of the first gene.
    :param end1: End coordinate of the first gene.
    :param chrom2: Chromosome of the second gene.
    :param start2: Start coordinate of the second gene.
    :param end2: End coordinate of the second gene.
    :return: A tuple of (distance, category, is_interchromosomal) where category is one of ``"2 kb"``, ``"20 kb"``, ``">20 kb"``, or ``"interchromosomal"``.
    """
    if chrom1 != chrom2:
        return None, "interchromosomal", True

    distance = min(abs(end1 - start2), abs(start1 - end2))

    if distance < 2000:
        category = "2 kb"
    elif distance < 20000:
        category = "20 kb"
    else:
        category = ">20 kb"

    return distance, category, False


# Color scheme for distance categories (Okabe-Ito colorblind-safe)
DISTANCE_COLORS = {
    "2 kb": "#009E73",           # Green
    "20 kb": "#56B4E9",          # Sky blue
    ">20 kb": "#0072B2",         # Blue
    "interchromosomal": "#D55E00"  # Vermillion
}


def plot_fusion_circos(
    sample_id: str,
    unique_fusion_df: pd.DataFrame,
    gene_coords: dict[str, tuple[str, int, int, str]],
    output_dir: Optional[str] = None,
    de_novo_signatures: Optional[set[str]] = None,
    da_gene_symbols: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    Plot a circos diagram showing fusion transcripts for a sample.

    Each curve connects the midpoints of two fused genes. Colors indicate
    distance category; dashed lines indicate multi-gene fusions. Fusions
    with >10 reads and >20 kb distance are highlighted with a glow effect.
    De novo fusions are marked with star markers.

    :param sample_id: Sample ID for the plot title and filename.
    :param unique_fusion_df: DataFrame containing unique fusion transcripts with columns 'associated_gene', 'isoform', 'uniq_reads', 'intron_signature', 'chrom', 'start', and 'end'.
    :param gene_coords: Dictionary mapping gene_id to (chrom, start, end, gene_name).
    :param output_dir: Output directory for the plot. Defaults to the unique_fusion_tx data directory.
    :param de_novo_signatures: Set of intron signatures that are de novo for this sample.
    :param da_gene_symbols: Set of disease-associated gene symbols.
    :return: List of fusion characteristics dictionaries for this sample.
    """
    if output_dir is None:
        output_dir = f"{DATA_DIR}/unique_fusion_tx"

    if de_novo_signatures is None:
        de_novo_signatures = set()

    if da_gene_symbols is None:
        da_gene_symbols = set()

    fusion_characteristics = []

    if unique_fusion_df.empty:
        logger.info(f"No unique fusion transcripts for sample {sample_id}, skipping circos plot.")
        return fusion_characteristics

    links = []

    for _, row in unique_fusion_df.iterrows():
        associated_gene = row['associated_gene']
        isoform = row['isoform']
        uniq_reads = row['uniq_reads']
        intron_signature = row['intron_signature']
        predicted_nmd = row.get('predicted_NMD', None)
        isoform_chrom = row['chrom']
        isoform_start = row['start']
        isoform_end = row['end']
        isoform_interval = f"{isoform_chrom}:{isoform_start}-{isoform_end}"

        gene_ids = associated_gene.split('_')

        if len(gene_ids) < 2:
            continue

        first_gene = gene_ids[0]
        last_gene = gene_ids[-1]
        more_than_2_genes = len(gene_ids) > 2

        is_de_novo = intron_signature in de_novo_signatures

        if first_gene not in gene_coords or last_gene not in gene_coords:
            logger.info(f"Warning: Could not find coordinates for genes {first_gene} or {last_gene}")
            continue

        chrom1, start1, end1, gene1_name = gene_coords[first_gene]
        chrom2, start2, end2, gene2_name = gene_coords[last_gene]

        distance, category, is_interchromosomal = get_distance_category(
            chrom1, start1, end1, chrom2, start2, end2
        )

        is_highlighted = (uniq_reads > 10) and (category == ">20 kb")

        mid1 = (start1 + end1) // 2
        mid2 = (start2 + end2) // 2

        links.append((chrom1, mid1, chrom2, mid2, more_than_2_genes, category, is_highlighted, is_de_novo))

        gene1_is_in_da = gene1_name in da_gene_symbols
        gene2_is_in_da = gene2_name in da_gene_symbols

        fusion_characteristics.append({
            "sample_id": sample_id,
            "isoform": isoform,
            "isoform_interval": isoform_interval,
            "uniq_reads": uniq_reads,
            "gene1": first_gene,
            "gene1_name": gene1_name,
            "chr1": chrom1,
            "start1": start1,
            "end1": end1,
            "gene2": last_gene,
            "gene2_name": gene2_name,
            "chr2": chrom2,
            "start2": start2,
            "end2": end2,
            "fusion_distance": distance if distance is not None else "NA",
            "interchromosomal": is_interchromosomal,
            "distance_category": category,
            "more_than_2_genes": more_than_2_genes,
            "is_de_novo": is_de_novo,
            "predicted_NMD": predicted_nmd,
            "gene1_is_in_DA": gene1_is_in_da,
            "gene2_is_in_DA": gene2_is_in_da
        })

    if not links:
        logger.info(f"No valid fusion links for sample {sample_id}, skipping circos plot.")
        return fusion_characteristics

    chroms_involved = set()
    for chrom1, _, chrom2, _, _, _, _, _ in links:
        chroms_involved.add(chrom1)
        chroms_involved.add(chrom2)

    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chroms_to_plot = [c for c in chrom_order if c in chroms_involved]

    sectors = {chrom: CHROM_SIZES[chrom] for chrom in chroms_to_plot}

    circos = Circos(sectors, space=3)

    for sector in circos.sectors:
        track = sector.add_track((95, 100))
        track.axis(fc="lightgrey")
        track.text(sector.name.replace("chr", ""), fontsize=8, r=105)

        track2 = sector.add_track((90, 95))
        track2.axis(fc="lightblue", ec="black", lw=0.5)

    de_novo_markers = []

    for chrom1, pos1, chrom2, pos2, more_than_2_genes, category, is_highlighted, is_de_novo in links:
        color = DISTANCE_COLORS[category]
        linestyle = "dashed" if more_than_2_genes else "solid"

        if is_highlighted:
            lw = 4.0
        elif category == ">20 kb":
            lw = 2.5
        else:
            lw = 1.5

        if is_de_novo:
            de_novo_markers.append((chrom1, pos1, chrom2, pos2))

        if is_highlighted:
            circos.link(
                (chrom1, pos1, pos1),
                (chrom2, pos2, pos2),
                color=color,
                alpha=0.3,
                lw=8.0,
                ls=linestyle
            )

        circos.link(
            (chrom1, pos1, pos1),
            (chrom2, pos2, pos2),
            color=color,
            alpha=0.8 if is_highlighted else 0.7,
            lw=lw,
            ls=linestyle
        )

    fig = circos.plotfig()
    ax = fig.axes[0]

    for chrom1, pos1, chrom2, pos2 in de_novo_markers:
        sector1 = circos.get_sector(chrom1)
        sector2 = circos.get_sector(chrom2)

        radius = 85

        theta1 = sector1.x_to_rad(pos1)
        theta2 = sector2.x_to_rad(pos2)

        ax.plot(theta1, radius, marker='*', markersize=12, color='#FFD700',
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        ax.plot(theta2, radius, marker='*', markersize=12, color='#FFD700',
                markeredgecolor='black', markeredgewidth=0.5, zorder=10)

    fig.suptitle(f"Fusion Transcripts - {sample_id}", fontsize=12, y=0.98)

    legend_elements = [
        Patch(facecolor=DISTANCE_COLORS["2 kb"], alpha=0.7, label='< 2 kb'),
        Patch(facecolor=DISTANCE_COLORS["20 kb"], alpha=0.7, label='2-20 kb'),
        Patch(facecolor=DISTANCE_COLORS[">20 kb"], alpha=0.7, label='> 20 kb'),
        Patch(facecolor=DISTANCE_COLORS["interchromosomal"], alpha=0.7, label='Interchromosomal'),
        Line2D([0], [0], color='gray', linestyle='dashed', lw=1.5, label='>2 genes'),
        Line2D([0], [0], color=DISTANCE_COLORS[">20 kb"], lw=4, alpha=0.8, label='>20kb & >10 reads'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markersize=10,
               markeredgecolor='black', markeredgewidth=0.5, label='De novo')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=7)

    output_path = f"{output_dir}/{sample_id}_fusion_circos.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved circos plot to {output_path}")

    return fusion_characteristics


def get_de_novo_fusion_signatures(
    sample_ids: list[str], rules_filter: bool = True
) -> dict[str, set[str]]:
    """
    Identify de novo fusion intron signatures per sample.

    A fusion is de novo if its intron signature is present in the proband
    (_3_R1) but absent from both parents (_1_R1 and _2_R1). Parents are
    assigned empty sets by definition.

    :param sample_ids: List of all sample IDs.
    :param rules_filter: If True, use rules-filtered transcripts; otherwise use ML-filtered transcripts.
    :return: Dictionary mapping sample_id to the set of de novo intron signatures.
    """
    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    suffix = "" if rules_filter else "_ml_filtered"

    trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])

    de_novo_signatures = {}

    for cur_trio in trio_ids:
        parent1_id = f"{cur_trio}_1_R1"
        parent2_id = f"{cur_trio}_2_R1"
        proband_id = f"{cur_trio}_3_R1"

        parent_signatures = set()
        proband_signatures = set()

        for sample_id in [parent1_id, parent2_id]:
            csv_path = f"{output_dir}/{sample_id}{suffix}_unique_fusion_transcripts.csv" if suffix else f"{output_dir}/{sample_id}_unique_fusion_transcripts.csv"
            if suffix:
                csv_path = f"{output_dir}/{sample_id}_ml_filtered_unique_fusion_transcripts.csv"
            else:
                csv_path = f"{output_dir}/{sample_id}_unique_fusion_transcripts.csv"
            try:
                df = pd.read_csv(csv_path)
                parent_signatures.update(df['intron_signature'].tolist())
            except FileNotFoundError:
                logger.info(f"Warning: Could not find {csv_path}")

        if rules_filter:
            proband_csv = f"{output_dir}/{proband_id}_unique_fusion_transcripts.csv"
        else:
            proband_csv = f"{output_dir}/{proband_id}_ml_filtered_unique_fusion_transcripts.csv"

        try:
            proband_df = pd.read_csv(proband_csv)
            proband_signatures = set(proband_df['intron_signature'].tolist())
        except FileNotFoundError:
            logger.info(f"Warning: Could not find {proband_csv}")
            proband_signatures = set()

        de_novo = proband_signatures - parent_signatures
        de_novo_signatures[proband_id] = de_novo

        # Parents have no de novo fusions by definition
        de_novo_signatures[parent1_id] = set()
        de_novo_signatures[parent2_id] = set()

        if de_novo:
            logger.info(f"Trio {cur_trio}: Found {len(de_novo)} de novo fusion(s) in proband")

    return de_novo_signatures


def generate_all_fusion_circos_plots(rules_filter: bool = True) -> None:
    """
    Generate circos plots for all samples and save a combined table.

    Loads gene coordinates, identifies de novo fusions, then generates
    a circos plot per sample and saves a combined CSV of all fusion
    characteristics.

    :param rules_filter: If True, use rules-filtered transcripts; otherwise use ML-filtered transcripts.
    """
    sample_ids = get_long_read_sample_ids()
    output_dir = f"{DATA_DIR}/unique_fusion_tx"

    logger.info("Loading gene coordinates from gencode GTF...")
    gene_coords = load_gene_coordinates()
    logger.info(f"Loaded coordinates for {len(gene_coords)} genes.")

    logger.info("Loading disease-associated genes...")
    da_gene_symbols = get_disease_associated_genes()

    logger.info("Identifying de novo fusion transcripts...")
    de_novo_signatures = get_de_novo_fusion_signatures(sample_ids, rules_filter)

    all_fusion_characteristics = []

    for sample_id in sample_ids:
        logger.info(f"Processing sample {sample_id}...")

        if rules_filter:
            csv_path = f"{DATA_DIR}/unique_fusion_tx/{sample_id}_unique_fusion_transcripts.csv"
        else:
            csv_path = f"{DATA_DIR}/unique_fusion_tx/{sample_id}_ml_filtered_unique_fusion_transcripts.csv"

        try:
            unique_fusion_df = pd.read_csv(csv_path)
            sample_de_novo = de_novo_signatures.get(sample_id, set())
            sample_characteristics = plot_fusion_circos(
                sample_id, unique_fusion_df, gene_coords, output_dir, sample_de_novo, da_gene_symbols
            )
            all_fusion_characteristics.extend(sample_characteristics)
        except FileNotFoundError:
            logger.info(f"Warning: Could not find {csv_path}, skipping.")

    if all_fusion_characteristics:
        characteristics_df = pd.DataFrame(all_fusion_characteristics)
        suffix = "" if rules_filter else "_ml_filtered"
        output_path = f"{output_dir}/fusion_characteristics{suffix}.csv"
        characteristics_df.to_csv(output_path, index=False)
        logger.info(f"Saved fusion characteristics table to {output_path}")


def plot_de_novo_circos_summary(rules_filter: bool = False) -> Optional[pd.DataFrame]:
    """
    Generate a simplified circos plot showing only de novo fusions in probands.

    Colors indicate disease-associated gene involvement: gold/orange for
    fusions involving at least one DA gene, gray for no DA gene
    involvement. Saves both PNG and PDF outputs.

    :param rules_filter: If True, use rules-filtered data; otherwise use ML-filtered data.
    :return: DataFrame of de novo fusions in probands, or None if no data is available.
    """
    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    suffix = "" if rules_filter else "_ml_filtered"

    fusion_path = f"{output_dir}/fusion_characteristics{suffix}.csv"
    try:
        fusion_df = pd.read_csv(fusion_path)
    except FileNotFoundError:
        logger.info(f"Error: Could not find {fusion_path}")
        return

    logger.info("Loading gene coordinates...")
    gene_coords = load_gene_coordinates()

    proband_fusions = fusion_df[fusion_df['sample_id'].str.endswith('_3_R1')].copy()
    de_novo_fusions = proband_fusions[proband_fusions['is_de_novo'] == True].copy()

    if de_novo_fusions.empty:
        logger.info("No de novo fusions found in probands")
        return

    logger.info(f"Found {len(de_novo_fusions)} de novo fusions across all probands")

    chroms_involved = set()
    for _, row in de_novo_fusions.iterrows():
        chroms_involved.add(row['chr1'])
        chroms_involved.add(row['chr2'])

    chrom_order = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    chroms_to_plot = [c for c in chrom_order if c in chroms_involved]

    sectors = {chrom: CHROM_SIZES[chrom] for chrom in chroms_to_plot}

    circos = Circos(sectors, space=3)

    for sector in circos.sectors:
        track = sector.add_track((95, 100))
        track.axis(fc="#E8E8E8")
        track.text(sector.name.replace("chr", ""), fontsize=9, r=107)

        track2 = sector.add_track((90, 95))
        track2.axis(fc="#B8D4E8", ec="none")

    da_color = '#E69F00'
    no_da_color = '#999999'

    da_links = []
    no_da_links = []

    for _, row in de_novo_fusions.iterrows():
        chrom1, chrom2 = row['chr1'], row['chr2']
        mid1 = (row['start1'] + row['end1']) // 2
        mid2 = (row['start2'] + row['end2']) // 2

        has_da = row['gene1_is_in_DA'] or row['gene2_is_in_DA']

        if has_da:
            da_links.append((chrom1, mid1, chrom2, mid2, row))
        else:
            no_da_links.append((chrom1, mid1, chrom2, mid2, row))

    for chrom1, mid1, chrom2, mid2, row in no_da_links:
        circos.link(
            (chrom1, mid1, mid1),
            (chrom2, mid2, mid2),
            color=no_da_color,
            alpha=0.4,
            lw=1.5
        )

    for chrom1, mid1, chrom2, mid2, row in da_links:
        circos.link(
            (chrom1, mid1, mid1),
            (chrom2, mid2, mid2),
            color=da_color,
            alpha=0.8,
            lw=2.5
        )

    fig = circos.plotfig()

    fig.suptitle('De Novo Fusion Transcripts in Probands', fontsize=12, y=0.98)

    legend_elements = [
        Patch(facecolor=da_color, alpha=0.8, edgecolor='black', linewidth=0.5,
              label=f'DA gene involved (n={len(da_links)})'),
        Patch(facecolor=no_da_color, alpha=0.5, edgecolor='black', linewidth=0.5,
              label=f'No DA gene (n={len(no_da_links)})'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

    output_path = f"{output_dir}/de_novo_circos_summary{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved de novo circos summary to {output_path}")

    pdf_path = f"{output_dir}/de_novo_circos_summary{suffix}.pdf"
    fig = circos.plotfig()
    fig.suptitle('De Novo Fusion Transcripts in Probands', fontsize=12, y=0.98)
    fig.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved PDF to {pdf_path}")

    logger.info(f"\nDe novo fusion summary:")
    logger.info(f"  Total de novo fusions: {len(de_novo_fusions)}")
    logger.info(f"  With DA gene: {len(da_links)} ({len(da_links)/len(de_novo_fusions)*100:.1f}%)")
    logger.info(f"  Without DA gene: {len(no_da_links)} ({len(no_da_links)/len(de_novo_fusions)*100:.1f}%)")

    de_novo_fusions.to_csv(f"{output_dir}/de_novo_fusions_probands{suffix}.csv", index=False)

    return de_novo_fusions


def filter_non_unique_gene_pairings(
    unique_fusion_tx: dict[str, pd.DataFrame], sample_ids: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Filter out fusion transcripts with non-unique gene pairings.

    A gene pairing is the ordered tuple of all genes in the fusion. A
    pairing is unique if it appears in at most one trio across all samples.

    :param unique_fusion_tx: Dictionary mapping sample_id to DataFrame of unique fusion transcripts, each with an 'associated_gene' column.
    :param sample_ids: List of all sample IDs used to derive trio groupings.
    :return: Dictionary mapping sample_id to filtered DataFrame with non-unique gene pairings removed.
    """
    trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])

    gene_pairing_freq = defaultdict(int)

    for cur_trio in trio_ids:
        cur_trio_gene_pairings = set()
        sample_ids_in_trio = [f"{cur_trio}_{count}_R1" for count in range(1, 4)]

        for cur_sample_in_trio in sample_ids_in_trio:
            if cur_sample_in_trio not in unique_fusion_tx:
                continue
            cur_fusion_df = unique_fusion_tx[cur_sample_in_trio]

            for _, row in cur_fusion_df.iterrows():
                associated_gene = row['associated_gene']
                gene_tuple = tuple(associated_gene.split('_'))
                cur_trio_gene_pairings.add(gene_tuple)

        for gene_pairing in cur_trio_gene_pairings:
            gene_pairing_freq[gene_pairing] += 1

    unique_gene_pairings = set(
        pairing for pairing, freq in gene_pairing_freq.items() if freq == 1
    )

    total_pairings = len(gene_pairing_freq)
    non_unique_count = total_pairings - len(unique_gene_pairings)
    logger.info(f"Gene pairing uniqueness: {len(unique_gene_pairings)}/{total_pairings} pairings are unique across trios")
    logger.info(f"  Removing {non_unique_count} non-unique gene pairings")

    filtered_fusion_tx = {}
    for sample_id, fusion_df in unique_fusion_tx.items():
        fusion_df = fusion_df.copy()
        fusion_df['gene_pairing'] = fusion_df['associated_gene'].apply(
            lambda x: tuple(x.split('_'))
        )
        fusion_df['unique_gene_pairing'] = fusion_df['gene_pairing'].isin(unique_gene_pairings)

        filtered_df = fusion_df[fusion_df['unique_gene_pairing']].drop(
            columns=['gene_pairing', 'unique_gene_pairing']
        )
        filtered_fusion_tx[sample_id] = filtered_df

        removed_count = len(fusion_df) - len(filtered_df)
        if removed_count > 0:
            logger.info(f"  {sample_id}: Removed {removed_count} fusion transcripts with non-unique gene pairings")

    return filtered_fusion_tx


def get_fusion_tx(sample_id: str, rules_filter: bool = True) -> pd.DataFrame:
    """
    Get fusion transcripts from an annotated transcripts file.

    Reads the SQANTI3-annotated transcript table for the given sample
    and filters to rows with structural_category equal to 'fusion'.

    :param sample_id: Sample identifier.
    :param rules_filter: If True, use rules-filtered transcripts; otherwise use ML-filtered transcripts.
    :return: DataFrame containing only fusion transcripts.
    """
    if rules_filter:
        logger.info("Using rules filtered transcripts")
        cur_annotated_df = read_sqanti3_annotated(sample_id)
    else:
        logger.info("Using ML filtered transcripts")
        cur_annotated_df = read_sqanti3_annotated(sample_id, rules_filter=False)
    cur_fusion_df = cur_annotated_df[
        cur_annotated_df["structural_category"] == "fusion"]
    return cur_fusion_df


def get_unique_fusion_tx(
    sample_ids: list[str], rule_filter: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Get trio-unique fusion transcripts for each sample.

    A fusion is excluded if either its intron signature or its gene tuple
    appears in more than one trio. Mono-exon and no-exon fusion
    transcripts are also excluded. Duplicate intron signatures within a
    sample are deduplicated, keeping the one with highest read support.

    :param sample_ids: List of all sample IDs.
    :param rule_filter: If True, use rules-filtered transcripts; otherwise use ML-filtered transcripts.
    :return: Dictionary mapping sample_id to DataFrame of unique fusion transcripts.
    """
    trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])

    # Single pass: read all data and compute intron signatures once
    sample_fusion_data = {}  # Cache: sample_id -> fusion_df with intron_signature

    for cur_sample in sample_ids:
        logger.info(f"Loading {cur_sample}...")
        cur_fusion_df = get_fusion_tx(cur_sample, rule_filter)
        cur_gtf = read_gtf(cur_sample)
        cur_fusion_df = cur_fusion_df.copy()
        cur_fusion_df["intron_signature"] = cur_fusion_df.apply(
            lambda row: map_tx_to_intron_signature(row["isoform"], cur_gtf),
            axis=1
        )

        mono_exon_count = (cur_fusion_df["intron_signature"] == "mono_exon").sum()
        if mono_exon_count > 0:
            logger.info(f"  Excluding {mono_exon_count} mono-exon fusion transcripts from {cur_sample}")
        cur_fusion_df = cur_fusion_df[cur_fusion_df["intron_signature"] != "mono_exon"]
        cur_fusion_df = cur_fusion_df[cur_fusion_df["intron_signature"] != "no_exons"]

        cur_fusion_df["gene_tuple"] = cur_fusion_df["associated_gene"].apply(
            lambda x: tuple(x.split("_"))
        )

        dup_count = cur_fusion_df["intron_signature"].duplicated().sum()
        if dup_count > 0:
            logger.info(f"  Removing {dup_count} duplicate intron signatures from {cur_sample}")
            cur_fusion_df = cur_fusion_df.sort_values("uniq_reads", ascending=False)
            cur_fusion_df = cur_fusion_df.drop_duplicates(subset=["intron_signature"], keep="first")

        sample_fusion_data[cur_sample] = cur_fusion_df

    # Count intron signature and gene tuple frequency across trios (not samples)
    intron_sig_freq = defaultdict(int)
    gene_tuple_freq = defaultdict(int)

    for cur_trio in trio_ids:
        logger.info(f"Processing trio {cur_trio}")
        cur_trio_intron_sigs = set()
        cur_trio_gene_tuples = set()

        sample_ids_in_trio = [f"{cur_trio}_{count}_R1" for count in range(1, 4)]
        for cur_sample_in_trio in sample_ids_in_trio:
            cur_fusion_df = sample_fusion_data[cur_sample_in_trio]
            cur_trio_intron_sigs.update(set(cur_fusion_df["intron_signature"]))
            cur_trio_gene_tuples.update(set(cur_fusion_df["gene_tuple"]))

        for intron_sig in cur_trio_intron_sigs:
            intron_sig_freq[intron_sig] += 1
        for gene_tuple in cur_trio_gene_tuples:
            gene_tuple_freq[gene_tuple] += 1

    unique_intron_sigs = set(
        sig for sig, freq in intron_sig_freq.items() if freq == 1
    )
    unique_gene_tuples = set(
        gt for gt, freq in gene_tuple_freq.items() if freq == 1
    )

    non_unique_intron = len(intron_sig_freq) - len(unique_intron_sigs)
    non_unique_gene = len(gene_tuple_freq) - len(unique_gene_tuples)
    logger.info(f"\nUniqueness filtering (union/OR logic):")
    logger.info(f"  Intron signatures: {len(unique_intron_sigs)} unique, {non_unique_intron} non-unique")
    logger.info(f"  Gene tuples: {len(unique_gene_tuples)} unique, {non_unique_gene} non-unique")

    # Filter: keep only fusions where BOTH intron signature AND gene tuple are unique
    # (i.e., remove if EITHER is non-unique - union/OR logic for exclusion)
    unique_fusion_tx = {}
    for cur_sample in sample_ids:
        cur_fusion_df = sample_fusion_data[cur_sample]

        # Both must be unique (OR logic for exclusion = AND logic for inclusion)
        intron_unique = cur_fusion_df["intron_signature"].isin(unique_intron_sigs)
        gene_unique = cur_fusion_df["gene_tuple"].isin(unique_gene_tuples)
        cur_fusion_df["unique_fusion_tx"] = intron_unique & gene_unique

        unique_fusion_tx[cur_sample] = cur_fusion_df[cur_fusion_df["unique_fusion_tx"]].copy()

    return unique_fusion_tx


def plot_uniq_fusion_tx(
    rules_filtered_dict: dict[str, int], ml_filtered_dict: dict[str, int]
) -> None:
    """
    Plot boxplots comparing unique fusion transcript counts per trio.

    Displays side-by-side boxplots for rules-filtered and ML-filtered
    unique fusion transcript counts.

    :param rules_filtered_dict: Dictionary mapping trio ID to unique fusion transcript count using rules-filtered data.
    :param ml_filtered_dict: Dictionary mapping trio ID to unique fusion transcript count using ML-filtered data.
    """
    rules_filtered_counts = [v for v in rules_filtered_dict.values()]
    ml_filtered_counts = [v for v in ml_filtered_dict.values()]
    labels = ['Rules Filtered', 'ML Filtered']
    x = range(1, len(labels)+1)

    data = [rules_filtered_counts, ml_filtered_counts]
    labels = ['Rules Filtered', 'ML Filtered']

    plt.figure(figsize=(8, 10))
    bp = plt.boxplot(data, labels=labels, patch_artist=True,
                     widths=0.6, showmeans=True)

    colors = ['#6ea357', '#d9ead3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks(x, labels)
    plt.ylabel('Average Number of Unique Fusion Transcripts')
    plt.title('Unique Fusion Transcripts Per Trio')
    plt.savefig(f"{DATA_DIR}/unique_fusion_tx/unique_fusion_tx_comparison.png")
    plt.show()


def compare_unique_fusion_tx() -> tuple[dict[str, int], dict[str, int]]:
    """
    Compare unique fusion transcript counts between filtering methods.

    For each trio, counts the number of unique fusion intron signatures
    under both rules-filtered and ML-filtered approaches, logs the
    comparison, and generates a boxplot.

    :return: A tuple of (rules_filtered_counts, ml_filtered_counts) where each is a dictionary mapping trio ID to unique fusion count.
    """
    sample_ids = get_long_read_sample_ids()
    trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])
    unique_fusion_tx_per_trio = defaultdict(int)
    unique_ml_filtered_fusion_tx_per_trio = defaultdict(int)
    for cur_trio in trio_ids:
        sample_ids_in_trio = [f"{cur_trio}_{count}_R1" for count in range(1, 4)]
        unique_fusion_tx_set = set()
        unique_ml_filtered_fusion_tx_set = set()
        for cur_sample_in_trio in sample_ids_in_trio:
            cur_unique_fusion_df = pd.read_csv(
                f"{DATA_DIR}/unique_fusion_tx/"
                f"{cur_sample_in_trio}_unique_fusion_transcripts.csv")
            cur_unique_ml_filtered_fusion_df = pd.read_csv(
                f"{DATA_DIR}/unique_fusion_tx/"
                f"{cur_sample_in_trio}_ml_filtered_unique_fusion_transcripts.csv")
            unique_fusion_tx_set.update(set(cur_unique_fusion_df["intron_signature"]))
            unique_ml_filtered_fusion_tx_set.update(
                set(cur_unique_ml_filtered_fusion_df["intron_signature"]))

        unique_fusion_tx_per_trio[cur_trio] = len(unique_fusion_tx_set)
        unique_ml_filtered_fusion_tx_per_trio[cur_trio] = len(
            unique_ml_filtered_fusion_tx_set)

    for key, value in unique_ml_filtered_fusion_tx_per_trio.items():
        logger.info(
            f"Trio {key} has {unique_fusion_tx_per_trio[key]} unique fusion transcripts.")
        logger.info(f"Trio {key} has {value} unique fusion transcripts (ml filtered).")

    plot_uniq_fusion_tx(unique_fusion_tx_per_trio,
                        unique_ml_filtered_fusion_tx_per_trio)

    return unique_fusion_tx_per_trio, unique_ml_filtered_fusion_tx_per_trio


def create_fusion_tx_table(uniq_fusion_tx_per_trio: dict[str, int]) -> None:
    """
    Create a fusion transcript summary table.

    Placeholder function for future implementation.

    :param uniq_fusion_tx_per_trio: Dictionary mapping trio ID to unique fusion transcript count.
    """
    pass


def analyze_de_novo_stats(
    sample_ids: list[str], rules_filter: bool = True
) -> pd.DataFrame:
    """
    Compute de novo fusion transcript statistics for each proband.

    For each trio, compares the proband's unique fusion intron signatures
    against both parents to determine de novo and inherited counts.

    :param sample_ids: List of all sample IDs.
    :param rules_filter: If True, use rules-filtered transcripts; otherwise use ML-filtered transcripts.
    :return: DataFrame with columns 'trio', 'proband_id', 'total_unique_fusions', 'de_novo_count', 'inherited_count', 'de_novo_fraction', and 'all_de_novo'.
    """
    output_dir = f"{DATA_DIR}/unique_fusion_tx"

    trio_ids = set(["_".join(f.split("_")[:2]) for f in sample_ids])

    results = []

    for cur_trio in trio_ids:
        parent1_id = f"{cur_trio}_1_R1"
        parent2_id = f"{cur_trio}_2_R1"
        proband_id = f"{cur_trio}_3_R1"

        parent_signatures = set()
        for sample_id in [parent1_id, parent2_id]:
            if rules_filter:
                csv_path = f"{output_dir}/{sample_id}_unique_fusion_transcripts.csv"
            else:
                csv_path = f"{output_dir}/{sample_id}_ml_filtered_unique_fusion_transcripts.csv"
            try:
                df = pd.read_csv(csv_path)
                parent_signatures.update(df['intron_signature'].tolist())
            except FileNotFoundError:
                logger.info(f"Warning: Could not find {csv_path}")

        if rules_filter:
            proband_csv = f"{output_dir}/{proband_id}_unique_fusion_transcripts.csv"
        else:
            proband_csv = f"{output_dir}/{proband_id}_ml_filtered_unique_fusion_transcripts.csv"

        try:
            proband_df = pd.read_csv(proband_csv)
            proband_signatures = set(proband_df['intron_signature'].tolist())
            total_unique = len(proband_signatures)

            de_novo_signatures = proband_signatures - parent_signatures
            num_de_novo = len(de_novo_signatures)

            inherited_signatures = proband_signatures & parent_signatures
            num_inherited = len(inherited_signatures)

            results.append({
                'trio': cur_trio,
                'proband_id': proband_id,
                'total_unique_fusions': total_unique,
                'de_novo_count': num_de_novo,
                'inherited_count': num_inherited,
                'de_novo_fraction': num_de_novo / total_unique if total_unique > 0 else 0,
                'all_de_novo': num_de_novo == total_unique
            })

            logger.info(f"\nTrio {cur_trio} ({proband_id}):")
            logger.info(f"  Total unique fusions: {total_unique}")
            logger.info(f"  De novo: {num_de_novo} ({num_de_novo/total_unique*100:.1f}%)" if total_unique > 0 else "  De novo: 0")
            logger.info(f"  Inherited from parents: {num_inherited}")

        except FileNotFoundError:
            logger.info(f"Warning: Could not find {proband_csv}")

    results_df = pd.DataFrame(results)
    return results_df


def compute_importance_scores(
    fusion_df: pd.DataFrame, sample_ids: list[str]
) -> pd.DataFrame:
    """
    Compute weighted importance scores for fusion transcripts.

    Scoring weights: read support (0.4, log-transformed min-max scaled),
    distance (0.2, categorical 0--3 / 3), NMD escape (0.3, binary), and
    disease-associated gene involvement (0.1, count / 2).

    :param fusion_df: DataFrame with fusion characteristics including 'uniq_reads', 'distance_category', 'predicted_NMD', 'gene1_name', 'gene2_name', 'gene1_is_in_DA', and 'gene2_is_in_DA'.
    :param sample_ids: List of sample IDs used to load transcript read counts for normalization.
    :return: Copy of ``fusion_df`` with added score columns: 'read_score', 'distance_score', 'nmd_escape_score', 'da_score', 'importance_score', and related helper columns.
    """
    fusion_df = fusion_df.copy()

    all_fusion_genes = set(fusion_df['gene1_name'].tolist() + fusion_df['gene2_name'].tolist())

    all_gene_reads = []
    for sample_id in sample_ids:
        try:
            sample_annotated = read_sqanti3_annotated(sample_id, rules_filter=False)
            gene_transcripts = sample_annotated[sample_annotated['gene_name'].isin(all_fusion_genes)]
            all_gene_reads.extend(gene_transcripts['uniq_reads'].tolist())
        except FileNotFoundError:
            continue

    if all_gene_reads:
        log_reads = np.log1p(all_gene_reads)
        log_min, log_max = np.min(log_reads), np.max(log_reads)

        read_75th_percentile = np.percentile(all_gene_reads, 75)

        fusion_df['log_reads'] = np.log1p(fusion_df['uniq_reads'])
        if log_max > log_min:
            fusion_df['read_score'] = (fusion_df['log_reads'] - log_min) / (log_max - log_min)
        else:
            fusion_df['read_score'] = 0.5
        fusion_df['high_read_support'] = fusion_df['uniq_reads'] >= read_75th_percentile
    else:
        fusion_df['read_score'] = 0.5
        fusion_df['high_read_support'] = False

    distance_map = {"2 kb": 0, "20 kb": 1, ">20 kb": 2, "interchromosomal": 3}
    fusion_df['distance_raw_score'] = fusion_df['distance_category'].map(distance_map)
    fusion_df['distance_score'] = fusion_df['distance_raw_score'] / 3.0

    # NMD=True → 0 (will be degraded), NMD=False → 1 (escapes NMD, more likely functional)
    fusion_df['nmd_escape_score'] = (~fusion_df['predicted_NMD'].fillna(False)).astype(int)
    fusion_df['escapes_nmd'] = ~fusion_df['predicted_NMD'].fillna(False)

    fusion_df['da_count'] = fusion_df['gene1_is_in_DA'].astype(int) + fusion_df['gene2_is_in_DA'].astype(int)
    fusion_df['da_score'] = fusion_df['da_count'] / 2.0
    fusion_df['has_da_gene'] = fusion_df['da_count'] > 0

    fusion_df['importance_score'] = (
        0.4 * fusion_df['read_score'] +
        0.2 * fusion_df['distance_score'] +
        0.3 * fusion_df['nmd_escape_score'] +
        0.1 * fusion_df['da_score']
    )

    return fusion_df


def plot_fusion_bubble_plots(rules_filter: bool = False) -> None:
    """
    Generate bubble plots of fusion transcripts per proband.

    Each bubble represents a fusion transcript with x-axis as read count,
    y-axis as fusion distance (log scale with visual break for
    interchromosomal), size proportional to importance score, and color
    indicating NMD-escape and/or DA gene involvement. Only fusions that
    escape NMD or involve a DA gene are included.

    :param rules_filter: If True, use rules-filtered data; otherwise use ML-filtered data.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from adjustText import adjust_text

    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    suffix = "" if rules_filter else "_ml_filtered"

    fusion_path = f"{output_dir}/fusion_characteristics{suffix}.csv"
    try:
        fusion_df = pd.read_csv(fusion_path)
    except FileNotFoundError:
        logger.info(f"Error: Could not find {fusion_path}")
        return

    logger.info(f"\nLoaded {len(fusion_df)} fusion transcripts from {fusion_path}")

    sample_ids = get_long_read_sample_ids()
    proband_ids = [s for s in sample_ids if s.endswith("_3_R1")]

    logger.info("Computing importance scores...")
    fusion_df = compute_importance_scores(fusion_df, sample_ids)

    proband_fusions = fusion_df[fusion_df['sample_id'].isin(proband_ids)].copy()

    filtered_fusions = proband_fusions[
        proband_fusions['escapes_nmd'] | proband_fusions['has_da_gene']
    ].copy()

    logger.info(f"Filtered to {len(filtered_fusions)} fusions with NMD-escape or DA genes (from {len(proband_fusions)} proband fusions)")

    def get_color_category(row: pd.Series) -> str:
        """
        Assign a color category based on NMD escape and DA gene status.

        :param row: Row with 'escapes_nmd' and 'has_da_gene' boolean fields.
        :return: One of 'NMD-escape + DA', 'NMD-escape only', or 'DA only'.
        """
        escapes_nmd = row['escapes_nmd']
        has_da = row['has_da_gene']
        if escapes_nmd and has_da:
            return 'NMD-escape + DA'
        elif escapes_nmd:
            return 'NMD-escape only'
        else:
            return 'DA only'

    filtered_fusions['color_category'] = filtered_fusions.apply(get_color_category, axis=1)

    color_map = {
        'NMD-escape only': '#E69F00',   # Orange
        'DA only': '#009E73',            # Green
        'NMD-escape + DA': '#CC79A7'     # Muted red/pink
    }

    top_fusions_list = []

    for proband_id in proband_ids:
        proband_data = filtered_fusions[filtered_fusions['sample_id'] == proband_id].copy()

        if proband_data.empty:
            logger.info(f"No NMD/DA fusions for {proband_id}, skipping plot.")
            continue

        logger.info(f"Plotting {len(proband_data)} fusions for {proband_id}...")

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(10, 8),
            gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05}
        )

        inter_chrom = proband_data[proband_data['interchromosomal'] == True].copy()
        intra_chrom = proband_data[proband_data['interchromosomal'] == False].copy()

        size_scale = 500
        min_size = 50

        if not intra_chrom.empty:
            intra_chrom['distance_numeric'] = intra_chrom['fusion_distance'].replace('NA', np.nan).astype(float)
            intra_chrom = intra_chrom.dropna(subset=['distance_numeric'])

            if not intra_chrom.empty:
                sizes = min_size + intra_chrom['importance_score'] * size_scale
                colors = [color_map[cat] for cat in intra_chrom['color_category']]

                ax_bottom.scatter(
                    intra_chrom['uniq_reads'],
                    intra_chrom['distance_numeric'],
                    s=sizes,
                    c=colors,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=0.5
                )
                ax_bottom.set_yscale('log')

        if not inter_chrom.empty:
            y_positions = np.ones(len(inter_chrom))
            sizes = min_size + inter_chrom['importance_score'].values * size_scale
            colors = [color_map[cat] for cat in inter_chrom['color_category']]

            ax_top.scatter(
                inter_chrom['uniq_reads'],
                y_positions,
                s=sizes,
                c=colors,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            ax_top.set_ylim(0.5, 1.5)
            ax_top.set_yticks([1])
            ax_top.set_yticklabels(['Inter-\nchrom'])
        else:
            ax_top.set_ylim(0.5, 1.5)
            ax_top.set_yticks([1])
            ax_top.set_yticklabels(['Inter-\nchrom'])
            ax_top.text(0.5, 1, 'No interchromosomal fusions',
                       ha='center', va='center', transform=ax_top.transAxes,
                       fontsize=9, color='gray')

        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)

        d = 0.015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_bottom.transAxes, color='k', clip_on=False)
        ax_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)
        ax_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

        ax_bottom.set_xlabel('Read Count', fontsize=11)
        ax_bottom.set_ylabel('Fusion Distance (bp)', fontsize=11)
        ax_top.set_ylabel('')

        all_reads = proband_data['uniq_reads']
        x_range = all_reads.max() - all_reads.min()
        x_padding = max(x_range * 0.15, all_reads.max() * 0.1)
        x_min, x_max = all_reads.min() - x_padding, all_reads.max() + x_padding
        x_min = max(0, x_min)
        ax_top.set_xlim(x_min, x_max)
        ax_bottom.set_xlim(x_min, x_max)

        top_5 = proband_data.nlargest(5, 'importance_score')
        texts_bottom = []
        texts_top = []

        for _, row in top_5.iterrows():
            label = f"{row['gene1_name']}_{row['gene2_name']}"
            x = row['uniq_reads']

            if row['interchromosomal']:
                txt = ax_top.text(x, 1, label, fontsize=8, ha='left')
                texts_top.append(txt)
            else:
                y = row['fusion_distance'] if row['fusion_distance'] != 'NA' else None
                if y is not None:
                    txt = ax_bottom.text(x, float(y), label, fontsize=8, ha='left')
                    texts_bottom.append(txt)

        if texts_bottom:
            adjust_text(texts_bottom, ax=ax_bottom, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        if texts_top:
            adjust_text(texts_top, ax=ax_top, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        ax_bottom.axhline(y=20000, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
        ax_bottom.text(x_max * 0.98, 20000, '20 kb', va='bottom', ha='right', fontsize=8, color='gray')

        legend_elements = [
            Patch(facecolor=color_map['NMD-escape only'], edgecolor='black', label='NMD-escape only', alpha=0.7),
            Patch(facecolor=color_map['DA only'], edgecolor='black', label='DA only', alpha=0.7),
            Patch(facecolor=color_map['NMD-escape + DA'], edgecolor='black', label='NMD-escape + DA', alpha=0.7),
        ]
        ax_bottom.legend(handles=legend_elements, loc='upper right', fontsize=9)

        fig.suptitle(f'Fusion Transcripts - {proband_id}', fontsize=12, y=0.98)

        plot_path = f"{output_dir}/{proband_id}_fusion_bubble_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved bubble plot to {plot_path}")

        top_5_with_sample = top_5.copy()
        top_fusions_list.append(top_5_with_sample)

    if top_fusions_list:
        top_fusions_df = pd.concat(top_fusions_list, ignore_index=True)
        score_cols = ['importance_score', 'read_score', 'distance_score', 'nmd_escape_score', 'da_score']
        helper_cols = ['log_reads', 'distance_raw_score', 'da_count', 'has_da_gene', 'escapes_nmd', 'high_read_support', 'color_category']
        other_cols = [c for c in top_fusions_df.columns if c not in score_cols and c not in helper_cols]
        final_cols = other_cols + score_cols
        top_fusions_df = top_fusions_df[final_cols]

        table_path = f"{output_dir}/top_fusions_by_importance{suffix}.csv"
        top_fusions_df.to_csv(table_path, index=False)
        logger.info(f"\nSaved top fusions table to {table_path}")
        logger.info(f"Total top fusions across all probands: {len(top_fusions_df)}")


def plot_unique_fusion_tx_per_sample() -> Optional[pd.DataFrame]:
    """
    Generate stacked bar plots of unique fusion transcripts per sample.

    Reads from the pre-computed ML-filtered fusion characteristics CSV
    and produces both absolute count and proportion plots, categorized
    by fusion distance. Probands are highlighted with bold x-axis labels.

    :return: DataFrame with counts per sample and distance category, or None if the input file is not found.
    """
    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    fusion_path = f"{output_dir}/fusion_characteristics_ml_filtered.csv"

    try:
        fusion_df = pd.read_csv(fusion_path)
    except FileNotFoundError:
        logger.info(f"Error: Could not find {fusion_path}")
        logger.info("Run generate_all_fusion_circos_plots(rules_filter=False) first.")
        return None

    logger.info(f"Loaded {len(fusion_df)} fusion transcripts from {fusion_path}")

    counts_df = fusion_df.groupby(['sample_id', 'distance_category']).size().unstack(fill_value=0)

    distance_categories = ['2 kb', '20 kb', '>20 kb', 'interchromosomal']
    for cat in distance_categories:
        if cat not in counts_df.columns:
            counts_df[cat] = 0

    counts_df = counts_df[distance_categories]

    counts_df['_total'] = counts_df.sum(axis=1)
    counts_df = counts_df.sort_values('_total', ascending=False)
    counts_df = counts_df.drop(columns=['_total'])

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [DISTANCE_COLORS[cat] for cat in distance_categories]

    counts_df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        width=0.75
    )

    ax.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Unique Fusion Transcripts', fontsize=12, fontweight='bold')
    ax.set_title('Unique Fusion Transcripts per Sample by Distance Category', fontsize=14, fontweight='bold', pad=15)

    sample_ids = counts_df.index.tolist()
    ax.set_xticks(range(len(sample_ids)))
    labels = ax.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=9)
    for i, label in enumerate(labels):
        if '_3_R1' in sample_ids[i]:
            label.set_fontweight('bold')

    legend = ax.legend(
        title='Distance Category',
        loc='upper right',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        edgecolor='black',
        framealpha=0.95
    )
    legend.get_title().set_fontweight('bold')

    totals = counts_df.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + 0.3, str(int(total)), ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()

    plot_path = f"{output_dir}/unique_fusion_tx_per_sample.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved plot to {plot_path}")

    proportions_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100

    fig2, ax2 = plt.subplots(figsize=(14, 6))

    proportions_df.plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        color=colors,
        edgecolor='black',
        linewidth=0.5,
        width=0.75
    )

    ax2.set_xlabel('Sample ID', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Proportion of Fusion Transcripts (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Proportion of Unique Fusion Transcripts by Distance Category', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 100)

    ax2.set_xticks(range(len(sample_ids)))
    labels2 = ax2.set_xticklabels(sample_ids, rotation=45, ha='right', fontsize=9)
    for i, label in enumerate(labels2):
        if '_3_R1' in sample_ids[i]:
            label.set_fontweight('bold')

    legend2 = ax2.legend(
        title='Distance Category',
        loc='upper right',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        edgecolor='black',
        framealpha=0.95
    )
    legend2.get_title().set_fontweight('bold')

    for i, total in enumerate(totals):
        ax2.text(i, 101, f'n={int(total)}', ha='center', va='bottom', fontsize=7, fontstyle='italic')

    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax2.set_axisbelow(True)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    prop_plot_path = f"{output_dir}/unique_fusion_tx_per_sample_proportion.png"
    plt.savefig(prop_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    logger.info(f"Saved proportion plot to {prop_plot_path}")

    table_path = f"{output_dir}/unique_fusion_tx_counts_per_sample.csv"
    counts_df.to_csv(table_path)
    logger.info(f"Saved counts table to {table_path}")

    logger.info("\nSummary Statistics:")
    logger.info(f"  Total samples: {len(counts_df)}")
    logger.info(f"  Total unique fusions: {totals.sum()}")
    logger.info(f"  Mean per sample: {totals.mean():.1f}")
    logger.info(f"  Median per sample: {totals.median():.1f}")
    logger.info(f"  Range: {totals.min()} - {totals.max()}")
    logger.info("\nBy distance category:")
    for cat in distance_categories:
        cat_total = counts_df[cat].sum()
        cat_pct = cat_total / totals.sum() * 100 if totals.sum() > 0 else 0
        logger.info(f"  {cat}: {cat_total} ({cat_pct:.1f}%)")

    return counts_df


def plot_proband_vs_parents_fusion_comparison(
    unique_fusion_tx: dict[str, pd.DataFrame], sample_ids: list[str]
) -> pd.DataFrame:
    """
    Compare unique fusion transcript counts between probands and parents.

    Generates a paired dot plot with connecting lines showing individual
    trio relationships and conducts a paired t-test of proband count
    versus mean parent count.

    :param unique_fusion_tx: Dictionary mapping sample_id to DataFrame of unique fusion transcripts.
    :param sample_ids: List of all sample IDs used to derive trio groupings.
    :return: DataFrame with columns 'trio', 'parent1_count', 'parent2_count', 'parent_mean', and 'proband_count'.
    """
    from scipy import stats

    output_dir = f"{DATA_DIR}/unique_fusion_tx"

    trio_ids = sorted(set(["_".join(f.split("_")[:2]) for f in sample_ids]))

    trio_data = []
    for cur_trio in trio_ids:
        parent1_id = f"{cur_trio}_1_R1"
        parent2_id = f"{cur_trio}_2_R1"
        proband_id = f"{cur_trio}_3_R1"

        parent1_count = len(unique_fusion_tx.get(parent1_id, pd.DataFrame()))
        parent2_count = len(unique_fusion_tx.get(parent2_id, pd.DataFrame()))
        proband_count = len(unique_fusion_tx.get(proband_id, pd.DataFrame()))
        parent_mean = (parent1_count + parent2_count) / 2

        trio_data.append({
            'trio': cur_trio,
            'parent1_count': parent1_count,
            'parent2_count': parent2_count,
            'parent_mean': parent_mean,
            'proband_count': proband_count
        })

    trio_df = pd.DataFrame(trio_data)

    t_stat, p_value = stats.ttest_rel(trio_df['proband_count'], trio_df['parent_mean'])

    if p_value < 0.001:
        sig_stars = '***'
    elif p_value < 0.01:
        sig_stars = '**'
    elif p_value < 0.05:
        sig_stars = '*'
    else:
        sig_stars = 'ns'

    logger.info(f"Paired t-test (Proband vs Parent Mean):")
    logger.info(f"  t-statistic: {t_stat:.3f}")
    logger.info(f"  p-value: {p_value:.4f} ({sig_stars})")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, ax = plt.subplots(figsize=(5, 6))

    parent_color = '#0072B2'
    proband_color = '#D55E00'

    x_parent, x_proband = 0, 1

    for _, row in trio_df.iterrows():
        parent_mean = row['parent_mean']
        proband_count = row['proband_count']

        line_color = '#CC79A7' if proband_count > parent_mean else '#999999'

        ax.plot([x_parent, x_proband], [parent_mean, proband_count],
                color=line_color, linewidth=1.2, alpha=0.6, zorder=1)

    ax.scatter([x_parent] * len(trio_df), trio_df['parent_mean'],
               s=80, c=parent_color, edgecolors='white', linewidths=1.5, zorder=2, label='Parent mean')
    ax.scatter([x_proband] * len(trio_df), trio_df['proband_count'],
               s=80, c=proband_color, edgecolors='white', linewidths=1.5, zorder=2, label='Proband')

    parent_means = trio_df['parent_mean'].values
    proband_counts = trio_df['proband_count'].values

    ax.errorbar(x_parent - 0.15, np.mean(parent_means), yerr=np.std(parent_means),
                fmt='D', color=parent_color, markersize=10, capsize=5, capthick=2,
                markeredgecolor='black', markeredgewidth=1, zorder=3)
    ax.errorbar(x_proband + 0.15, np.mean(proband_counts), yerr=np.std(proband_counts),
                fmt='D', color=proband_color, markersize=10, capsize=5, capthick=2,
                markeredgecolor='black', markeredgewidth=1, zorder=3)

    y_max = max(trio_df['parent_mean'].max(), trio_df['proband_count'].max())
    bracket_y = y_max * 1.1

    ax.plot([x_parent, x_parent, x_proband, x_proband],
            [bracket_y, bracket_y + y_max * 0.03, bracket_y + y_max * 0.03, bracket_y],
            color='black', linewidth=1.2)
    ax.text((x_parent + x_proband) / 2, bracket_y + y_max * 0.05,
            f'{sig_stars}\np = {p_value:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks([x_parent, x_proband])
    ax.set_xticklabels(['Parent\nmean', 'Proband'], fontsize=11)
    ax.set_ylabel('Unique fusion transcripts', fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, bracket_y + y_max * 0.15)

    ax.text(0.5, -0.12, f'n = {len(trio_df)} trios', ha='center', va='top',
            transform=ax.transAxes, fontsize=9, style='italic')

    legend_elements = [
        Line2D([0], [0], color='#CC79A7', linewidth=2, label='Increased in proband'),
        Line2D([0], [0], color='#999999', linewidth=2, label='Decreased in proband'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9)

    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    plot_path = f"{output_dir}/proband_vs_parents_fusion_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved comparison plot to {plot_path}")

    trio_df.to_csv(f"{output_dir}/proband_vs_parents_counts.csv", index=False)
    logger.info(f"Saved counts table to {output_dir}/proband_vs_parents_counts.csv")

    logger.info(f"\nSummary:")
    logger.info(f"  Number of trios: {len(trio_df)}")
    logger.info(f"  Parent mean fusion count: {np.mean(parent_means):.1f} +/- {np.std(parent_means):.1f}")
    logger.info(f"  Proband mean fusion count: {np.mean(proband_counts):.1f} +/- {np.std(proband_counts):.1f}")

    return trio_df


def plot_fusion_heatmap(
    sample_ids: list[str], rules_filter: bool = False
) -> Optional[pd.DataFrame]:
    """
    Generate a heatmap of fusion characteristics per trio.

    Rows represent trios (probands only) and columns represent features
    including de novo percentage, NMD escape percentage, DA gene
    percentage, and distance category proportions.

    :param sample_ids: List of all sample IDs used to derive trio groupings.
    :param rules_filter: If True, use rules-filtered data; otherwise use ML-filtered data.
    :return: Heatmap data DataFrame indexed by trio, or None if the input file is not found.
    """
    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    suffix = "" if rules_filter else "_ml_filtered"

    fusion_path = f"{output_dir}/fusion_characteristics{suffix}.csv"
    try:
        fusion_df = pd.read_csv(fusion_path)
    except FileNotFoundError:
        logger.info(f"Error: Could not find {fusion_path}")
        return

    trio_ids = sorted(set(["_".join(f.split("_")[:2]) for f in sample_ids]))

    heatmap_data = []

    for cur_trio in trio_ids:
        proband_id = f"{cur_trio}_3_R1"
        proband_fusions = fusion_df[fusion_df['sample_id'] == proband_id]

        if proband_fusions.empty:
            continue

        total = len(proband_fusions)

        dist_counts = proband_fusions['distance_category'].value_counts()
        pct_2kb = dist_counts.get('2 kb', 0) / total * 100
        pct_20kb = dist_counts.get('20 kb', 0) / total * 100
        pct_gt20kb = dist_counts.get('>20 kb', 0) / total * 100
        pct_inter = dist_counts.get('interchromosomal', 0) / total * 100

        pct_nmd_escape = (~proband_fusions['predicted_NMD'].fillna(False)).sum() / total * 100
        pct_da_gene = ((proband_fusions['gene1_is_in_DA'] | proband_fusions['gene2_is_in_DA'])).sum() / total * 100

        de_novo_count = proband_fusions['is_de_novo'].sum() if 'is_de_novo' in proband_fusions.columns else 0
        pct_de_novo = de_novo_count / total * 100

        heatmap_data.append({
            'Trio': cur_trio,
            'Total fusions': total,
            'De novo (%)': pct_de_novo,
            'NMD escape (%)': pct_nmd_escape,
            'DA gene (%)': pct_da_gene,
            '<2 kb (%)': pct_2kb,
            '2-20 kb (%)': pct_20kb,
            '>20 kb (%)': pct_gt20kb,
            'Inter-chrom (%)': pct_inter,
        })

    heatmap_df = pd.DataFrame(heatmap_data).set_index('Trio')

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
    })

    fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_df) * 0.4)))

    plot_df = heatmap_df.drop(columns=['Total fusions'])

    sns.heatmap(
        plot_df,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Percentage (%)'},
        ax=ax
    )

    ax2 = ax.twinx()
    ax2.barh(
        range(len(heatmap_df)),
        heatmap_df['Total fusions'],
        height=0.6,
        color='#56B4E9',
        alpha=0.7,
        left=len(plot_df.columns) + 0.5
    )
    ax2.set_ylabel('Total fusions', fontsize=10)
    ax2.set_ylim(-0.5, len(heatmap_df) - 0.5)
    ax2.invert_yaxis()

    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title('Fusion Transcript Characteristics by Trio (Probands)', fontsize=12, pad=15)

    plt.tight_layout()

    plot_path = f"{output_dir}/fusion_heatmap{suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved heatmap to {plot_path}")

    heatmap_df.to_csv(f"{output_dir}/fusion_heatmap_data{suffix}.csv")

    return heatmap_df


def plot_fusion_upset(
    sample_ids: list[str], rules_filter: bool = False
) -> Optional[dict[str, set[str]]]:
    """
    Generate an UpSet plot of shared fusion gene pairs across trios.

    Shows which gene pair fusions are unique to specific trios versus
    shared across multiple trios.

    :param sample_ids: List of all sample IDs used to derive trio groupings.
    :param rules_filter: If True, use rules-filtered data; otherwise use ML-filtered data.
    :return: Dictionary mapping trio ID to set of gene pair strings, or None if no data is found or the upsetplot library is unavailable.
    """
    try:
        from upsetplot import UpSet, from_contents
    except ImportError:
        logger.info("Error: upsetplot not installed. Install with: pip install upsetplot")
        return

    output_dir = f"{DATA_DIR}/unique_fusion_tx"
    suffix = "" if rules_filter else "_ml_filtered"

    trio_ids = sorted(set(["_".join(f.split("_")[:2]) for f in sample_ids]))

    trio_gene_pairs = {}

    for cur_trio in trio_ids:
        gene_pairs = set()
        for member in ['1', '2', '3']:
            sample_id = f"{cur_trio}_{member}_R1"
            csv_path = f"{output_dir}/{sample_id}{suffix}_unique_fusion_transcripts.csv"
            if suffix:
                csv_path = f"{output_dir}/{sample_id}_ml_filtered_unique_fusion_transcripts.csv"
            else:
                csv_path = f"{output_dir}/{sample_id}_unique_fusion_transcripts.csv"

            try:
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    genes = row['associated_gene'].split('_')
                    if len(genes) >= 2:
                        gene_pair = f"{genes[0]}_{genes[-1]}"
                        gene_pairs.add(gene_pair)
            except FileNotFoundError:
                continue

        if gene_pairs:
            trio_gene_pairs[cur_trio] = gene_pairs

    if not trio_gene_pairs:
        logger.info("No data found for upset plot")
        return

    upset_data = from_contents(trio_gene_pairs)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
    })

    fig = plt.figure(figsize=(12, 8))

    upset = UpSet(
        upset_data,
        subset_size='count',
        show_counts=True,
        sort_by='cardinality',
        sort_categories_by='cardinality',
        facecolor='#0072B2',
        element_size=40
    )
    upset.plot(fig=fig)

    plt.suptitle('Shared Fusion Gene Pairs Across Trios', fontsize=12, y=1.02)

    plot_path = f"{output_dir}/fusion_upset{suffix}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved upset plot to {plot_path}")

    all_pairs = set().union(*trio_gene_pairs.values())
    logger.info(f"\nUpset plot summary:")
    logger.info(f"  Total unique gene pairs: {len(all_pairs)}")
    logger.info(f"  Trios analyzed: {len(trio_gene_pairs)}")

    return trio_gene_pairs


if __name__ == "__main__":
    sample_ids = get_long_read_sample_ids()
    unique_fusion_tx = get_unique_fusion_tx(sample_ids, rule_filter=False)

    logger.info("\n" + "="*60)
    logger.info("PROBAND VS PARENTS COMPARISON")
    logger.info("="*60)
    plot_proband_vs_parents_fusion_comparison(unique_fusion_tx, sample_ids)

    gene_coords = load_gene_coordinates()
    for sample_id, df in unique_fusion_tx.items():
        df["isoform_interval"] = df["chrom"] + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
        df["gene_names"] = df["associated_gene"].apply(
            lambda ag: "_".join(
                gene_coords[gid][3] if gid in gene_coords else gid
                for gid in ag.split("_")
            )
        )

        def _compute_distance_category(row: pd.Series) -> str:
            """
            Compute the distance category for a fusion transcript row.

            :param row: Row with an 'associated_gene' field containing underscore-separated gene IDs.
            :return: Distance category string or 'unknown' if gene coordinates cannot be resolved.
            """
            gene_ids = row["associated_gene"].split("_")
            if len(gene_ids) < 2:
                return "unknown"
            first_gene, last_gene = gene_ids[0], gene_ids[-1]
            if first_gene not in gene_coords or last_gene not in gene_coords:
                return "unknown"
            chrom1, start1, end1, _ = gene_coords[first_gene]
            chrom2, start2, end2, _ = gene_coords[last_gene]
            _, category, _ = get_distance_category(chrom1, start1, end1, chrom2, start2, end2)
            return category

        df["distance_category"] = df.apply(_compute_distance_category, axis=1)

    unique_fusion_tx_json = {}
    for sample_id, df in unique_fusion_tx.items():
        logger.info(f"Sample {sample_id} has {df.shape[0]} unique fusion transcripts.")
        df.to_csv(f"{DATA_DIR}/unique_fusion_tx/"
                  f"{sample_id}_ml_filtered_unique_fusion_transcripts.csv",
                  index=False)
        unique_fusion_tx_json[sample_id] = df.to_dict(orient='records')

    json_output_path = f"{DATA_DIR}/unique_fusion_tx/unique_fusion_tx_ml_filtered.json"
    with open(json_output_path, 'w') as f:
        json.dump(unique_fusion_tx_json, f, indent=2)
    logger.info(f"Saved unique fusion transcripts to {json_output_path}")

    logger.info("\n" + "="*60)
    logger.info("DE NOVO FUSION TRANSCRIPT ANALYSIS")
    logger.info("="*60)
    de_novo_stats = analyze_de_novo_stats(sample_ids, rules_filter=False)
    de_novo_stats.to_csv(f"{DATA_DIR}/unique_fusion_tx/de_novo_stats_ml_filtered.csv", index=False)
    logger.info(f"\nSaved de novo stats to {DATA_DIR}/unique_fusion_tx/de_novo_stats_ml_filtered.csv")

    logger.info("\n" + "-"*60)
    logger.info("SUMMARY:")
    logger.info(f"  Total probands: {len(de_novo_stats)}")
    logger.info(f"  Probands where ALL unique fusions are de novo: {de_novo_stats['all_de_novo'].sum()}")
    logger.info(f"  Average de novo fraction: {de_novo_stats['de_novo_fraction'].mean()*100:.1f}%")
    logger.info("-"*60)

    generate_all_fusion_circos_plots(rules_filter=False)

    logger.info("\n" + "="*60)
    logger.info("DE NOVO CIRCOS SUMMARY (PROBANDS ONLY)")
    logger.info("="*60)
    plot_de_novo_circos_summary(rules_filter=False)

    logger.info("\n" + "="*60)
    logger.info("UNIQUE FUSION TX PER SAMPLE PLOT")
    logger.info("="*60)
    plot_unique_fusion_tx_per_sample()

    logger.info("\n" + "="*60)
    logger.info("FUSION IMPORTANCE BUBBLE PLOTS")
    logger.info("="*60)
    plot_fusion_bubble_plots(rules_filter=False)

    logger.info("\n" + "="*60)
    logger.info("FUSION CHARACTERISTICS HEATMAP")
    logger.info("="*60)
    plot_fusion_heatmap(sample_ids, rules_filter=False)

    logger.info("\n" + "="*60)
    logger.info("FUSION UPSET PLOT")
    logger.info("="*60)
    plot_fusion_upset(sample_ids, rules_filter=False)

