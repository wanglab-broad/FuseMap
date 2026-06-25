#!/usr/bin/env python
"""
Step ③ Druggability Analysis
============================

Connects altered niche marker genes to drug target databases
(DGIdb + Open Targets) to identify actionable drug-gene-niche relationships.

Usage:
    python scripts/druggability_analysis.py \
        --dataset 13months-disease-replicate_1.h5ad

Output:
    <discgal_dir>/<dataset>/niche_analysis/druggability/
"""
import argparse
import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
DGIDB_URL = "https://dgidb.org/api/graphql"
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
AD_DISEASE_ID = "EFO_0000249"  # Alzheimer's disease

# Known mouse → human gene name exceptions (most are just .upper())
MOUSE_TO_HUMAN = {
    "Cst3": "CST3", "Cst7": "CST7", "C1qa": "C1QA", "C1qb": "C1QB", "C1qc": "C1QC",
    "Hexb": "HEXB", "Trem2": "TREM2", "Gfap": "GFAP", "Ctss": "CTSS", "Ctsb": "CTSB",
    "Ctsd": "CTSD", "Ctsl": "CTSL", "Flt1": "FLT1", "Bsg": "BSG", "Vim": "VIM",
    "Apoe": "APOE", "Cd68": "CD68", "Cd74": "CD74", "Axl": "AXL", "Csf1r": "CSF1R",
    "Clec7a": "CLEC7A", "Cd9": "CD9", "Cd63": "CD63", "Cd83": "CD83",
    "Slc1a3": "SLC1A3", "Aldoc": "ALDOC", "Atp1a2": "ATP1A2",
    "Gad1": "GAD1", "Gad2": "GAD2", "Mag": "MAG", "Mog": "MOG", "Plp1": "PLP1",
    "Snap25": "SNAP25", "Dbi": "DBI", "Nnat": "NNAT",
}


def parse_args():
    parser = argparse.ArgumentParser(description='Druggability analysis for altered niche markers')
    parser.add_argument('--discgal_dir', type=str, default='./output/discgal_disease_test')
    parser.add_argument('--dataset', type=str, default='13months-disease-replicate_1.h5ad')
    parser.add_argument('--pval_cutoff', type=float, default=0.05)
    parser.add_argument('--top_n_genes', type=int, default=10,
                        help='Top N genes per niche to query')
    return parser.parse_args()


# ============================================================
# Stage 1: Gene Name Conversion
# ============================================================
def mouse_to_human(gene_name):
    """Convert mouse gene symbol to human ortholog."""
    if gene_name in MOUSE_TO_HUMAN:
        return MOUSE_TO_HUMAN[gene_name]
    return gene_name.upper()


# ============================================================
# Stage 2: DGIdb Query
# ============================================================
def query_dgidb(gene_list, max_retries=3):
    """Query DGIdb v5 GraphQL API for drug-gene interactions."""
    logger.info(f"Querying DGIdb for {len(gene_list)} genes...")

    query = """
    query($genes: [String!]!) {
      genes(names: $genes) {
        nodes {
          name
          geneCategoriesWithSources {
            name
          }
          interactions {
            interactionScore
            interactionTypes {
              type
              directionality
            }
            drug {
              name
              conceptId
              approved
              immunotherapy
              antiNeoplastic
            }
            interactionClaims {
              source {
                fullName
              }
            }
          }
        }
      }
    }
    """

    variables = {"genes": gene_list}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                DGIDB_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                logger.warning(f"  DGIdb returned errors: {data['errors']}")

            results = []
            if "data" in data and data["data"]["genes"]["nodes"]:
                for gene_node in data["data"]["genes"]["nodes"]:
                    gene_name = gene_node["name"]

                    # Gene categories (druggability info)
                    categories = [c["name"] for c in (gene_node.get("geneCategoriesWithSources") or [])]

                    for interaction in (gene_node.get("interactions") or []):
                        drug = interaction.get("drug", {})
                        drug_name = drug.get("name", "Unknown")
                        approved = drug.get("approved", False)

                        approval_str = "Approved" if approved else "Not Approved"

                        # Interaction types
                        int_types = interaction.get("interactionTypes") or []
                        int_type_str = ", ".join([t.get("type", "") for t in int_types]) if int_types else "Unknown"

                        # Sources
                        sources = []
                        for claim in (interaction.get("interactionClaims") or []):
                            src = claim.get("source", {})
                            if src and src.get("fullName"):
                                sources.append(src["fullName"])
                        source_str = "; ".join(set(sources)) if sources else ""

                        score = interaction.get("interactionScore", None)

                        results.append({
                            "gene": gene_name,
                            "gene_categories": "; ".join(categories),
                            "drug": drug_name,
                            "interaction_type": int_type_str,
                            "approval_status": approval_str,
                            "interaction_score": score,
                            "sources": source_str,
                            "is_approved": approved,
                        })

            logger.info(f"  DGIdb returned {len(results)} drug-gene interactions")
            return pd.DataFrame(results)

        except requests.exceptions.RequestException as e:
            logger.warning(f"  DGIdb attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    logger.error("  DGIdb query failed after all retries")
    return pd.DataFrame()


# ============================================================
# Stage 3: Open Targets Query
# ============================================================
def query_opentargets(gene_symbols, disease_id=AD_DISEASE_ID, max_retries=3):
    """Query Open Targets for disease association scores."""
    logger.info(f"Querying Open Targets for {len(gene_symbols)} genes (disease: {disease_id})...")

    # Skipping Open Targets due to API v4 schema changes
    logger.info("  Skipping Open Targets query (temporarily disabled)")
    return {}


# ============================================================
# Stage 4: Integration & Visualization
# ============================================================
def integrate_and_visualize(marker_df, dgidb_df, ot_results, out_dir, dataset_name):
    """Merge all results and generate visualizations."""
    short = dataset_name.replace('.h5ad', '')
    logger.info("Integrating results...")

    # Build integrated table
    rows = []
    for _, marker in marker_df.iterrows():
        gene_mouse = marker['gene']
        gene_human = mouse_to_human(gene_mouse)
        niche = marker['niche']
        lfc = marker['logfoldchange']
        pval = marker['pval_adj']
        score = marker['score']

        # DGIdb matches
        dgi_matches = dgidb_df[dgidb_df['gene'] == gene_human] if len(dgidb_df) > 0 else pd.DataFrame()

        # Open Targets matches
        ot_info = ot_results.get(gene_human, {})

        if len(dgi_matches) > 0:
            for _, dgi in dgi_matches.iterrows():
                rows.append({
                    'niche': niche,
                    'gene_mouse': gene_mouse,
                    'gene_human': gene_human,
                    'logfoldchange': lfc,
                    'pval_adj': pval,
                    'wilcoxon_score': score,
                    'drug': dgi['drug'],
                    'interaction_type': dgi['interaction_type'],
                    'approval_status': dgi['approval_status'],
                    'is_approved': dgi['is_approved'],
                    'gene_categories': dgi['gene_categories'],
                    'dgidb_sources': dgi['sources'],
                    'ot_approved_name': ot_info.get('approved_name', ''),
                    'ot_n_known_drugs': ot_info.get('n_known_drugs', 0),
                    'ot_ad_drugs': len(ot_info.get('ad_related_drugs', [])),
                })
        else:
            rows.append({
                'niche': niche,
                'gene_mouse': gene_mouse,
                'gene_human': gene_human,
                'logfoldchange': lfc,
                'pval_adj': pval,
                'wilcoxon_score': score,
                'drug': '',
                'interaction_type': '',
                'approval_status': '',
                'is_approved': False,
                'gene_categories': '',
                'dgidb_sources': '',
                'ot_approved_name': ot_info.get('approved_name', ''),
                'ot_n_known_drugs': ot_info.get('n_known_drugs', 0),
                'ot_ad_drugs': len(ot_info.get('ad_related_drugs', [])),
            })

    result_df = pd.DataFrame(rows)

    # Save full results
    result_df.to_csv(os.path.join(out_dir, 'druggability_results.csv'), index=False)
    logger.info(f"  Saved druggability_results.csv ({len(result_df)} rows)")

    # --- Summary statistics ---
    druggable_genes = result_df[result_df['drug'] != '']['gene_human'].unique()
    all_genes = result_df['gene_human'].unique()
    logger.info(f"\n  === DRUGGABILITY SUMMARY: {short} ===")
    logger.info(f"  Total unique marker genes: {len(all_genes)}")
    logger.info(f"  Druggable genes (with known interactions): {len(druggable_genes)}")
    logger.info(f"  Druggability rate: {len(druggable_genes) / len(all_genes):.1%}")

    approved_drugs = result_df[(result_df['drug'] != '') & (result_df['is_approved'] == True)]
    logger.info(f"  Interactions with approved drugs: {len(approved_drugs)}")

    # Per-niche druggability summary
    niche_summary = []
    for niche in sorted(result_df['niche'].unique()):
        niche_data = result_df[result_df['niche'] == niche]
        n_genes = niche_data['gene_human'].nunique()
        n_druggable = niche_data[niche_data['drug'] != '']['gene_human'].nunique()
        n_drugs = niche_data[niche_data['drug'] != '']['drug'].nunique()
        top_drugs = ', '.join(niche_data[niche_data['drug'] != ''].drop_duplicates('drug')['drug'].head(3).tolist())
        niche_summary.append({
            'niche': niche,
            'n_marker_genes': n_genes,
            'n_druggable_genes': n_druggable,
            'n_unique_drugs': n_drugs,
            'top_drugs': top_drugs,
        })
    summary_df = pd.DataFrame(niche_summary)
    summary_df.to_csv(os.path.join(out_dir, 'druggability_niche_summary.csv'), index=False)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # --- Visualization 1: Druggable gene heatmap ---
    druggable_data = result_df[result_df['drug'] != ''].copy()
    if len(druggable_data) > 0:
        # Pivot: genes x niches, color by LFC
        pivot_genes = druggable_data.drop_duplicates(['gene_human', 'niche'])
        if len(pivot_genes) > 0:
            fig, ax = plt.subplots(figsize=(max(10, len(pivot_genes['niche'].unique()) * 1.5),
                                            max(6, len(pivot_genes['gene_human'].unique()) * 0.4)))

            # Bubble plot: x=niche, y=gene, size=n_drugs, color=lfc
            niches = sorted(pivot_genes['niche'].unique())
            genes = sorted(pivot_genes['gene_human'].unique(), key=lambda g:
                          pivot_genes[pivot_genes['gene_human'] == g]['logfoldchange'].max(), reverse=True)

            for i, gene in enumerate(genes):
                for j, niche in enumerate(niches):
                    subset = pivot_genes[(pivot_genes['gene_human'] == gene) & (pivot_genes['niche'] == niche)]
                    if len(subset) > 0:
                        lfc = subset['logfoldchange'].values[0]
                        n_drugs = druggable_data[(druggable_data['gene_human'] == gene) &
                                                  (druggable_data['niche'] == niche)]['drug'].nunique()
                        sc = ax.scatter(j, i, s=max(50, n_drugs * 80), c=lfc,
                                       cmap='RdYlBu_r', vmin=-2, vmax=4,
                                       edgecolors='black', linewidth=0.5, zorder=3)

            ax.set_xticks(range(len(niches)))
            ax.set_xticklabels(niches, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(genes)))
            ax.set_yticklabels(genes, fontsize=9)
            ax.set_xlabel('Altered Niche')
            ax.set_ylabel('Druggable Gene')
            ax.set_title(f'Druggable Genes in Altered Niches: {short}', fontsize=13, fontweight='bold')
            ax.set_xlim(-0.5, len(niches) - 0.5)
            ax.set_ylim(-0.5, len(genes) - 0.5)
            ax.grid(True, alpha=0.2)

            if 'sc' in locals():
                cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
                cbar.set_label('Log Fold Change', fontsize=9)

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, 'druggable_genes_bubble.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info("  Saved druggable_genes_bubble.png")

    # --- Visualization 2: Top drug-gene pairs bar chart ---
    if len(druggable_data) > 0:
        # Show top gene-drug pairs ranked by wilcoxon score
        top_pairs = (druggable_data.drop_duplicates(['gene_human', 'drug'])
                     .nlargest(20, 'wilcoxon_score'))

        if len(top_pairs) > 0:
            fig, ax = plt.subplots(figsize=(10, max(6, len(top_pairs) * 0.35)))
            labels = [f"{row['gene_human']} → {row['drug']}" for _, row in top_pairs.iterrows()]
            colors = ['#2a9d8f' if row['is_approved'] else '#e76f51' for _, row in top_pairs.iterrows()]

            y_pos = range(len(labels))
            ax.barh(y_pos, top_pairs['wilcoxon_score'].values, color=colors, edgecolor='white', height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Wilcoxon Score (gene vs atlas-compatible)')
            ax.set_title(f'Top Drug-Gene Pairs: {short}', fontsize=13, fontweight='bold')
            ax.invert_yaxis()

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='#2a9d8f', label='Approved Drug'),
                              Patch(facecolor='#e76f51', label='Not Approved / Investigational')]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, 'top_drug_gene_pairs.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info("  Saved top_drug_gene_pairs.png")

    # --- Save Open Targets AD drug details ---
    ot_rows = []
    for gene, info in ot_results.items():
        for ad_drug in info.get('ad_related_drugs', []):
            ot_rows.append({
                'gene': gene,
                'gene_name': info.get('approved_name', ''),
                'drug': ad_drug['drug_name'],
                'clinical_phase': ad_drug['phase'],
                'status': ad_drug['status'],
                'disease': ad_drug['disease'],
            })
    if ot_rows:
        ot_df = pd.DataFrame(ot_rows)
        ot_df.to_csv(os.path.join(out_dir, 'opentargets_ad_drugs.csv'), index=False)
        logger.info(f"  Saved opentargets_ad_drugs.csv ({len(ot_df)} rows)")

    return result_df, summary_df


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    dataset = args.dataset
    short = dataset.replace('.h5ad', '')

    niche_dir = os.path.join(args.discgal_dir, dataset, 'niche_analysis')
    out_dir = os.path.join(niche_dir, 'druggability')
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Druggability Analysis: {short}")
    logger.info(f"{'=' * 60}")

    # Load marker genes
    marker_path = os.path.join(niche_dir, 'marker_genes.csv')
    marker_df = pd.read_csv(marker_path)
    logger.info(f"Loaded {len(marker_df)} marker-gene-niche pairs")

    # Filter to significant genes
    sig_df = marker_df[marker_df['pval_adj'] < args.pval_cutoff].copy()
    logger.info(f"Significant markers (p < {args.pval_cutoff}): {len(sig_df)}")

    # Take top N genes per niche by score
    top_df = sig_df.sort_values('score', ascending=False).groupby('niche').head(args.top_n_genes)
    logger.info(f"Top {args.top_n_genes} per niche: {len(top_df)} gene-niche pairs")

    # Convert to human gene names
    unique_mouse_genes = top_df['gene'].unique().tolist()
    unique_human_genes = list(set(mouse_to_human(g) for g in unique_mouse_genes))
    logger.info(f"Unique genes to query: {len(unique_human_genes)}")

    # Stage 2: DGIdb
    dgidb_df = query_dgidb(unique_human_genes)

    # Stage 3: Open Targets (query druggable genes only)
    genes_to_query_ot = unique_human_genes
    ot_results = query_opentargets(genes_to_query_ot)

    # Stage 4: Integrate
    result_df, summary_df = integrate_and_visualize(
        top_df, dgidb_df, ot_results, out_dir, dataset
    )

    logger.info(f"\n✅ Druggability analysis complete. Results in {out_dir}/")


if __name__ == '__main__':
    main()
