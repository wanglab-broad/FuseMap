#!/usr/bin/env python
"""
Step ② Identify Altered Niches
==============================

Identifies disease-specific spatial niches from DiscGAL-mapped embeddings
by comparing disease cells to the normal atlas in latent space.

Pipeline:
  Stage 1: Atlas distance scoring (KNN in 64-dim latent)
  Stage 2: Cell classification (atlas-compatible / altered / novel)
  Stage 3: Leiden clustering on altered cells + DEG analysis
  Stage 4: Visualization (UMAP, spatial map, dotplot)

Usage:
    python scripts/identify_altered_niches.py \
        --discgal_dir ./output/discgal_disease_test \
        --atlas_path ./molCCF/latent_embeddings_all_single_final.pkl \
        --raw_data_dir ./example_data/application_data/disease \
        --dataset 13months-disease-replicate_1.h5ad

Output:
    <discgal_dir>/<dataset>/niche_analysis/
"""
import argparse
import os
import pickle
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
K_NEIGHBORS = 15         # for atlas distance computation
ATLAS_SUBSAMPLE = 50000  # subsample atlas for KNN
LEIDEN_RESOLUTION = 0.5  # clustering resolution for altered cells
N_TOP_GENES = 50         # DEG genes per cluster
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='Identify altered niches from DiscGAL-mapped embeddings')
    parser.add_argument('--discgal_dir', type=str, default='./output/discgal_disease_test',
                        help='Base output directory from DiscGAL training')
    parser.add_argument('--atlas_path', type=str, default='./molCCF/latent_embeddings_all_single_final.pkl',
                        help='Path to atlas single-cell embedding pickle')
    parser.add_argument('--raw_data_dir', type=str, default='./example_data/application_data/disease',
                        help='Directory containing original h5ad files')
    parser.add_argument('--dataset', type=str, default='13months-disease-replicate_1.h5ad',
                        help='Dataset name (directory name in discgal_dir)')
    parser.add_argument('--k_neighbors', type=int, default=K_NEIGHBORS)
    parser.add_argument('--atlas_subsample', type=int, default=ATLAS_SUBSAMPLE)
    parser.add_argument('--leiden_resolution', type=float, default=LEIDEN_RESOLUTION)
    return parser.parse_args()


# ============================================================
# Stage 1: Atlas Distance Scoring
# ============================================================
def compute_atlas_distances(disease_emb, atlas_path, k=15, atlas_subsample=50000):
    """Compute per-cell distance to nearest atlas neighbors in latent space."""
    logger.info("Stage 1: Computing atlas distances...")

    # Load & subsample atlas
    with open(atlas_path, 'rb') as f:
        atlas_embs = pickle.load(f)
    atlas_all = np.concatenate(atlas_embs, axis=0)
    logger.info(f"  Atlas total: {atlas_all.shape[0]:,} cells")

    np.random.seed(RANDOM_SEED)
    if atlas_all.shape[0] > atlas_subsample:
        idx = np.random.choice(atlas_all.shape[0], atlas_subsample, replace=False)
        atlas_sub = atlas_all[idx]
    else:
        atlas_sub = atlas_all
    logger.info(f"  Atlas subsampled: {atlas_sub.shape[0]:,} cells")

    # Compute atlas internal distances (for threshold calibration)
    nn_atlas = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    nn_atlas.fit(atlas_sub)
    atlas_self_dists, _ = nn_atlas.kneighbors(atlas_sub)
    atlas_self_dists = atlas_self_dists[:, 1:]  # exclude self
    atlas_mean_dists = atlas_self_dists.mean(axis=1)

    atlas_mu = atlas_mean_dists.mean()
    atlas_sigma = atlas_mean_dists.std()
    logger.info(f"  Atlas internal distances: μ={atlas_mu:.4f}, σ={atlas_sigma:.4f}")

    # Compute disease-to-atlas distances
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(atlas_sub)
    disease_dists, disease_indices = nn.kneighbors(disease_emb)
    disease_mean_dists = disease_dists.mean(axis=1)

    del atlas_all, atlas_embs  # free memory

    return disease_mean_dists, atlas_mu, atlas_sigma, atlas_sub


# ============================================================
# Stage 2: Cell Classification
# ============================================================
def classify_cells(distances, atlas_mu, atlas_sigma):
    """Classify cells into atlas-compatible, altered, or novel."""
    logger.info("Stage 2: Classifying cells...")

    threshold_altered = atlas_mu + 1.0 * atlas_sigma
    threshold_novel = atlas_mu + 3.0 * atlas_sigma

    categories = np.where(
        distances <= threshold_altered, 'atlas-compatible',
        np.where(distances <= threshold_novel, 'altered', 'novel')
    )

    n_compat = (categories == 'atlas-compatible').sum()
    n_altered = (categories == 'altered').sum()
    n_novel = (categories == 'novel').sum()

    logger.info(f"  Thresholds: altered > {threshold_altered:.4f}, novel > {threshold_novel:.4f}")
    logger.info(f"  Atlas-compatible: {n_compat:,} ({n_compat / len(categories):.1%})")
    logger.info(f"  Altered:          {n_altered:,} ({n_altered / len(categories):.1%})")
    logger.info(f"  Novel:            {n_novel:,} ({n_novel / len(categories):.1%})")

    return categories, threshold_altered, threshold_novel


# ============================================================
# Stage 3: Clustering & DEG
# ============================================================
def cluster_and_deg(ad_disease, ad_raw, categories, leiden_res=0.5, n_top_genes=50):
    """Cluster altered+novel cells and find marker genes."""
    logger.info("Stage 3: Clustering & DEG analysis...")

    # Add classification to the embedding AnnData
    ad_disease.obs['niche_category'] = pd.Categorical(
        categories, categories=['atlas-compatible', 'altered', 'novel']
    )

    # Subset to altered + novel cells
    mask_altered = ad_disease.obs['niche_category'].isin(['altered', 'novel'])
    n_altered = mask_altered.sum()

    if n_altered < 10:
        logger.warning("  Too few altered/novel cells for clustering. Skipping.")
        ad_disease.obs['altered_niche'] = 'none'
        return ad_disease, None

    logger.info(f"  Clustering {n_altered:,} altered+novel cells...")

    # Create a temporary AnnData for clustering (using latent embeddings as X)
    ad_altered = ad_disease[mask_altered].copy()

    # Build neighbor graph in latent space and cluster
    sc.pp.neighbors(ad_altered, use_rep='X', n_neighbors=15)
    sc.tl.leiden(ad_altered, resolution=leiden_res, key_added='altered_niche')

    n_clusters = ad_altered.obs['altered_niche'].nunique()
    logger.info(f"  Found {n_clusters} altered niche clusters")
    logger.info(f"  Cluster sizes:\n{ad_altered.obs['altered_niche'].value_counts().to_string()}")

    # Transfer cluster labels back
    ad_disease.obs['altered_niche'] = 'atlas-compatible'
    ad_disease.obs.loc[mask_altered, 'altered_niche'] = [
        f"niche_{x}" for x in ad_altered.obs['altered_niche']
    ]

    # DEG analysis using original expression
    # Align raw data with disease embeddings
    if ad_raw is not None and ad_raw.shape[0] == ad_disease.shape[0]:
        logger.info("  Running DEG analysis...")
        ad_raw_copy = ad_raw.copy()
        ad_raw_copy.obs['niche_category'] = ad_disease.obs['niche_category'].values
        ad_raw_copy.obs['altered_niche'] = ad_disease.obs['altered_niche'].values

        # Normalize for DEG
        sc.pp.normalize_total(ad_raw_copy, target_sum=1e4)
        sc.pp.log1p(ad_raw_copy)

        # DEG: altered niches vs atlas-compatible
        try:
            sc.tl.rank_genes_groups(
                ad_raw_copy,
                groupby='altered_niche',
                reference='atlas-compatible',
                method='wilcoxon',
                n_genes=n_top_genes,
            )
            deg_result = ad_raw_copy.uns['rank_genes_groups']

            # Extract to DataFrame
            deg_rows = []
            for group in deg_result['names'].dtype.names:
                if group == 'atlas-compatible':
                    continue
                for i in range(min(n_top_genes, len(deg_result['names'][group]))):
                    deg_rows.append({
                        'niche': group,
                        'gene': deg_result['names'][group][i],
                        'logfoldchange': deg_result['logfoldchanges'][group][i],
                        'pval_adj': deg_result['pvals_adj'][group][i],
                        'score': deg_result['scores'][group][i],
                    })
            deg_df = pd.DataFrame(deg_rows)
            logger.info(f"  DEG analysis complete: {len(deg_df)} gene-niche pairs")
            return ad_disease, deg_df
        except Exception as e:
            logger.warning(f"  DEG analysis failed: {e}")
            return ad_disease, None
    else:
        logger.warning("  Raw data not aligned, skipping DEG")
        return ad_disease, None


# ============================================================
# Stage 4: Visualization
# ============================================================
def generate_visualizations(ad_disease, atlas_sub, deg_df, out_dir, dataset_name):
    """Generate UMAP, spatial map, and dotplot figures."""
    logger.info("Stage 4: Generating visualizations...")

    short = dataset_name.replace('.h5ad', '')

    # --- UMAP (disease + atlas subsample) ---
    logger.info("  Computing UMAP...")
    combined = np.concatenate([atlas_sub, ad_disease.X], axis=0)

    pca = PCA(n_components=min(20, combined.shape[1]))
    combined_pca = pca.fit_transform(combined)

    import umap
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(combined_pca)

    atlas_coords = coords[:atlas_sub.shape[0]]
    disease_coords = coords[atlas_sub.shape[0]:]

    # Store UMAP in adata
    ad_disease.obsm['X_umap'] = disease_coords

    # --- Figure 1: UMAP niche categories ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
               c='#e8e8e8', s=0.3, alpha=0.15, label='Normal Atlas', rasterized=True)

    cat_colors = {
        'atlas-compatible': '#4a90d9',
        'altered': '#e6a817',
        'novel': '#d63031',
    }
    for cat, color in cat_colors.items():
        mask = ad_disease.obs['niche_category'] == cat
        n = mask.sum()
        if n > 0:
            ax.scatter(disease_coords[mask, 0], disease_coords[mask, 1],
                       c=color, s=3, alpha=0.6, label=f'{cat} (n={n:,})', rasterized=True)

    ax.set_title(f'Niche Categories: {short}', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=10, markerscale=5)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'umap_niche_categories.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved umap_niche_categories.png")

    # --- Figure 2: UMAP altered niche clusters ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
               c='#e8e8e8', s=0.3, alpha=0.15, rasterized=True)

    niche_labels = ad_disease.obs['altered_niche'].unique()
    cmap = plt.cm.get_cmap('tab20', len(niche_labels))
    for i, niche in enumerate(sorted(niche_labels)):
        mask = ad_disease.obs['altered_niche'] == niche
        n = mask.sum()
        color = '#cccccc' if niche == 'atlas-compatible' else cmap(i)
        alpha = 0.1 if niche == 'atlas-compatible' else 0.7
        s = 0.5 if niche == 'atlas-compatible' else 4
        ax.scatter(disease_coords[mask, 0], disease_coords[mask, 1],
                   c=[color], s=s, alpha=alpha, label=f'{niche} ({n:,})', rasterized=True)

    ax.set_title(f'Altered Niche Clusters: {short}', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=8, markerscale=4, ncol=2)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'umap_altered_clusters.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved umap_altered_clusters.png")

    # --- Figure 3: Spatial niche map ---
    has_coords = 'x' in ad_disease.obs.columns and 'y' in ad_disease.obs.columns
    if has_coords:
        try:
            x = ad_disease.obs['x'].astype(float).values
            y = ad_disease.obs['y'].astype(float).values

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Left: niche categories on spatial map
            for cat, color in cat_colors.items():
                mask = ad_disease.obs['niche_category'] == cat
                if mask.sum() > 0:
                    ax1.scatter(x[mask], y[mask], c=color, s=1, alpha=0.5,
                               label=f'{cat} ({mask.sum():,})', rasterized=True)
            ax1.set_title(f'Spatial Niche Map: {short}', fontsize=13, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.legend(fontsize=9, markerscale=5)
            ax1.set_aspect('equal')
            ax1.invert_yaxis()

            # Right: atlas distance on spatial map
            dists = ad_disease.obs['atlas_distance'].astype(float).values
            vmin, vmax = np.percentile(dists, [5, 95])
            sc_plot = ax2.scatter(x, y, c=np.clip(dists, vmin, vmax),
                                  cmap='RdYlBu_r', s=1, alpha=0.5, rasterized=True)
            plt.colorbar(sc_plot, ax=ax2, shrink=0.6, label='Atlas Distance')
            ax2.set_title(f'Atlas Distance (spatial): {short}', fontsize=13, fontweight='bold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
            ax2.invert_yaxis()

            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, 'spatial_niche_map.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info("  Saved spatial_niche_map.png")
        except Exception as e:
            logger.warning(f"  Spatial map failed: {e}")
    else:
        logger.info("  No spatial coordinates found, skipping spatial map")

    # --- Figure 4: Top marker genes dotplot ---
    if deg_df is not None and len(deg_df) > 0:
        try:
            sig_genes = deg_df[deg_df['pval_adj'] < 0.05].copy()
            if len(sig_genes) > 0:
                # Get top 5 genes per niche
                top_genes = (sig_genes.sort_values('score', ascending=False)
                             .groupby('niche').head(5))
                gene_list = top_genes['gene'].unique().tolist()

                fig, ax = plt.subplots(figsize=(max(12, len(gene_list) * 0.6), 4))
                niches = top_genes['niche'].unique()
                for i, niche in enumerate(niches):
                    niche_genes = top_genes[top_genes['niche'] == niche]
                    ax.barh(
                        [f"{g} ({niche})" for g in niche_genes['gene']],
                        niche_genes['score'],
                        alpha=0.7, label=niche
                    )
                ax.set_xlabel('Wilcoxon Score')
                ax.set_title(f'Top Marker Genes per Altered Niche: {short}', fontsize=13, fontweight='bold')
                ax.legend(fontsize=9)
                plt.tight_layout()
                fig.savefig(os.path.join(out_dir, 'marker_genes_barplot.png'), dpi=200, bbox_inches='tight')
                plt.close(fig)
                logger.info("  Saved marker_genes_barplot.png")
        except Exception as e:
            logger.warning(f"  Marker gene plot failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    dataset = args.dataset
    short = dataset.replace('.h5ad', '')

    data_dir = os.path.join(args.discgal_dir, dataset)
    out_dir = os.path.join(data_dir, 'niche_analysis')
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Identify Altered Niches: {short}")
    logger.info(f"{'=' * 60}")

    # Load disease cell embeddings
    emb_path = os.path.join(data_dir, 'ad_celltype_embedding.h5ad')
    logger.info(f"Loading disease embeddings: {emb_path}")
    ad_disease = sc.read_h5ad(emb_path)
    logger.info(f"  {ad_disease.shape[0]:,} cells, {ad_disease.shape[1]} latent dims")

    # Load raw expression data for DEG
    raw_path = os.path.join(args.raw_data_dir, dataset)
    ad_raw = None
    if os.path.exists(raw_path):
        logger.info(f"Loading raw expression: {raw_path}")
        ad_raw = sc.read_h5ad(raw_path)
        # Handle coords
        if 'x' not in ad_raw.obs.columns:
            try:
                ad_raw.obs['x'] = ad_raw.obs['col']
                ad_raw.obs['y'] = ad_raw.obs['row']
            except:
                try:
                    ad_raw.obs['x'] = ad_raw.obsm['spatial'][:, 0]
                    ad_raw.obs['y'] = ad_raw.obsm['spatial'][:, 1]
                except:
                    pass
        logger.info(f"  Raw: {ad_raw.shape[0]:,} cells, {ad_raw.shape[1]:,} genes")

        # Check alignment
        if ad_raw.shape[0] != ad_disease.shape[0]:
            logger.warning(f"  Cell count mismatch: raw={ad_raw.shape[0]}, embedded={ad_disease.shape[0]}")
            ad_raw = None
    else:
        logger.warning(f"  Raw data not found at {raw_path}")

    # ── Stage 1 ──
    distances, atlas_mu, atlas_sigma, atlas_sub = compute_atlas_distances(
        ad_disease.X, args.atlas_path, k=args.k_neighbors, atlas_subsample=args.atlas_subsample
    )
    ad_disease.obs['atlas_distance'] = distances

    # ── Stage 2 ──
    categories, thresh_altered, thresh_novel = classify_cells(distances, atlas_mu, atlas_sigma)

    # ── Stage 3 ──
    ad_disease, deg_df = cluster_and_deg(
        ad_disease, ad_raw, categories, leiden_res=args.leiden_resolution
    )

    # ── Stage 4 ──
    generate_visualizations(ad_disease, atlas_sub, deg_df, out_dir, dataset)

    # ── Save outputs ──
    logger.info("Saving outputs...")

    # Annotated AnnData
    for col in ad_disease.obs.columns:
        ad_disease.obs[col] = ad_disease.obs[col].astype(str)
    ad_disease.write_h5ad(os.path.join(out_dir, 'ad_niche_annotated.h5ad'))
    logger.info(f"  Saved ad_niche_annotated.h5ad")

    # DEG table
    if deg_df is not None:
        deg_df.to_csv(os.path.join(out_dir, 'marker_genes.csv'), index=False)
        logger.info(f"  Saved marker_genes.csv ({len(deg_df)} rows)")

    # Summary table
    summary_rows = []
    for niche in sorted(ad_disease.obs['altered_niche'].unique()):
        mask = ad_disease.obs['altered_niche'] == niche
        n = mask.sum()
        mean_dist = ad_disease.obs.loc[mask, 'atlas_distance'].astype(float).mean()

        top_genes = ''
        if deg_df is not None:
            niche_genes = deg_df[(deg_df['niche'] == niche) & (deg_df['pval_adj'] < 0.05)]
            if len(niche_genes) > 0:
                top_genes = ', '.join(niche_genes.nlargest(5, 'score')['gene'].tolist())

        summary_rows.append({
            'niche': niche,
            'n_cells': n,
            'pct_total': f"{n / ad_disease.shape[0]:.1%}",
            'mean_atlas_distance': f"{mean_dist:.4f}",
            'top_marker_genes': top_genes,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, 'niche_summary.csv'), index=False)
    logger.info(f"  Saved niche_summary.csv")
    logger.info(f"\n{summary_df.to_string(index=False)}")

    logger.info(f"\n✅ All outputs saved to {out_dir}/")


if __name__ == '__main__':
    main()
