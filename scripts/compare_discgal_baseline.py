#!/usr/bin/env python
"""
Compare DiscGAL vs Baseline (original) training results side-by-side.
Generates UMAP comparison figures.

Usage:
    python scripts/compare_discgal_baseline.py
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap

# ============================================================
# Config
# ============================================================
DISCGAL_DIR = "./output/discgal_disease_test"
BASELINE_DIR = "./output/baseline_disease_test"
ATLAS_PATH = "./molCCF/latent_embeddings_all_single_final.pkl"
DISEASE_DIRS = [
    "13months-disease-replicate_1.h5ad",
    "adata_ad_cosmx.h5ad",
]
ATLAS_SUBSAMPLE = 20000
RANDOM_SEED = 42
FIG_DIR = "./output/comparison_figures"
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ============================================================
# Load atlas (subsampled)
# ============================================================
print("Loading atlas embeddings...")
with open(ATLAS_PATH, "rb") as f:
    atlas_embs = pickle.load(f)
atlas_all = np.concatenate(atlas_embs, axis=0)
idx = np.random.choice(atlas_all.shape[0], ATLAS_SUBSAMPLE, replace=False)
atlas_sub = atlas_all[idx]
del atlas_all, atlas_embs
print(f"  Atlas subsampled to {atlas_sub.shape[0]:,} cells")

# ============================================================
# Load disease embeddings for both methods
# ============================================================
def load_disease_embs(base_dir, disease_dirs):
    data = {}
    for dname in disease_dirs:
        path = os.path.join(base_dir, dname, "latent_embeddings_all_single_map.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                emb = pickle.load(f)
            data[dname] = np.concatenate(emb, axis=0)
            print(f"  {dname}: {data[dname].shape[0]:,} cells")
    return data

print("\nLoading DiscGAL embeddings...")
discgal_data = load_disease_embs(DISCGAL_DIR, DISEASE_DIRS)

print("\nLoading Baseline embeddings...")
baseline_data = load_disease_embs(BASELINE_DIR, DISEASE_DIRS)

# ============================================================
# For each disease dataset: joint UMAP of atlas + discgal + baseline
# ============================================================
for dname in DISEASE_DIRS:
    short = dname.replace(".h5ad", "")
    if dname not in discgal_data or dname not in baseline_data:
        print(f"\nSkipping {short}: missing data in one method")
        continue

    print(f"\n{'='*60}")
    print(f"Processing: {short}")
    print(f"{'='*60}")

    emb_discgal = discgal_data[dname]
    emb_baseline = baseline_data[dname]

    # Combine all for joint UMAP
    combined = np.concatenate([atlas_sub, emb_discgal, emb_baseline], axis=0)
    n_atlas = atlas_sub.shape[0]
    n_discgal = emb_discgal.shape[0]
    n_baseline = emb_baseline.shape[0]

    print(f"  Atlas: {n_atlas:,}, DiscGAL: {n_discgal:,}, Baseline: {n_baseline:,}")
    print(f"  Combined: {combined.shape}")

    # PCA + UMAP
    pca = PCA(n_components=min(20, combined.shape[1]))
    combined_pca = pca.fit_transform(combined)
    print(f"  PCA variance: {pca.explained_variance_ratio_.sum():.2%}")

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(combined_pca)
    print("  UMAP done.")

    atlas_coords = coords[:n_atlas]
    discgal_coords = coords[n_atlas:n_atlas + n_discgal]
    baseline_coords = coords[n_atlas + n_discgal:]

    # ----------------------------------------------------------
    # Figure 1: Side-by-side overlay
    # ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))

    for ax, disease_coords, method_name, color in [
        (ax1, baseline_coords, "Baseline (Original)", "#e63946"),
        (ax2, discgal_coords, "DiscGAL", "#2a9d8f"),
    ]:
        ax.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
                   c='#e0e0e0', s=0.5, alpha=0.2, label=f"Atlas (n={n_atlas:,})", rasterized=True)
        ax.scatter(disease_coords[:, 0], disease_coords[:, 1],
                   c=color, s=2, alpha=0.5, label=f"{method_name} (n={disease_coords.shape[0]:,})", rasterized=True)
        ax.set_title(f"{method_name}\n{short}", fontsize=14, fontweight='bold')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(loc='upper right', fontsize=9, markerscale=5)
        ax.set_aspect('equal')

    plt.suptitle(f"Baseline vs DiscGAL: {short}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"comparison_overlay_{short}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

    # ----------------------------------------------------------
    # Figure 2: Side-by-side distance coloring
    # ----------------------------------------------------------
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(atlas_coords)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))

    # Compute shared color range
    dists_bl, _ = nn.kneighbors(baseline_coords)
    dists_dg, _ = nn.kneighbors(discgal_coords)
    mean_bl = dists_bl.mean(axis=1)
    mean_dg = dists_dg.mean(axis=1)
    vmin = min(np.percentile(mean_bl, 5), np.percentile(mean_dg, 5))
    vmax = max(np.percentile(mean_bl, 95), np.percentile(mean_dg, 95))

    for ax, disease_coords, mean_dist, method_name in [
        (ax1, baseline_coords, mean_bl, "Baseline (Original)"),
        (ax2, discgal_coords, mean_dg, "DiscGAL"),
    ]:
        ax.scatter(atlas_coords[:, 0], atlas_coords[:, 1],
                   c='#e0e0e0', s=0.5, alpha=0.15, rasterized=True)
        dist_clipped = np.clip(mean_dist, vmin, vmax)
        sc = ax.scatter(disease_coords[:, 0], disease_coords[:, 1],
                        c=dist_clipped, cmap='RdYlBu_r', s=3, alpha=0.7,
                        vmin=vmin, vmax=vmax, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
        cbar.set_label('Distance to Atlas\n(blue=atlas-like, red=disease-specific)', fontsize=9)

        n_far = (mean_dist > np.percentile(mean_dist, 75)).sum()
        ax.set_title(f"{method_name}\n~{n_far:,} potentially disease-specific", fontsize=13, fontweight='bold')
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_aspect('equal')

    plt.suptitle(f"Atlas Distance: {short}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"comparison_distance_{short}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

    # ----------------------------------------------------------
    # Figure 3: Histogram comparison (overlaid)
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_bl, bins=50, alpha=0.5, color='#e63946', label=f'Baseline (median={np.median(mean_bl):.2f})', edgecolor='white')
    ax.hist(mean_dg, bins=50, alpha=0.5, color='#2a9d8f', label=f'DiscGAL (median={np.median(mean_dg):.2f})', edgecolor='white')
    ax.set_title(f"Distance Distribution Comparison: {short}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Distance to 5 Nearest Atlas Neighbors")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"comparison_histogram_{short}.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)

print(f"\n✅ All comparison figures saved to {FIG_DIR}/")
