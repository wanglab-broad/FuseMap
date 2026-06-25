#!/usr/bin/env python
"""
Visualize DiscGAL results: UMAP of disease embeddings overlaid on subsampled atlas.
Generates figures showing whether disease-specific niches are preserved.

Usage:
    python scripts/visualize_discgal.py

Output:
    output/discgal_disease_test/figures/
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# ============================================================
# Config
# ============================================================
OUTPUT_BASE = "./output/discgal_disease_test"
ATLAS_PATH = "./molCCF/latent_embeddings_all_single_final.pkl"
DISEASE_DIRS = [
    "13months-disease-replicate_1.h5ad",
    "adata_ad_cosmx.h5ad",
]
ATLAS_SUBSAMPLE = 20000   # subsample atlas to this many cells for UMAP
RANDOM_SEED = 42
FIG_DIR = os.path.join(OUTPUT_BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ============================================================
# Load data
# ============================================================
print("Loading atlas embeddings (subsampling)...")
with open(ATLAS_PATH, "rb") as f:
    atlas_embs = pickle.load(f)

# Subsample atlas: take random cells across all sections
atlas_all = np.concatenate(atlas_embs, axis=0)
print(f"  Total atlas cells: {atlas_all.shape[0]:,}")
if atlas_all.shape[0] > ATLAS_SUBSAMPLE:
    idx = np.random.choice(atlas_all.shape[0], ATLAS_SUBSAMPLE, replace=False)
    atlas_sub = atlas_all[idx]
else:
    atlas_sub = atlas_all
print(f"  Subsampled to: {atlas_sub.shape[0]:,}")
del atlas_all, atlas_embs  # free memory

# Load disease embeddings
disease_data = {}
for dname in DISEASE_DIRS:
    emb_path = os.path.join(OUTPUT_BASE, dname, "latent_embeddings_all_single_map.pkl")
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            emb = pickle.load(f)
        disease_data[dname] = np.concatenate(emb, axis=0)
        print(f"  {dname}: {disease_data[dname].shape[0]:,} cells")

# ============================================================
# Compute UMAP (atlas + all disease datasets together)
# ============================================================
print("\nComputing UMAP...")
all_embs = [atlas_sub]
labels = ["Normal Atlas"] * atlas_sub.shape[0]
dataset_ids = [0] * atlas_sub.shape[0]

for i, (dname, emb) in enumerate(disease_data.items(), start=1):
    all_embs.append(emb)
    short_name = dname.replace(".h5ad", "")
    labels.extend([short_name] * emb.shape[0])
    dataset_ids.extend([i] * emb.shape[0])

combined = np.concatenate(all_embs, axis=0)
print(f"  Combined shape: {combined.shape}")

# PCA to 20 dims first for speed
pca = PCA(n_components=min(20, combined.shape[1]))
combined_pca = pca.fit_transform(combined)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# UMAP
reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30, min_dist=0.3)
umap_coords = reducer.fit_transform(combined_pca)
print("  UMAP done.")

# Split back
dataset_ids = np.array(dataset_ids)
unique_datasets = ["Normal Atlas"] + [d.replace(".h5ad", "") for d in disease_data.keys()]

# ============================================================
# Plot 1: Atlas vs Disease overlay
# ============================================================
print("\nGenerating plots...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
colors = ["#cccccc", "#e63946", "#457b9d"]

# Plot atlas first (background)
mask_atlas = dataset_ids == 0
ax.scatter(
    umap_coords[mask_atlas, 0], umap_coords[mask_atlas, 1],
    c=colors[0], s=1, alpha=0.3, label=f"Normal Atlas (n={mask_atlas.sum():,})", rasterized=True
)

# Plot disease datasets on top
for i, dname in enumerate(disease_data.keys(), start=1):
    mask = dataset_ids == i
    short = dname.replace(".h5ad", "")
    ax.scatter(
        umap_coords[mask, 0], umap_coords[mask, 1],
        c=colors[i], s=3, alpha=0.6, label=f"{short} (n={mask.sum():,})", rasterized=True
    )

ax.set_title("DiscGAL: Disease Cells Mapped onto Normal Atlas", fontsize=14, fontweight='bold')
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(loc='upper right', fontsize=9, markerscale=5)
ax.set_aspect('equal')
plt.tight_layout()

path1 = os.path.join(FIG_DIR, "umap_atlas_vs_disease.png")
fig.savefig(path1, dpi=200, bbox_inches='tight')
print(f"  Saved: {path1}")
plt.close(fig)

# ============================================================
# Plot 2: Per-disease separate panels
# ============================================================
n_disease = len(disease_data)
fig, axes = plt.subplots(1, n_disease, figsize=(10 * n_disease, 9))
if n_disease == 1:
    axes = [axes]

for idx, (dname, emb) in enumerate(disease_data.items()):
    ax = axes[idx]
    short = dname.replace(".h5ad", "")

    # Atlas background
    ax.scatter(
        umap_coords[mask_atlas, 0], umap_coords[mask_atlas, 1],
        c='#e0e0e0', s=0.5, alpha=0.2, rasterized=True
    )

    # Disease cells colored by density (using local density as proxy for "atlas-compatibility")
    mask = dataset_ids == (idx + 1)
    disease_umap = umap_coords[mask]

    # Compute distance to nearest atlas neighbor in UMAP space (proxy for DiscGAL gate)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(umap_coords[mask_atlas])
    dists, _ = nn.kneighbors(disease_umap)
    mean_dist = dists.mean(axis=1)

    # Normalize for coloring
    vmin, vmax = np.percentile(mean_dist, [5, 95])
    mean_dist_clipped = np.clip(mean_dist, vmin, vmax)

    sc = ax.scatter(
        disease_umap[:, 0], disease_umap[:, 1],
        c=mean_dist_clipped, cmap='RdYlBu_r', s=3, alpha=0.7, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Distance to Atlas\n(blue=atlas-like, red=disease-specific)', fontsize=9)

    # Stats
    n_disease_specific = (mean_dist > np.percentile(mean_dist, 75)).sum()
    ax.set_title(f"{short}\n{emb.shape[0]:,} cells, ~{n_disease_specific:,} potentially disease-specific",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect('equal')

plt.tight_layout()
path2 = os.path.join(FIG_DIR, "umap_disease_distance.png")
fig.savefig(path2, dpi=200, bbox_inches='tight')
print(f"  Saved: {path2}")
plt.close(fig)

# ============================================================
# Plot 3: Distribution of atlas distances
# ============================================================
fig, axes = plt.subplots(1, n_disease, figsize=(6 * n_disease, 4))
if n_disease == 1:
    axes = [axes]

for idx, (dname, emb) in enumerate(disease_data.items()):
    ax = axes[idx]
    short = dname.replace(".h5ad", "")

    mask = dataset_ids == (idx + 1)
    disease_umap = umap_coords[mask]
    dists, _ = nn.kneighbors(disease_umap)
    mean_dist = dists.mean(axis=1)

    ax.hist(mean_dist, bins=50, color='#457b9d', alpha=0.7, edgecolor='white')
    p75 = np.percentile(mean_dist, 75)
    ax.axvline(p75, color='#e63946', linestyle='--', linewidth=2,
               label=f'75th pct = {p75:.2f}')
    ax.set_title(f"{short}: Distance to Atlas Distribution", fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean Distance to 5 Nearest Atlas Neighbors")
    ax.set_ylabel("Count")
    ax.legend()

plt.tight_layout()
path3 = os.path.join(FIG_DIR, "distance_histogram.png")
fig.savefig(path3, dpi=200, bbox_inches='tight')
print(f"  Saved: {path3}")
plt.close(fig)

print(f"\n✅ All figures saved to {FIG_DIR}/")
print("Key files:")
print(f"  1. {path1}  — Atlas vs Disease overlay")
print(f"  2. {path2}  — Disease cells colored by atlas distance")
print(f"  3. {path3}  — Distance distribution histogram")
