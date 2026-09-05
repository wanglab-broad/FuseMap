"""One-line plotting helpers for FuseMap embeddings (scanpy-style)."""

import numpy as np
import pandas as pd

__all__ = ["umap", "spatial_clusters", "transfer_heatmap"]


def umap(adata, by="file_name", n_neighbors=50, size=1.2, recompute=False, **kwargs):
    """UMAP of an embedding AnnData colored by one or more obs columns.

    Computes neighbors/UMAP once and caches them in the object.
    ``by`` may be a str or list of str.
    """
    import scanpy as sc

    if recompute or "X_umap" not in adata.obsm:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
        sc.tl.umap(adata)
    cols = [by] if isinstance(by, str) else list(by)
    return sc.pl.umap(adata, color=cols, size=size, wspace=0.45, **kwargs)


def spatial_clusters(adata, cluster_key="leiden", batch_key="file_name",
                     resolution=0.5, point_size=2, figsize_per_panel=(6, 6)):
    """Joint Leiden clustering of the embedding, plotted back into each
    section's spatial coordinates (one panel per section, shared colors)."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    if cluster_key not in adata.obs:
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=50, use_rep="X")
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
    adata.obs["x"] = pd.to_numeric(adata.obs["x"], errors="coerce")
    adata.obs["y"] = pd.to_numeric(adata.obs["y"], errors="coerce")
    batches = list(adata.obs[batch_key].astype(str).unique())
    cls = list(adata.obs[cluster_key].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    pal = {c: cmap(i % 20) for i, c in enumerate(sorted(cls))}
    w, h = figsize_per_panel
    fig, axes = plt.subplots(1, len(batches), figsize=(w * len(batches), h))
    axes = np.atleast_1d(axes)
    for ax, bname in zip(axes, batches):
        sub = adata.obs[adata.obs[batch_key].astype(str) == bname]
        for c in sorted(cls):
            m = sub[cluster_key].astype(str) == c
            if m.sum():
                ax.scatter(sub["x"][m], sub["y"][m], s=point_size, color=pal[c])
        ax.set_title(bname, fontsize=11)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    plt.tight_layout()
    return fig


def transfer_heatmap(gt, pred, normalize="index", figsize=(9, 8), cmap="Greens"):
    """Row-normalized confusion heatmap between ground-truth and
    transferred labels (rows = ground truth, columns = prediction)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    ct = pd.crosstab(pd.Series(gt, name="ground truth"),
                     pd.Series(pred, name="transferred"), normalize=normalize) * 100
    shared = [c for c in ct.index if c in ct.columns]
    ct = ct.loc[shared, shared + [c for c in ct.columns if c not in shared]]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ct, cmap=cmap, ax=ax, cbar_kws={"label": "% of row"})
    plt.tight_layout()
    return fig
