"""Integration-quality metrics.

All metrics operate on embedding AnnData objects produced by
:func:`fusemap.integrate` / :func:`fusemap.map_to_reference`
(64-dim latent in ``X``).
"""

import numpy as np
import pandas as pd

__all__ = ["ilisi", "transfer_accuracy", "spatial_coherence"]


def _knn_indices(X, k, chunk=4096, device=None):
    import torch

    t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t = t.to(device)
    idx_all = []
    for s in range(0, t.shape[0], chunk):
        d = torch.cdist(t[s:s + chunk], t)
        for j in range(d.shape[0]):
            d[j, s + j] = float("inf")
        idx_all.append(d.topk(k, largest=False).indices.cpu().numpy())
    return np.vstack(idx_all)


def ilisi(adata, batch_key="file_name", k=50, use_rep=None):
    """Batch-mixing iLISI in [0, 1]: 0 = fully separated, 1 = perfectly mixed.

    Normalized inverse Simpson index of batch composition among each cell's
    ``k`` nearest latent neighbors, averaged over cells.
    """
    X = adata.obsm[use_rep] if use_rep else adata.X
    codes = pd.Categorical(adata.obs[batch_key]).codes
    n_batch = codes.max() + 1
    if n_batch < 2:
        raise ValueError("need at least two batches")
    idx = _knn_indices(X, k)
    p = np.stack([(codes[idx] == j).mean(1) for j in range(n_batch)], axis=1)
    simpson = (p ** 2).sum(1)
    return float(((1.0 / simpson) - 1).mean() / (n_batch - 1))


def transfer_accuracy(adata, label_key, batch_key="file_name",
                      reference=None, k=30, balanced=True, use_rep=None):
    """kNN label-transfer accuracy from reference batch(es) to the rest.

    Parameters
    ----------
    reference
        Value (or list of values) of ``obs[batch_key]`` to use as the labeled
        reference; all other cells with a ground-truth label are evaluated.
    balanced
        Average per-class recall instead of plain accuracy — plain kNN
        accuracy silently ignores rare cell types (recommended: True).

    Returns
    -------
    dict with ``overall`` and ``per_class`` (pandas Series).
    """
    import torch

    X = np.asarray(adata.obsm[use_rep] if use_rep else adata.X, dtype=np.float32)
    lab = adata.obs[label_key].astype(str).values
    b = adata.obs[batch_key].astype(str).values
    if reference is None:
        raise ValueError("specify reference batch value(s)")
    ref_vals = [reference] if isinstance(reference, str) else list(reference)
    is_ref = np.isin(b, ref_vals) & (lab != "nan")
    is_qry = ~np.isin(b, ref_vals) & (lab != "nan")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xr = torch.as_tensor(X[is_ref]).to(device)
    Xq = torch.as_tensor(X[is_qry]).to(device)
    ref_lab = lab[is_ref]
    pred = []
    for s in range(0, Xq.shape[0], 4096):
        idx = torch.cdist(Xq[s:s + 4096], Xr).topk(k, largest=False).indices.cpu().numpy()
        pred.append(pd.DataFrame(ref_lab[idx]).mode(axis=1)[0].values)
    pred = np.concatenate(pred)
    gt = lab[is_qry]
    per_class = pd.DataFrame({"gt": gt, "ok": pred == gt}).groupby("gt")["ok"].mean()
    overall = float(per_class.mean()) if balanced else float((pred == gt).mean())
    return {"overall": overall, "per_class": per_class}


def spatial_coherence(adata, label_key, batch_key=None, batch=None, k=15):
    """Spatial smoothness of a (predicted) label: fraction of each cell's
    spatial k-nearest neighbors sharing its label. Label-free quality proxy
    for annotation transfer when no ground truth exists.
    """
    from scipy.spatial import cKDTree

    obs = adata.obs
    if batch_key and batch is not None:
        obs = obs[obs[batch_key] == batch]
    xy = np.c_[pd.to_numeric(obs["x"]), pd.to_numeric(obs["y"])]
    lab = obs[label_key].astype(str).values
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return float((lab[idx[:, 1:]] == lab[:, None]).mean())
