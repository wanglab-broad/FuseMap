"""Bead deconvolution against a saved, unchanged FuseMap reference."""

import hashlib
import logging
import os
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans

from fusemap.data.graph import preprocess_adata, construct_graph, preprocess_adj_sparse
from fusemap.postprocess.mixtures import fit_mixtures


def resolve_files(tokens, candidates, argument):
    """Resolve explicitly declared roles; never infer resolution from filenames."""
    if not isinstance(tokens, str) or not tokens.strip():
        raise ValueError(f"{argument} must contain comma-separated filename substrings")
    selected = []
    for token in tokens.split(","):
        token = token.strip()
        hits = [name for name in candidates if token and token in name]
        if len(hits) != 1 or hits[0] in selected:
            raise ValueError(f"{argument}: {token!r} must match exactly one distinct file; matches={hits}")
        selected.append(hits[0])
    # Preserve training order, independently of the order of the user's tokens.
    return [name for name in candidates if name in selected]


def model_fingerprint(reference_dir):
    path = Path(reference_dir) / "trained_model" / "FuseMap_final_model_final.pt"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_output_path(output, reference_dir):
    output, reference = Path(output).resolve(), Path(reference_dir).resolve()
    if output == reference or reference in output.parents:
        raise ValueError("Mapping outputs must be outside pretrain_model_path to preserve the reference")


def save_npz(path, **arrays):
    """Replace a single artifact only after it has been completely written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def normalized_expression(adata):
    """Use the same gene deduplication, filtering and scaling as Stage-A."""
    result = preprocess_adata([adata.copy()], 1)[0].copy()
    values = result.X.data if sp.issparse(result.X) else np.asarray(result.X)
    if result.n_obs == 0 or result.n_vars == 0:
        raise ValueError("No observations or genes remain after FuseMap preprocessing")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Stage-B requires finite, nonnegative expression")
    return result


def align_observations(adata, names, context):
    names = np.asarray(names, dtype=str)
    if not adata.obs_names.is_unique or len(np.unique(names)) != len(names):
        raise ValueError(f"{context}: observation names must be unique within each dataset")
    if len(names) != adata.n_obs or set(names) != set(adata.obs_names):
        raise ValueError(f"{context}: expression observations do not match the saved embeddings")
    return adata[names].copy()


def build_reference_signatures(reference_dir, data_dir, sig_ref,
                               n_archetypes=40, min_cells=20):
    """Build signatures from saved cell embeddings and original reference X.

    Only declared single-cell reference sections participate. No encoder,
    decoder or reference embedding is trained or updated.
    """
    if n_archetypes < 1 or min_cells < 1:
        raise ValueError("n_archetypes and min_cells must be positive")
    fingerprint = model_fingerprint(reference_dir)
    ref = ad.read_h5ad(Path(reference_dir) / "ad_celltype_embedding.h5ad", backed="r")
    try:
        candidates = list(dict.fromkeys(ref.obs["file_name"].astype(str)))
        selected = resolve_files(sig_ref, candidates, "sig_ref")
        matrices, expression = [], []
        for name in selected:
            if Path(name).name != name:
                raise ValueError(f"Reference file_name must be a basename: {name!r}")
            rows = np.flatnonzero(ref.obs["file_name"].astype(str).values == name)
            block = ref[rows].to_memory()
            matrices.append(np.asarray(block.X, dtype=np.float32))
            raw = ad.read_h5ad(Path(data_dir) / name)
            normalized = normalized_expression(raw)
            expression.append(align_observations(normalized, block.obs_names, name))
    finally:
        ref.file.close()
    latent = np.concatenate(matrices)
    if not np.isfinite(latent).all() or len(latent) < n_archetypes:
        raise ValueError("Reference must have finite embeddings and at least n_archetypes cells")
    km = MiniBatchKMeans(n_clusters=n_archetypes, random_state=0, batch_size=4096, n_init=3)
    labels = km.fit_predict(latent)
    counts = np.bincount(labels, minlength=n_archetypes)
    kept = np.flatnonzero(counts >= min_cells)
    if not len(kept):
        raise ValueError("No reference archetypes have min_cells members; use a larger reference or lower min_cells")
    genes = sorted(set().union(*(set(x.var_names) for x in expression)))
    gene_index = {gene: i for i, gene in enumerate(genes)}
    # Pool expression only over references in which each gene was measured.
    sums = np.zeros((len(kept), len(genes)), dtype=np.float64)
    denominators = np.zeros_like(sums)
    offset = 0
    for x in expression:
        local_labels = labels[offset:offset + x.n_obs]
        offset += x.n_obs
        columns = np.array([gene_index[g] for g in x.var_names])
        for j, k in enumerate(kept):
            members = np.flatnonzero(local_labels == k)
            if len(members):
                sums[j, columns] += np.asarray(x.X[members].astype(np.float64).sum(axis=0)).ravel()
                denominators[j, columns] += len(members)
    signatures = np.divide(sums, denominators, out=np.zeros_like(sums), where=denominators > 0)
    logging.info("[Stage-B] prepared %d frozen archetypes from %s; %d genes",
                 len(kept), selected, len(genes))
    return dict(format_version=np.array(1), reference_model_sha256=np.array(fingerprint),
                reference_files=np.asarray(selected), genes=np.asarray(genes),
                archetype_centers=km.cluster_centers_[kept].astype(np.float32),
                signatures=signatures.astype(np.float32), kept_archetypes=kept,
                member_counts=counts[kept], signature_member_counts=denominators,
                n_archetypes=np.array(n_archetypes), min_cells=np.array(min_cells))


def load_reference_signatures(path, reference_dir, sig_ref=None):
    with np.load(path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    required = {"format_version", "reference_model_sha256", "reference_files", "genes",
                "archetype_centers", "signatures", "kept_archetypes"}
    if not required.issubset(bundle) or int(bundle["format_version"]) != 1:
        raise ValueError("Unsupported or incomplete reference signatures artifact")
    if str(bundle["reference_model_sha256"]) != model_fingerprint(reference_dir):
        raise ValueError("Reference signatures belong to a different pretrained model")
    if sig_ref and resolve_files(sig_ref, list(bundle["reference_files"]), "sig_ref") != list(bundle["reference_files"]):
        raise ValueError("sig_ref differs from the references in the signatures artifact")
    s, c, genes = bundle["signatures"], bundle["archetype_centers"], bundle["genes"]
    if (s.ndim != 2 or c.ndim != 2 or s.shape != (len(c), len(genes))
            or len(c) == 0 or len(set(genes)) != len(genes)
            or not np.isfinite(s).all() or not np.isfinite(c).all() or (s < 0).any()):
        raise ValueError("Invalid reference signature dimensions or values")
    return bundle


def query_signatures(query, bundle):
    positions = {gene: i for i, gene in enumerate(bundle["genes"])}
    index = np.array([positions.get(gene, -1) for gene in query.var_names])
    covered = index >= 0
    if not covered.any():
        raise ValueError("No shared genes between query and reference signatures")
    signatures = np.zeros((len(bundle["archetype_centers"]), query.n_vars), dtype=np.float32)
    signatures[:, covered] = bundle["signatures"][:, index[covered]]
    if not (signatures[:, covered] > 0).any():
        raise ValueError("Shared genes have no reference signature signal")
    logging.info("[Stage-B] %d/%d query genes covered by reference signatures",
                 int(covered.sum()), query.n_vars)
    return signatures, covered


def deconvolve_mapped_query(query, output_dir, bundle, entropy_weight=5e-4,
                            n_steps=800, device=None):
    """Replace query embeddings with pi @ frozen centers; preserve mapped originals."""
    output_dir = Path(output_dir)
    embeddings = {level: ad.read_h5ad(output_dir / f"ad_{level}_embedding.h5ad")
                  for level in ("celltype", "tissueregion")}
    cell = embeddings["celltype"]
    query = align_observations(query, cell.obs_names, "query")
    tissue = embeddings["tissueregion"]
    if not np.array_equal(cell.obs_names, tissue.obs_names):
        raise ValueError("Cell and tissue embedding observation order differs")
    centers = bundle["archetype_centers"]
    if cell.n_vars != centers.shape[1] or tissue.n_vars != centers.shape[1]:
        raise ValueError("Mapped and reference embedding dimensions differ")
    # Construct the spatial readout before fitting, so invalid coordinates fail early.
    construct_graph([query], 1, ["delaunay"], ["ST"])
    preprocess_adj_sparse([query], 1, ["ST"])
    adjacency = sp.csr_matrix(query.obsm["adj_normalized"])
    if adjacency.shape != (query.n_obs, query.n_obs) or not np.isfinite(adjacency.data).all():
        raise ValueError("Invalid query spatial graph")
    signatures, covered = query_signatures(query, bundle)
    fit = fit_mixtures(query.X, signatures, covered, entropy_weight=entropy_weight,
                       n_steps=n_steps, device=device)
    pi = fit["pi"]
    z = pi @ centers
    spatial = np.asarray(adjacency.T.dot(z), dtype=np.float32)
    metadata = dict(mode="frozen_reference_mapping", reference_files=bundle["reference_files"],
                    reference_model_sha256=str(bundle["reference_model_sha256"]),
                    entropy_weight=float(entropy_weight), n_genes_used=int(covered.sum()),
                    n_steps=int(fit["n_steps"]), weight_meaning="archetype mixture weights; not calibrated cell counts")
    for level, values in (("celltype", z), ("tissueregion", spatial)):
        embedding = embeddings[level]
        canonical = output_dir / f"ad_{level}_embedding.h5ad"
        original = output_dir / f"ad_{level}_embedding_nodeconv.h5ad"
        if not original.exists():
            # Save the original as read, before changing X or adding Stage-B fields.
            embedding.write_h5ad(original)
        embedding.X = values.astype(np.float32)
        embedding.obsm["stageB_pi"] = pi
        embedding.obs["stageB_reconstruction_mse"] = fit["reconstruction_mse"]
        embedding.obs["stageB_mixture_entropy"] = -(pi * np.log(pi + 1e-8)).sum(1)
        embedding.uns["stageB"] = metadata
        stage = output_dir / f"ad_{level}_embedding_stageB.h5ad"
        embedding.write_h5ad(stage)
        # Keep the canonical file intact if writing the new version fails.
        import shutil
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".h5ad", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(stage, temporary)
            os.replace(temporary, canonical)
        finally:
            temporary.unlink(missing_ok=True)
    bead_name = str(query.obs["file_name"].iloc[0])
    save_npz(output_dir / "stageB_pi.npz", pi=pi, archetype_centers=centers,
             kept_archetypes=bundle["kept_archetypes"], bead_files=np.asarray([bead_name]),
             obs_names=np.asarray(query.obs_names, dtype=str), genes=np.asarray(query.var_names, dtype=str),
             gene_mask=covered, beta=fit["beta"], alpha=fit["alpha"],
             reconstruction_mse=fit["reconstruction_mse"],
             reference_files=bundle["reference_files"],
             reference_model_sha256=bundle["reference_model_sha256"],
             **{f"pi__{Path(bead_name).stem}": pi})
    logging.info("[Stage-B] wrote %s; %d beads, %d archetypes, reconstruction MSE %.6g",
                 output_dir, query.n_obs, pi.shape[1], float(fit["reconstruction_mse"].mean()))
    return fit
