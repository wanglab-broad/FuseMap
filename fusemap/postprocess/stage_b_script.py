"""
Stage-B in-model deconvolution for FuseMap (bead-resolution datasets, e.g.
Slide-seq pucks). Supports MULTIPLE bead datasets and MULTIPLE signature
references.

Re-express each bead as a convex mixture over FROZEN archetype
signatures derived from the already-trained Stage-A model, with explicit
per-gene platform terms (multiplicative beta, additive alpha), then rebuild
bead embeddings from the mixture weights. No input data is modified; the main
model is NOT retrained.

Standalone script. Run:
  CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  FUSEMAP_BEAD_FILES=<substr>[,<substr>...] FUSEMAP_SIG_REF=<substr>[,<substr>...] \
    /ewsc/yhe/miniconda3/envs/FuseMap_952261_env/bin/python stage_b_deconv.py

Env:
  FUSEMAP_BEAD_FILES  comma-separated filename substrings, each resolving to one
                      DISTINCT bead-resolution dataset to deconvolve
                      (FUSEMAP_BEAD_FILE is kept as a single-file alias).
  FUSEMAP_SIG_REF     comma-separated filename substrings, each resolving to one
                      NON-bead single-cell reference for empirical signatures
                      (at least one required in empirical mode).
  STAGEB_DATA_DIR     input h5ad folder (default: finalize_FuseMap_0831/data3/).
  STAGEB_OUT_DIR      Stage-A output folder to read latents/model from and write
                      Stage-B outputs into.
  STAGEB_SIG          'empirical' (default) or 'decode'.
  STAGEB_ENT_W        entropy regularization weight (default 0.0005).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import pickle
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
from sklearn.cluster import MiniBatchKMeans

# ------------------------------------------------------------------ paths
BASE = "/ewsc/yhe/FuseMap-revision3/finalize_FuseMap_0831"
import os as _os
DATA_DIR = _os.environ.get("STAGEB_DATA_DIR", f"{BASE}/data3/")
if not DATA_DIR.endswith("/"):
    DATA_DIR += "/"
OUT_DIR = _os.environ.get("STAGEB_OUT_DIR", f"{BASE}/output_data3_pw/")
if not OUT_DIR.endswith("/"):
    OUT_DIR += "/"
MODEL_PT = f"{OUT_DIR}/trained_model/FuseMap_final_model_final.pt"
LATENT_PKL = f"{OUT_DIR}/latent_embeddings_all_single_final.pkl"

# atlas order = os.listdir order of DATA_DIR (replicated from main.py lines 26-30)
FILE_NAMES = [
    f for f in os.listdir(DATA_DIR)
    if os.path.isfile(os.path.join(DATA_DIR, f))
]
def _resolve_token(env_key, token, purpose, candidates):
    hits = [f for f in candidates if token in f]
    if len(hits) != 1:
        raise SystemExit(
            f"BLOCKER: {env_key} token {token!r} matches {hits or 'nothing'} among "
            f"{candidates}; each token must match exactly one file "
            f"(identifying the {purpose}).")
    return hits[0]


def _resolve_file_list(env_key, purpose, candidates, alias_key=None):
    """Resolve a comma-separated env var of filename substrings; every token must
    match exactly one candidate file and all resolved files must be distinct."""
    raw = os.environ.get(env_key, "").strip()
    if not raw and alias_key:
        raw = os.environ.get(alias_key, "").strip()
    if not raw:
        alias_msg = f" (or {alias_key} for a single file)" if alias_key else ""
        raise SystemExit(
            f"BLOCKER: {env_key} is required{alias_msg} — set it to comma-separated "
            f"filename substrings identifying the {purpose}. "
            f"Candidates: {candidates}")
    files = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        hit = _resolve_token(env_key, token, purpose, candidates)
        if hit in files:
            raise SystemExit(
                f"BLOCKER: {env_key} tokens resolve to duplicate file {hit!r}; "
                f"each token must identify a distinct file.")
        files.append(hit)
    if not files:
        raise SystemExit(f"BLOCKER: {env_key}={raw!r} contains no usable tokens.")
    return files


BEAD_FILES = _resolve_file_list(
    "FUSEMAP_BEAD_FILES", "bead-resolution dataset(s) to deconvolve",
    FILE_NAMES, alias_key="FUSEMAP_BEAD_FILE")

SIG_MODE = os.environ.get("STAGEB_SIG", "empirical")
NON_BEAD_FILES = [f for f in FILE_NAMES if f not in BEAD_FILES]
if not NON_BEAD_FILES:
    raise SystemExit(
        "BLOCKER: every atlas file is declared a bead dataset; at least one "
        "non-bead single-cell atlas is required for archetypes/signatures.")
SIG_REF_FILES = (
    _resolve_file_list(
        "FUSEMAP_SIG_REF",
        "NON-bead single-cell reference(s) for empirical signatures "
        "(bead files are excluded from candidates; pick atlases with wide "
        "gene coverage)",
        NON_BEAD_FILES)
    if SIG_MODE == "empirical" else [])

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ helpers
def log(msg):
    print(msg, flush=True)


def var_names_after_preproc(ad):
    """Replicate fusemap.preprocess.preprocess_adata gene filtering to recover
    the exact per-atlas var_name (gene order) used by the trained decoder.

    Steps (from preprocess.py):
      1. uppercase var index
      2. keep unique genes via np.unique (reorders columns to sorted order)
      3. if the matrix is all integers (raw counts): filter genes by
         col-sum > 1 then col-max > 1 (cell filter does not change var_name)
    """
    vidx = [g.upper() for g in ad.var.index]
    ad2 = ad.copy()
    ad2.var.index = vidx
    _, indices = np.unique(ad2.var.index, return_index=True)
    ad2 = ad2[:, indices]
    X = ad2.X
    dense = X.toarray() if sp.issparse(X) else np.asarray(X)
    is_int = np.all(dense % 1 == 0)
    if is_int:
        m1 = np.asarray(np.sum(dense, axis=0)).ravel() > 1
        ad2 = ad2[:, m1]
        dense = dense[:, m1]
        m2 = np.asarray(np.max(dense, axis=0)).ravel() > 1
        ad2 = ad2[:, m2]
    return [g.upper() for g in ad2.var.index]


def main():
    log(f"[env] device={DEV}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log(f"[env] atlas order (os.listdir): {FILE_NAMES}")
    log(f"[env] DATA_DIR={DATA_DIR}  OUT_DIR={OUT_DIR}")
    log(f"[env] bead datasets: {BEAD_FILES}")
    bead_idx = [FILE_NAMES.index(f) for f in BEAD_FILES]
    bead_idx_set = set(bead_idx)

    # -------------------------------------------------- load Stage-A latents
    with open(LATENT_PKL, "rb") as f:
        latents = pickle.load(f)
    log(f"[latents] shapes: {[x.shape for x in latents]}")
    sc_atlas_idx = [i for i in range(len(latents)) if i not in bead_idx_set]
    for bf, bi in zip(BEAD_FILES, bead_idx):
        log(f"[latents] bead atlas = {bf} idx={bi} (n_obs={latents[bi].shape[0]})")
    log(f"[latents] single-cell atlas indices for clustering = {sc_atlas_idx}")

    # -------------------------------------------------- reconstruct var_index
    var_names = []
    atlas_obs_cols = []   # per-atlas obs columns (for eval reference selection)
    for f in FILE_NAMES:
        ad = sc.read_h5ad(DATA_DIR + f)
        vn = var_names_after_preproc(ad)
        var_names.append(vn)
        atlas_obs_cols.append(set(ad.obs.columns))
        log(f"[genes] {f}: n_var_after_preproc={len(vn)}")
        del ad
    all_unique_genes = sorted(set().union(*var_names))
    log(f"[genes] all_unique_genes = {len(all_unique_genes)}")
    var_index_bead = {}
    for bf, bi in zip(BEAD_FILES, bead_idx):
        vi = [all_unique_genes.index(g) for g in var_names[bi]]
        var_index_bead[bf] = vi
        log(f"[genes] var_index {bf}: len={len(vi)} "
            f"(min={min(vi)}, max={max(vi)})")

    # -------------------------------------------------- gene_embedding width check
    sd = torch.load(MODEL_PT, map_location="cpu")
    gene_embedding = sd["gene_embedding"]  # [64, n_all_genes]
    if gene_embedding.shape[1] != len(all_unique_genes):
        log(f"BLOCKER: gene_embedding width {gene_embedding.shape[1]} != reconstructed "
            f"vocabulary {len(all_unique_genes)}; var_index mismatch. STOP.")
        raise SystemExit(1)
    log(f"[model] gene_embedding shape = {tuple(gene_embedding.shape)}")
    assert gene_embedding.shape[0] == 64

    # -------------------------------------------------- fusemap preprocess module
    # (used to rebuild the training normalization for any raw-count input)
    import importlib.util as _ilu
    _pp_dir = os.path.dirname(os.path.abspath(__file__))
    _pp_path = os.path.join(_pp_dir, "fusemap", "preprocess.py")
    if not os.path.exists(_pp_path):
        _pp_path = os.path.join(_pp_dir, "preprocess.py")
    _spec = _ilu.spec_from_file_location(
        "fusemap_preprocess_standalone", _pp_path)
    _ppmod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_ppmod)

    # -------------------------------------------------- Step 1: archetypes
    C_input = np.concatenate([latents[i] for i in sc_atlas_idx], axis=0).astype(np.float32)
    log(f"[kmeans] clustering single-cell latents jointly: {C_input.shape}")
    km = MiniBatchKMeans(n_clusters=40, random_state=0, batch_size=4096)
    km_labels = km.fit_predict(C_input)
    C = km.cluster_centers_.astype(np.float32)  # [40, 64]
    sizes = np.bincount(km_labels, minlength=40)
    log(f"[kmeans] 40 archetype cluster sizes: {sizes.tolist()}")
    log(f"[kmeans] min/median/max cluster size: "
        f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")

    # -------------------------------------------------- Step 2: signature inputs
    # STAGEB_SIG=empirical (default): per-archetype MEAN of normalized reference
    #   expression of the kmeans member cells, POOLED over all FUSEMAP_SIG_REF
    #   references (in-distribution signatures).
    # STAGEB_SIG=decode (legacy): S = relu(C @ G_bead) via decoder weights
    #   (measured to be out-of-distribution: decoder saw graph-aggregated inputs).
    log(f"[sig] signature mode STAGEB_SIG = {SIG_MODE}")
    kept = np.arange(40, dtype=np.int64)               # kept archetypes (orig ids)
    km_labels_kept = km_labels.astype(np.int64).copy() # labels in kept-archetype space
    ref_data = []  # per reference: vocab, member counts [40], member means [40, n_genes_r]
    if SIG_MODE == "decode":
        K = 40
        C_used = C
    elif SIG_MODE == "empirical":
        log(f"[sig] empirical signature references: {SIG_REF_FILES}")
        for RF in SIG_REF_FILES:
            ref_atlas_idx = FILE_NAMES.index(RF)
            # km_labels order = concat of latents[sc_atlas_idx]; slice the ref block
            pos = sc_atlas_idx.index(ref_atlas_idx)
            offset = sum(latents[i].shape[0] for i in sc_atlas_idx[:pos])
            ref_labels = km_labels[offset:offset + latents[ref_atlas_idx].shape[0]]
            assert ref_labels.shape[0] == latents[ref_atlas_idx].shape[0], (
                f"{RF}: kmeans label slice mismatch")

            # ---- reference normalized expression exactly as training preprocessing.
            # float non-integer X was left untouched by training preprocessing
            # (X == spatial_input); integer counts were normalized; rebuild with
            # the SAME preprocess_adata function.
            ad_rf_raw = sc.read_h5ad(DATA_DIR + RF)
            _Xr = ad_rf_raw.X
            raw_all_int = (np.all(_Xr.data % 1 == 0) if sp.issparse(_Xr)
                           else np.all(np.asarray(_Xr) % 1 == 0))
            log(f"[sig] {RF}: raw X all-integer counts: {raw_all_int} -> "
                + ("rebuilding normalization via fusemap.preprocess.preprocess_adata"
                   if raw_all_int else
                   "X already normalized; preprocess only dedups genes"))
            ad_rf_pp = _ppmod.preprocess_adata([ad_rf_raw], 1)[0]
            vn_rf = [g.upper() for g in ad_rf_pp.var.index]
            assert vn_rf == var_names[ref_atlas_idx], (
                f"{RF}: preprocessed vocab mismatch vs training reconstruction")
            assert ad_rf_pp.n_obs == ref_labels.shape[0], (
                f"{RF}: preprocessed cell count {ad_rf_pp.n_obs} != "
                f"{ref_labels.shape[0]} (latent rows)")
            Xrf = ad_rf_pp.X
            Xrf = (Xrf.tocsr() if sp.issparse(Xrf) else sp.csr_matrix(Xrf)).astype(np.float32)
            assert float(Xrf.max()) <= 10.0 + 1e-4, f"{RF}: normalized X max > 10 (scale)"
            assert not np.all(Xrf.data % 1 == 0), f"{RF}: X still integer after preprocess"
            log(f"[sig] {RF}: normalized expr {Xrf.shape}  max={Xrf.max():.4f} "
                f"(verified vocab == training var_names, n_obs == latent rows)")

            # ---- per-archetype member counts and mean expression in this ref
            rf_counts = np.bincount(ref_labels, minlength=40).astype(np.int64)
            rf_means = np.zeros((40, Xrf.shape[1]), dtype=np.float32)
            for k in range(40):
                idx = np.where(ref_labels == k)[0]
                if len(idx):
                    rf_means[k] = np.asarray(Xrf[idx].mean(axis=0)).ravel().astype(np.float32)
            log(f"[sig] {RF}: member counts per archetype: {rf_counts.tolist()}")
            ref_data.append({"file": RF, "vn": vn_rf,
                             "counts": rf_counts, "means": rf_means})
            del ad_rf_raw, ad_rf_pp, Xrf, rf_means

        # ---- archetype membership pooled across refs; drop tiny archetypes
        total_counts = np.sum([r["counts"] for r in ref_data], axis=0)
        kept = np.where(total_counts >= 20)[0].astype(np.int64)
        dropped = np.where(total_counts < 20)[0]
        K = len(kept)
        log(f"[sig] pooled reference member counts per archetype: {total_counts.tolist()}")
        log(f"[sig] DROPPED {len(dropped)}/40 archetypes with <20 pooled reference "
            f"members: {dropped.tolist()} (pooled counts {total_counts[dropped].tolist()}); K={K}")
        C_used = C[kept]

        # remap kmeans labels to kept-archetype space; cells from dropped
        # archetypes are reassigned to the nearest kept centroid
        remap = -np.ones(40, dtype=np.int64)
        remap[kept] = np.arange(K)
        km_labels_kept = remap[km_labels]
        _bad = np.where(km_labels_kept < 0)[0]
        if len(_bad):
            _d = ((C_input[_bad][:, None, :] - C_used[None, :, :]) ** 2).sum(-1)
            km_labels_kept[_bad] = np.argmin(_d, axis=1)
        log(f"[sig] reassigned {len(_bad)} single cells from dropped archetypes "
            f"to nearest kept centroid")
    else:
        raise SystemExit(f"BLOCKER: unknown STAGEB_SIG={SIG_MODE!r} "
                         "(expected 'empirical' or 'decode')")

    ENT_W = float(os.environ.get("STAGEB_ENT_W", "0.0005"))
    eps = 1e-8
    CHUNK = 4096
    N_STEPS = 800
    results = {}  # per bead file: pi, Z, Z_spatial, gene_mask, var_index, ...

    # ================================================== per-bead-dataset loop
    for bf, bi in zip(BEAD_FILES, bead_idx):
        log(f"\n================ bead dataset {bf} (atlas idx {bi}) ================")
        var_index_b = var_index_bead[bf]
        G_bead = gene_embedding[:, var_index_b].to(DEV).float()  # [64, n_gene_b]
        log(f"[model] {bf}: G_bead shape = {tuple(G_bead.shape)}")

        # ---- bead normalized expression, verified like training preprocessing:
        # float non-integer X == spatial_input (used as-is); integer counts are
        # rebuilt with the SAME preprocess_adata function.
        ad_bead_raw = sc.read_h5ad(DATA_DIR + bf)
        _Xb = ad_bead_raw.X
        bead_all_int = (np.all(_Xb.data % 1 == 0) if sp.issparse(_Xb)
                        else np.all(np.asarray(_Xb) % 1 == 0))
        if bead_all_int:
            log(f"[data] {bf}: X is integer counts -> rebuilding normalization via "
                f"fusemap.preprocess.preprocess_adata")
            ad_bead = _ppmod.preprocess_adata([ad_bead_raw], 1)[0]
        else:
            log(f"[data] {bf}: X is float non-integer -> already normalized "
                f"(file X == spatial_input); used as-is")
            ad_bead = ad_bead_raw
        assert [g.upper() for g in ad_bead.var.index] == var_names[bi], (
            f"{bf}: file gene order != training vocab reconstruction")
        Xbead = ad_bead.X
        if not sp.issparse(Xbead):
            Xbead = sp.csr_matrix(Xbead)
        Xbead = Xbead.tocsr().astype(np.float32)
        n_bead, n_gene = Xbead.shape
        assert n_gene == len(var_index_b), f"{bf}: gene dim mismatch"
        assert n_bead == latents[bi].shape[0], (
            f"{bf}: n_obs {n_bead} != latent rows {latents[bi].shape[0]}")
        log(f"[data] {bf}: normalized expr X: {Xbead.shape}")

        # ---- per-bead signatures S over THIS bead's vocabulary + gene mask
        gene_mask_np = np.ones(n_gene, dtype=np.float32)   # 1 = gene used in recon loss
        if SIG_MODE == "decode":
            Ct = torch.from_numpy(C).to(DEV)  # [40,64]
            with torch.no_grad():
                S = torch.relu(Ct @ G_bead)  # [40, n_gene]
            log(f"[sig] {bf}: archetype signatures S = relu(C @ G_bead): {tuple(S.shape)}, "
                f"mean={S.mean().item():.4f}, nnz_frac={(S>0).float().mean().item():.3f}")
        else:
            # empirical, pooled across references:
            #   signature_kg = (sum_r member-expression sums) / (sum_r member counts)
            # using ONLY references whose vocab contains gene g; genes covered by
            # NO reference are masked from the reconstruction loss.
            vn_b = var_names[bi]
            num = np.zeros((K, n_gene), dtype=np.float64)
            den = np.zeros((K, n_gene), dtype=np.float64)
            covered = np.zeros(n_gene, dtype=bool)
            for r in ref_data:
                pos_r = {g: j for j, g in enumerate(r["vn"])}
                col_map = np.array([pos_r.get(g, -1) for g in vn_b], dtype=np.int64)
                present = col_map >= 0
                covered |= present
                cnts = r["counts"][kept].astype(np.float64)          # [K]
                num[:, present] += (r["means"][kept][:, col_map[present]].astype(np.float64)
                                    * cnts[:, None])
                den[:, present] += cnts[:, None]
                log(f"[sig] {bf}: reference {r['file']} covers "
                    f"{int(present.sum())}/{n_gene} bead genes")
            gene_mask_np = covered.astype(np.float32)
            S_np = np.zeros((K, n_gene), dtype=np.float32)
            nz = den > 0
            S_np[nz] = (num[nz] / den[nz]).astype(np.float32)
            S = torch.from_numpy(S_np).to(DEV)
            n_masked = int((~covered).sum())
            log(f"[sig] {bf}: empirical signatures S (pooled mean ref expr per archetype): "
                f"{tuple(S.shape)}, mean={S_np.mean():.4f}, nnz_frac={(S_np > 0).mean():.3f}")
            log(f"[sig] {bf}: MASKED GENES: {n_masked}/{n_gene} bead genes absent from "
                f"ALL reference vocabs (signature 0, excluded from reconstruction loss)")
            del num, den, S_np

        # ---------------------------------------------- Step 3: bead deconvolution
        # x_b ~= beta (*) (pi_b @ S) + alpha ; pi_b = softmax(logits_b)
        # loss = MSE(x, pred) + ENT_W * mean_entropy(pi)
        # full-batch Adam(lr=0.05); gradient accumulated over bead chunks (exact
        # full-batch gradient, memory-bounded). logits/beta/alpha are fit PER
        # bead dataset (platform terms are per-dataset).
        logits = torch.zeros(n_bead, K, device=DEV, requires_grad=True)
        beta_init = float(torch.log(torch.expm1(torch.tensor(1.0))))     # softplus->1.0
        alpha_init = float(torch.log(torch.expm1(torch.tensor(0.01))))   # softplus->0.01
        beta_raw = torch.full((n_gene,), beta_init, device=DEV, requires_grad=True)
        alpha_raw = torch.full((n_gene,), alpha_init, device=DEV, requires_grad=True)

        opt = torch.optim.Adam([logits, beta_raw, alpha_raw], lr=0.05)

        gene_mask = torch.from_numpy(gene_mask_np).to(DEV)  # [n_gene]
        n_gene_used = int(gene_mask_np.sum())
        log(f"[opt] {bf}: reconstruction loss over {n_gene_used}/{n_gene} genes "
            f"({n_gene - n_gene_used} masked)")
        N_ELEM = float(n_bead * n_gene_used)

        prev_loss = None
        mse = float("nan")
        # pre-convert sparse chunks to dense GPU tensors ONCE (the per-step
        # .toarray() conversion dominated runtime: ~40x slowdown)
        dense_chunks = []
        for st in range(0, n_bead, CHUNK):
            en = min(st + CHUNK, n_bead)
            dense_chunks.append(
                torch.from_numpy(Xbead[st:en].toarray()).to(DEV)
            )
        for step in range(N_STEPS):
            opt.zero_grad(set_to_none=True)
            total_se = 0.0
            total_ent = 0.0
            for ci, st in enumerate(range(0, n_bead, CHUNK)):
                en = min(st + CHUNK, n_bead)
                # recompute per chunk so each chunk builds its own autograd graph;
                # gradients into beta_raw/alpha_raw accumulate across chunks -> exact
                # full-batch gradient
                beta = torch.nn.functional.softplus(beta_raw)   # [n_gene]
                alpha = torch.nn.functional.softplus(alpha_raw)  # [n_gene]
                xb = dense_chunks[ci]  # [c, n_gene], preloaded on device
                pi = torch.softmax(logits[st:en], dim=1)  # [c,K]
                pred = beta * (pi @ S) + alpha            # [c,n_gene]
                se = (((xb - pred) ** 2) * gene_mask).sum()
                ent = (-(pi * torch.log(pi + eps)).sum(dim=1)).sum()
                loss_chunk = se / N_ELEM + ENT_W * ent / n_bead
                loss_chunk.backward()
                total_se += se.item()
                total_ent += ent.item()
                del pred, pi
            opt.step()
            mse = total_se / N_ELEM
            mean_ent = total_ent / n_bead
            total_loss = mse + ENT_W * mean_ent
            if step % 100 == 0 or step == N_STEPS - 1:
                log(f"[opt] {bf}: step {step:4d}  loss={total_loss:.6f}  "
                    f"mse={mse:.6f}  mean_entropy={mean_ent:.4f}")
            if prev_loss is not None and abs(prev_loss - total_loss) < 1e-7 and step > 200:
                log(f"[opt] {bf}: plateaued at step {step}")
                break
            prev_loss = total_loss

        with torch.no_grad():
            beta = torch.nn.functional.softplus(beta_raw)
            alpha = torch.nn.functional.softplus(alpha_raw)
            pi = torch.softmax(logits, dim=1)  # [n_bead,K]
            pi_entropy = (-(pi * torch.log(pi + eps)).sum(dim=1)).mean().item()
        log(f"[opt] {bf}: FINAL recon MSE={mse:.6f}  pi_mean_entropy={pi_entropy:.4f}")
        log(f"[opt] {bf}: beta (per-gene mult): mean={beta.mean().item():.4f} "
            f"min={beta.min().item():.4f} max={beta.max().item():.4f}")
        log(f"[opt] {bf}: alpha (per-gene add):  mean={alpha.mean().item():.4f} "
            f"min={alpha.min().item():.4f} max={alpha.max().item():.4f}")

        pi_np = pi.detach().cpu().numpy().astype(np.float32)

        # ---------------------------------------------- Step 4: rebuild embeddings
        Z_bead = pi_np @ C_used  # [n_bead, 64]
        log(f"[rebuild] {bf}: Z_bead (new cell latent) = pi @ C: {Z_bead.shape}")
        if "adj_normalized" not in ad_bead.obsm:
            # rebuild the spatial graph exactly as training does (main.py x/y
            # fallback + delaunay construct_graph + preprocess_adj_sparse)
            from fusemap.data.graph import construct_graph, preprocess_adj_sparse
            if "x" not in ad_bead.obs.columns:
                if "col" in ad_bead.obs.columns and "row" in ad_bead.obs.columns:
                    ad_bead.obs["x"], ad_bead.obs["y"] = ad_bead.obs["col"], ad_bead.obs["row"]
                else:
                    ad_bead.obs["x"] = ad_bead.obsm["spatial"][:, 0]
                    ad_bead.obs["y"] = ad_bead.obsm["spatial"][:, 1]
            construct_graph([ad_bead], 1, ["delaunay"], ["ST"])
            preprocess_adj_sparse([ad_bead], 1, ["ST"])
            log(f"[rebuild] {bf}: adj_normalized rebuilt via training preprocessing (delaunay)")
        adjn = ad_bead.obsm["adj_normalized"]
        assert adjn.shape == (n_bead, n_bead), f"{bf}: adj_normalized shape {adjn.shape}"
        adjn = sp.csr_matrix(adjn)
        Z_bead_spatial = np.asarray(adjn.T.dot(Z_bead)).astype(np.float32)
        log(f"[rebuild] {bf}: Z_bead_spatial = adj_normalized.T @ Z_bead: "
            f"{Z_bead_spatial.shape} (used file obsm['adj_normalized'])")

        top_struct = (ad_bead.obs["TopStruct"].astype(str).values
                      if "TopStruct" in ad_bead.obs.columns else None)
        results[bf] = dict(
            idx=bi, pi=pi_np, Z=Z_bead, Z_spatial=Z_bead_spatial,
            gene_mask=gene_mask_np, var_index=np.asarray(var_index_b, dtype=np.int64),
            n_bead=n_bead, top_struct=top_struct)

        del dense_chunks, logits, beta_raw, alpha_raw, beta, alpha, opt, S, gene_mask
        del Xbead, ad_bead, ad_bead_raw, pi, G_bead, adjn
        torch.cuda.empty_cache()

    # -------------------------------------------------- Step 5: write outputs
    # single write per output file, covering ALL bead datasets
    def write_stageB(src_name, dst_name, field):
        ad = sc.read_h5ad(OUT_DIR + src_name)
        fn = ad.obs["file_name"].values
        Xnew = np.asarray(ad.X).astype(np.float32).copy()
        for bf in BEAD_FILES:
            res = results[bf]
            mask = (fn == bf)
            assert mask.sum() == res["n_bead"], (
                f"{src_name}: {bf} rows {mask.sum()} != {res['n_bead']}")
            rows = np.where(mask)[0]
            assert (rows.max() - rows.min() + 1) == len(rows), f"{bf}: rows not contiguous"
            Xnew[rows] = res[field]
            log(f"[write] {dst_name}: replaced {len(rows)} {bf} rows "
                f"[{rows.min()}:{rows.max()+1}]")
        ad.X = Xnew
        ad.write_h5ad(OUT_DIR + dst_name)
        log(f"[write] {OUT_DIR}{dst_name}")

    write_stageB("ad_celltype_embedding.h5ad",
                 "ad_celltype_embedding_stageB.h5ad", "Z")
    write_stageB("ad_tissueregion_embedding.h5ad",
                 "ad_tissueregion_embedding_stageB.h5ad", "Z_spatial")

    npz_payload = dict(
        archetype_centers=C_used,
        kept_archetypes=kept.astype(np.int32),
        kmeans_labels_singlecell=km_labels_kept.astype(np.int32),
        kmeans_labels_original=km_labels.astype(np.int32),
        bead_files=np.array(BEAD_FILES),
    )
    for bf in BEAD_FILES:
        stem = os.path.splitext(bf)[0]
        npz_payload[f"pi__{stem}"] = results[bf]["pi"]
        npz_payload[f"gene_mask__{stem}"] = results[bf]["gene_mask"].astype(np.float32)
        npz_payload[f"var_index__{stem}"] = results[bf]["var_index"]
    if len(BEAD_FILES) == 1:
        _only = results[BEAD_FILES[0]]
        npz_payload["pi"] = _only["pi"]                       # legacy single-bead alias
        npz_payload["gene_mask"] = _only["gene_mask"].astype(np.float32)
        npz_payload["var_index_slideseq"] = _only["var_index"]
    np.savez_compressed(OUT_DIR + "stageB_pi.npz", **npz_payload)
    log(f"[write] {OUT_DIR}stageB_pi.npz (per-bead keys "
        f"{['pi__' + os.path.splitext(bf)[0] for bf in BEAD_FILES]}, K={K}, "
        f"archetype_centers[{C_used.shape}], kmeans_labels_singlecell in kept-archetype space)")

    # -------------------------------------------------- Step 6: quick eval
    log("\n========================= QUICK EVAL =========================")
    # NN reference = LARGEST non-bead single-cell atlas (by n_obs) that has an
    # 'annotation' obs column (cell-type labels).
    ann_candidates = [i for i in sc_atlas_idx if "annotation" in atlas_obs_cols[i]]
    ref_eval_idx = None
    if not ann_candidates:
        log("[eval] SKIP quick eval: no non-bead single-cell atlas has an "
            "'annotation' obs column to serve as the NN reference.")
    else:
        ref_eval_idx = max(ann_candidates, key=lambda i: latents[i].shape[0])
        REF_EVAL_FILE = FILE_NAMES[ref_eval_idx]
        Z_ref = latents[ref_eval_idx].astype(np.float32)
        ad_ref = sc.read_h5ad(DATA_DIR + REF_EVAL_FILE)
        ref_ct = ad_ref.obs["annotation"].astype(str).values
        if len(ref_ct) != Z_ref.shape[0]:
            log(f"[eval] SKIP quick eval: {REF_EVAL_FILE} n_obs {len(ref_ct)} != "
                f"latent rows {Z_ref.shape[0]}; labels cannot be aligned.")
            ref_eval_idx = None
        else:
            log(f"[eval] NN reference atlas = {REF_EVAL_FILE} (n_obs={Z_ref.shape[0]}, "
                f"largest single-cell atlas with obs['annotation'])")

    if ref_eval_idx is not None:
        def knn_target_fraction(Zq, Zref, ref_labels, target_set, k=30, chunk=512):
            Zr = torch.from_numpy(Zref).to(DEV)
            ref_is_target = torch.from_numpy(
                np.isin(ref_labels, list(target_set)).astype(np.float32)
            ).to(DEV)
            fracs = []
            for st in range(0, Zq.shape[0], chunk):
                en = min(st + chunk, Zq.shape[0])
                q = torch.from_numpy(Zq[st:en]).to(DEV)
                d = torch.cdist(q, Zr)                        # [c, n_ref]
                nn_idx = d.topk(k, dim=1, largest=False).indices  # [c,k]
                frac = ref_is_target[nn_idx].mean(dim=1)      # [c]
                fracs.append(frac.cpu().numpy())
                del q, d, nn_idx
            return float(np.concatenate(fracs).mean())

        def ilisi(Z_a, Z_b, k=50, chunk=512):
            Z = np.concatenate([Z_a, Z_b], axis=0).astype(np.float32)
            batch = np.concatenate([np.zeros(len(Z_a)), np.ones(len(Z_b))]).astype(np.float32)
            Zt = torch.from_numpy(Z).to(DEV)
            bt = torch.from_numpy(batch).to(DEV)
            vals = []
            for st in range(0, Z.shape[0], chunk):
                en = min(st + chunk, Z.shape[0])
                d = torch.cdist(Zt[st:en], Zt)
                d[torch.arange(en - st), torch.arange(st, en)] = float("inf")
                nn_idx = d.topk(k, dim=1, largest=False).indices  # [c,k]
                nb = bt[nn_idx]                                    # [c,k]
                p1 = nb.mean(dim=1)
                p0 = 1.0 - p1
                simpson = p0 ** 2 + p1 ** 2
                lisi = 1.0 / simpson
                vals.append(lisi.cpu().numpy())
                del d, nn_idx, nb
            return float(np.concatenate(vals).mean())

        for bf in BEAD_FILES:
            res = results[bf]
            bi = res["idx"]
            Z_bead = res["Z"]
            log(f"\n[eval] ---- bead dataset {bf} ----")
            top_struct = res["top_struct"]
            if top_struct is None:
                log(f"[eval] SKIP HPF/Isocortex checks for {bf}: file has no "
                    f"obs['TopStruct'] column (needed to select HPF/Isocortex beads).")
            else:
                hpf_mask = (top_struct == "HPF")
                tgt_hpf = {"EX CA", "GN DG"}
                frac_hpf_B = knn_target_fraction(Z_bead[hpf_mask], Z_ref, ref_ct, tgt_hpf) \
                    if hpf_mask.sum() else float("nan")
                frac_hpf_A = knn_target_fraction(
                    latents[bi][hpf_mask].astype(np.float32), Z_ref, ref_ct, tgt_hpf)
                log(f"[eval i]  HPF beads (n={int(hpf_mask.sum())}) -> EX CA/GN DG frac in 30 NN reference:")
                log(f"          Stage-B (new) = {frac_hpf_B:.3f}   "
                    f"[recomputed Stage-A orig = {frac_hpf_A:.3f}; "
                    f"reported Stage-A = 0.224; chance ~0.112]")

                iso_mask = (top_struct == "Isocortex")
                tgt_iso = {"EX L2/3", "EX L4", "EX L5/6", "EX L6"}
                frac_iso_B = knn_target_fraction(Z_bead[iso_mask], Z_ref, ref_ct, tgt_iso) \
                    if iso_mask.sum() else float("nan")
                frac_iso_A = knn_target_fraction(
                    latents[bi][iso_mask].astype(np.float32), Z_ref, ref_ct, tgt_iso)
                log(f"[eval ii] Isocortex beads (n={int(iso_mask.sum())}) -> EX L2/3|L4|L5/6|L6 frac in 30 NN reference:")
                log(f"          Stage-B (new) = {frac_iso_B:.3f}   "
                    f"[recomputed Stage-A orig = {frac_iso_A:.3f}; "
                    f"reported Stage-A = 0.167; chance ~0.086]")

            ilisi_B = ilisi(Z_ref, Z_bead, k=50)
            ilisi_A = ilisi(Z_ref, latents[bi].astype(np.float32), k=50)
            log(f"[eval iii] iLISI {os.path.splitext(REF_EVAL_FILE)[0]}<->{os.path.splitext(bf)[0]} "
                f"(k=50, range [1,2], higher=more mixed):")
            log(f"          Stage-B (new) = {ilisi_B:.4f}   [recomputed Stage-A orig = {ilisi_A:.4f}]")

    log("\n[done] Stage-B deconvolution complete.")


if __name__ == "__main__":
    main()
