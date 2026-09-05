"""
Part A: niche composition alignment.

Tissue representation = spatial-neighborhood average of cell-type COMPOSITION
(archetype mixture pi for Slide-seq beads, one-hot kmeans archetype label for
single cells) -- modality-invariant by construction.

Loads the Stage-B (empirical-signature) stageB_pi.npz, builds per-sample niche
vectors (2D kNN graph, k=15, cKDTree, self included, then L1 row norm), and
evaluates region separation / platform mixing in niche space against the SAME
metrics computed on the Stage-A tissue embedding
(ad_tissueregion_embedding.h5ad, X 64-dim).

Run:
  CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    /ewsc/yhe/miniconda3/envs/FuseMap_952261_env/bin/python niche_align.py
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import torch
from scipy.spatial import cKDTree

BASE = "/ewsc/yhe/FuseMap-revision3/finalize_FuseMap_0831"
DATA_DIR = f"{BASE}/data3/"
OUT_DIR = f"{BASE}/output_data3_pw/"
NPZ = OUT_DIR + "stageB_pi.npz"
TISSUE_H5AD = OUT_DIR + "ad_tissueregion_embedding.h5ad"

F13 = "13months-disease-replicate_1.h5ad"
FST = "stereoseq_mousebrain.h5ad"
FSL = "slideseq_Puck60.h5ad"

KNN_NICHE = 15
K_EVAL = 30
K_LISI = 50

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ helpers
def get_xy(ad, name):
    """main.py coordinate logic: obs['x'/'y'] -> obs['col'/'row'] -> obsm['spatial']."""
    if "x" in ad.obs.columns:
        x, y, src = ad.obs["x"].values, ad.obs["y"].values, "obs[x/y]"
    elif "col" in ad.obs.columns and "row" in ad.obs.columns:
        x, y, src = ad.obs["col"].values, ad.obs["row"].values, "obs[col/row]"
    elif "spatial" in ad.obsm:
        x, y, src = ad.obsm["spatial"][:, 0], ad.obsm["spatial"][:, 1], "obsm[spatial]"
    else:
        raise SystemExit(f"BLOCKER: no spatial coordinates for {name}")
    xy = np.stack([np.asarray(x, dtype=np.float64),
                   np.asarray(y, dtype=np.float64)], axis=1)
    log(f"[coords] {name}: {src}  n={xy.shape[0]}")
    return xy


def niche_vectors(comp, xy, k=KNN_NICHE):
    """Mean composition over the k spatial nearest neighbors (self included),
    then L1 row normalization."""
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k)          # self is first neighbor (d=0)
    niche = comp[idx].mean(axis=1)
    s = niche.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (niche / s).astype(np.float32)


def knn_idx(Q, R, k, same=False, chunk=1024):
    """GPU chunked cdist kNN indices of Q rows among R rows.
    same=True -> Q is R (exclude self)."""
    Qt = torch.from_numpy(np.ascontiguousarray(Q)).float().to(DEV)
    Rt = torch.from_numpy(np.ascontiguousarray(R)).float().to(DEV)
    out = []
    for st in range(0, Q.shape[0], chunk):
        en = min(st + chunk, Q.shape[0])
        d = torch.cdist(Qt[st:en], Rt)
        if same:
            d[torch.arange(en - st), torch.arange(st, en)] = float("inf")
        out.append(d.topk(k, dim=1, largest=False).indices.cpu().numpy())
        del d
    del Qt, Rt
    torch.cuda.empty_cache() if DEV.type == "cuda" else None
    return np.concatenate(out, axis=0)


def frac_target(Q, R, ref_labels, targets, k=K_EVAL):
    idx = knn_idx(Q, R, k)
    is_t = np.isin(ref_labels, list(targets)).astype(np.float32)
    return float(is_t[idx].mean())


def self_consistency(Z, labels, min_count=500, k=K_EVAL):
    uniq, cnt = np.unique(labels, return_counts=True)
    keep = set(uniq[cnt >= min_count])
    mask = np.isin(labels, list(keep))
    Zs, ls = Z[mask], labels[mask]
    idx = knn_idx(Zs, Zs, k, same=True)
    frac = float((ls[idx] == ls[:, None]).mean())
    return frac, sorted(keep), int(mask.sum())


def knn_transfer(Ztr, ytr, Zte, yte, k=K_EVAL):
    """Binary kNN classifier (majority vote). Returns per-class + overall acc."""
    idx = knn_idx(Zte, Ztr, k)
    votes = ytr[idx].astype(np.float32).mean(axis=1)
    pred = (votes > 0.5).astype(np.int64)
    accs = {}
    for c in (0, 1):
        m = yte == c
        accs[c] = float((pred[m] == c).mean())
    overall = float((pred == yte).mean())
    return accs[0], accs[1], overall


def ilisi_norm(Za, Zb, k=K_LISI):
    """iLISI between two batches, normalized to [0,1] as (LISI-1)/(nb-1)."""
    Z = np.concatenate([Za, Zb], axis=0)
    b = np.concatenate([np.zeros(len(Za)), np.ones(len(Zb))]).astype(np.float32)
    idx = knn_idx(Z, Z, k, same=True)
    p1 = b[idx].mean(axis=1)
    simpson = p1 ** 2 + (1.0 - p1) ** 2
    lisi = 1.0 / simpson
    return float((lisi.mean() - 1.0) / (2.0 - 1.0))


def main():
    log(f"[env] device={DEV}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # ---------------------------------------------- Stage-B pi + kmeans labels
    z = np.load(NPZ)
    pi = z["pi"].astype(np.float32)                       # [n_beads, K]
    labels_sc = z["kmeans_labels_singlecell"].astype(np.int64)  # [60512], kept space
    K = pi.shape[1]
    if "archetype_centers" in z:
        assert z["archetype_centers"].shape[0] == K, "K mismatch pi vs centers"
    if "kept_archetypes" in z:
        log(f"[npz] kept archetypes ({K}): {z['kept_archetypes'].tolist()}")
    n_beads = pi.shape[0]
    assert n_beads == 54787, "unexpected bead count"
    assert labels_sc.min() >= 0 and labels_sc.max() < K, "labels outside [0,K)"
    log(f"[npz] pi {pi.shape}, K={K}, single-cell labels {labels_sc.shape}")

    # ---------------------------------------------- raw data (coords + labels)
    ad13 = sc.read_h5ad(DATA_DIR + F13)
    adst = sc.read_h5ad(DATA_DIR + FST)
    adsl = sc.read_h5ad(DATA_DIR + FSL)
    n13, nst, nsl = ad13.n_obs, adst.n_obs, adsl.n_obs
    assert (n13, nst, nsl) == (10372, 50140, 54787), "raw n_obs mismatch"
    # label order in stage_b: concat of latents[0] (13months) then latents[1]
    # (stereo) -- verified below against the tissue-embedding obs order too.
    assert labels_sc.shape[0] == n13 + nst, "label vector length mismatch"

    xy13 = get_xy(ad13, F13)
    xyst = get_xy(adst, FST)
    xysl = get_xy(adsl, FSL)

    stereo_ct = adst.obs["gt_cell_type_main"].astype(str).values
    top_struct = adsl.obs["TopStruct"].astype(str).values
    region13 = ad13.obs["region"].astype(str).values

    # ---------------------------------------------- composition vectors
    def onehot(lab):
        oh = np.zeros((lab.shape[0], K), dtype=np.float32)
        oh[np.arange(lab.shape[0]), lab] = 1.0
        return oh

    comp13 = onehot(labels_sc[:n13])
    compst = onehot(labels_sc[n13:])
    compsl = pi

    # ---------------------------------------------- niche vectors (per sample)
    nich13 = niche_vectors(comp13, xy13)
    nichst = niche_vectors(compst, xyst)
    nichsl = niche_vectors(compsl, xysl)
    log(f"[niche] built niche vectors (kNN={KNN_NICHE}, self incl., L1-normed): "
        f"13months {nich13.shape}, stereo {nichst.shape}, slideseq {nichsl.shape}")

    # ---------------------------------------------- Stage-A tissue embedding
    adT = sc.read_h5ad(TISSUE_H5AD)
    XT = np.asarray(adT.X, dtype=np.float32)
    fn = adT.obs["file_name"].astype(str).values
    m13 = np.char.find(fn.astype(str), "13months") >= 0
    mst = np.char.find(fn.astype(str), "stereoseq") >= 0
    msl = np.char.find(fn.astype(str), "slideseq") >= 0
    assert m13.sum() == n13 and mst.sum() == nst and msl.sum() == nsl, \
        "tissue-embedding sample row counts mismatch"
    # verify atlas order 13months -> stereo -> slideseq (labels_sc slicing)
    assert np.where(m13)[0].max() < np.where(mst)[0].min() < np.where(msl)[0].min(), \
        "tissue-embedding row order != [13months, stereo, slideseq]"
    T13, Tst, Tsl = XT[m13], XT[mst], XT[msl]
    log(f"[stageA] tissue embedding {XT.shape}: 13months {T13.shape}, "
        f"stereo {Tst.shape}, slideseq {Tsl.shape}")

    results = {}

    # ------------------------------------------------------ (i) HPF -> stereo
    hpf = top_struct == "HPF"
    tgt_hpf = {"EX CA", "GN DG"}
    results["i_niche"] = frac_target(nichsl[hpf], nichst, stereo_ct, tgt_hpf)
    results["i_stageA"] = frac_target(Tsl[hpf], Tst, stereo_ct, tgt_hpf)
    log(f"[eval i]   HPF beads (n={int(hpf.sum())}) -> frac EX CA/GN DG in "
        f"{K_EVAL}-NN stereo: niche={results['i_niche']:.3f}  "
        f"stageA={results['i_stageA']:.3f}  (chance ~0.112)")

    # ------------------------------------------------ (ii) Isocortex -> stereo
    iso = top_struct == "Isocortex"
    tgt_iso = {"EX L2/3", "EX L4", "EX L5/6", "EX L6"}
    results["ii_niche"] = frac_target(nichsl[iso], nichst, stereo_ct, tgt_iso)
    results["ii_stageA"] = frac_target(Tsl[iso], Tst, stereo_ct, tgt_iso)
    log(f"[eval ii]  Isocortex beads (n={int(iso.sum())}) -> frac EX L2/3|L4|L5/6|L6 "
        f"in {K_EVAL}-NN stereo: niche={results['ii_niche']:.3f}  "
        f"stageA={results['ii_stageA']:.3f}  (chance ~0.086)")

    # ------------------------------- (iii) within-slideseq region separability
    results["iii_niche"], labs_n, n_used = self_consistency(nichsl, top_struct)
    results["iii_stageA"], labs_a, _ = self_consistency(Tsl, top_struct)
    assert labs_n == labs_a
    log(f"[eval iii] slideseq TopStruct {K_EVAL}-NN self-consistency "
        f"(labels>=500 beads: {labs_n}; n={n_used}): "
        f"niche={results['iii_niche']:.3f}  stageA={results['iii_stageA']:.3f}")

    # ------------------------------- (iv) cross-platform region transfer
    tr_mask = np.isin(top_struct, ["HPF", "Isocortex"])
    ytr = (top_struct[tr_mask] == "Isocortex").astype(np.int64)  # 0=HPF 1=Isocortex
    te_mask = np.isin(region13, ["Hippocampus", "Cortex"])
    yte = (region13[te_mask] == "Cortex").astype(np.int64)
    log(f"[eval iv]  transfer train slideseq n={int(tr_mask.sum())} "
        f"(HPF={int((ytr==0).sum())}, Iso={int((ytr==1).sum())}); "
        f"test 13months n={int(te_mask.sum())} "
        f"(Hippocampus={int((yte==0).sum())}, Cortex={int((yte==1).sum())})")
    a0n, a1n, aon = knn_transfer(nichsl[tr_mask], ytr, nich13[te_mask], yte)
    a0a, a1a, aoa = knn_transfer(Tsl[tr_mask], ytr, T13[te_mask], yte)
    results["iv_niche_hpf"], results["iv_niche_iso"], results["iv_niche_all"] = a0n, a1n, aon
    results["iv_stageA_hpf"], results["iv_stageA_iso"], results["iv_stageA_all"] = a0a, a1a, aoa
    log(f"[eval iv]  acc Hippocampus->HPF: niche={a0n:.3f}  stageA={a0a:.3f}")
    log(f"[eval iv]  acc Cortex->Isocortex: niche={a1n:.3f}  stageA={a1a:.3f}")
    log(f"[eval iv]  overall acc: niche={aon:.3f}  stageA={aoa:.3f}")

    # ------------------------------- (v) iLISI stereo <-> slideseq
    results["v_niche"] = ilisi_norm(nichst, nichsl)
    results["v_stageA"] = ilisi_norm(Tst, Tsl)
    log(f"[eval v]   iLISI stereo<->slideseq (k={K_LISI}, normalized [0,1]): "
        f"niche={results['v_niche']:.4f}  stageA={results['v_stageA']:.4f}")

    # ------------------------------------------------------- final table
    log("\n================== NICHE SPACE vs STAGE-A TISSUE EMBEDDING ==================")
    log(f"{'metric':<58s} {'niche':>8s} {'stageA':>8s}")
    rows = [
        ("(i)   HPF beads -> EX CA/GN DG frac (chance ~0.112)",
         results["i_niche"], results["i_stageA"]),
        ("(ii)  Isocortex beads -> EX L2/3|L4|L5/6|L6 frac (chance ~0.086)",
         results["ii_niche"], results["ii_stageA"]),
        ("(iii) slideseq TopStruct kNN self-consistency",
         results["iii_niche"], results["iii_stageA"]),
        ("(iv)  transfer acc Hippocampus->HPF",
         results["iv_niche_hpf"], results["iv_stageA_hpf"]),
        ("(iv)  transfer acc Cortex->Isocortex",
         results["iv_niche_iso"], results["iv_stageA_iso"]),
        ("(iv)  transfer acc overall",
         results["iv_niche_all"], results["iv_stageA_all"]),
        ("(v)   iLISI stereo<->slideseq (norm 0-1, higher=more mixed)",
         results["v_niche"], results["v_stageA"]),
    ]
    for name, vn, va in rows:
        log(f"{name:<58s} {vn:>8.3f} {va:>8.3f}")
    log("==============================================================================")
    log("\n[done] niche composition alignment evaluation complete.")


if __name__ == "__main__":
    main()
