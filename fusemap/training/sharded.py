"""Atlas-sharded multi-GPU integration. EXPERIMENTAL - DO NOT USE FOR RESULTS.

Status (2026-09-05): trains and produces outputs, and the single-GPU default
path is bit-identical to the original code (verified 6x by md5), but the
adversarial game does not yet reproduce single-GPU cross-atlas mixing
(data2 iLISI 0.03-0.09 vs the single-GPU band 0.25-0.33 across v1-v7
iterations; see project notes). Kept for further R&D. For foundation-scale
training use the validated single-GPU path.


Splits the atlases (input sections) across GPUs: each rank trains only its
shard's per-atlas encoders/decoders while the shared components (universal
gene embedding + the two discriminators) are gradient-synchronized every
step, so all ranks apply identical updates to the shared parameters.

Semantics vs single-GPU training:
- per-atlas AE terms: exact (sum over atlases is preserved by the SUM
  all-reduce of shared gradients; atlas-local gradients never leave their
  owner rank)
- adversarial terms (sample-mean scaled by lambda_disc): corrected by
  dividing each rank's lambda_disc by the world size, so the sum of
  per-rank means approximates the global mean for (near-)equal shard sizes
- discriminator classes use GLOBAL atlas indices on every rank
  (train_model.ATLAS_LABEL_MAP)

v0 restrictions: integrate mode only, FUSEMAP_ANCHOR_LAMBDA must be 0,
use_llm_gene_embedding='false'.
"""

import copy
import logging
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

__all__ = ["spatial_integrate_sharded"]


def _shared_params(model):
    """Parameters shared across atlases: gene embedding + discriminators."""
    seen, out = set(), []
    mods = [model.discriminator_single, model.discriminator_spatial]
    for m in mods:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); out.append(p)
    for name in ["gene_embedding", "gene_embedding_pretrained", "gene_embedding_new"]:
        p = getattr(model, name, None)
        if isinstance(p, torch.nn.Parameter) and id(p) not in seen:
            seen.add(id(p)); out.append(p)
    return out


class _DistHooks:
    def __init__(self, shared, device):
        self.shared = shared
        self.device = device
        self.flagconfig = None

    def sync_grads(self, _model=None):
        for p in self.shared:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if self.flagconfig is not None:
            t = torch.tensor([float(self.flagconfig.lambda_disc_single)], device=self.device)
            dist.broadcast(t, src=0)
            self.flagconfig.lambda_disc_single = t.item()

    def sync_value(self, x):
        t = torch.tensor([float(x)], device=self.device)
        dist.broadcast(t, src=0)
        return t.item()


def _make_dist_gather(device):
    """Cross-rank gather for discriminator inputs: local slice keeps grads,
    remote slices are detached. Concatenation order = rank order, so every
    rank computes the identical global adversarial loss."""

    def gather(z, flags, weights=None):
        world = dist.get_world_size()
        rank = dist.get_rank()
        n_loc = torch.tensor([z.shape[0]], device=device)
        sizes = [torch.zeros_like(n_loc) for _ in range(world)]
        dist.all_gather(sizes, n_loc)
        sizes = [int(x.item()) for x in sizes]
        n_max = max(sizes)

        def pad_gather(t, dtype):
            buf = torch.zeros((n_max,) + tuple(t.shape[1:]), dtype=dtype, device=device)
            buf[: t.shape[0]] = t
            outs = [torch.zeros_like(buf) for _ in range(world)]
            dist.all_gather(outs, buf)
            return [o[: sizes[r]] for r, o in enumerate(outs)]

        z_parts = pad_gather(z.detach(), z.dtype)
        z_parts[rank] = z  # re-attach local gradients
        f_parts = pad_gather(flags, flags.dtype)
        out_z = torch.cat(z_parts)
        out_f = torch.cat(f_parts)
        if weights is None:
            return out_z, out_f, None
        w_parts = pad_gather(weights.detach(), weights.dtype)
        return out_z, out_f, torch.cat(w_parts)

    def gather_vec(w):
        world = dist.get_world_size()
        n_loc = torch.tensor([w.shape[0]], device=device)
        sizes = [torch.zeros_like(n_loc) for _ in range(world)]
        dist.all_gather(sizes, n_loc)
        sizes = [int(x.item()) for x in sizes]
        n_max = max(sizes)
        buf = torch.zeros((n_max,) + tuple(w.shape[1:]), dtype=w.dtype, device=device)
        buf[: w.shape[0]] = w.detach()
        outs = [torch.zeros_like(buf) for _ in range(world)]
        dist.all_gather(outs, buf)
        return torch.cat([o[: sizes[r]] for r, o in enumerate(outs)])

    gather.vec = gather_vec
    return gather


class _FixedLengthLoader:
    """Wraps a CustomGraphDataLoader to yield exactly `steps` batches per
    epoch (cycling if the underlying loader is shorter), so every rank runs
    the same number of optimizer steps and NCCL collectives stay aligned."""

    def __init__(self, dl, steps):
        self.dl = dl
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        it = iter(self.dl)
        for _ in range(self.steps):
            try:
                yield next(it)
            except StopIteration:
                it = iter(self.dl)
                yield next(it)


class _ShardView(torch.nn.Module):
    """Local-index view over the full model for one shard of atlases."""

    def __init__(self, base, shard):
        super().__init__()
        self.base = base
        self.encoder = torch.nn.ModuleDict(
            {f"atlas{li}": base.encoder[f"atlas{gi}"] for li, gi in enumerate(shard)})
        self.decoder = torch.nn.ModuleDict(
            {f"atlas{li}": base.decoder[f"atlas{gi}"] for li, gi in enumerate(shard)})
        self.scrna_seq_adj = torch.nn.ModuleDict(
            {f"atlas{li}": base.scrna_seq_adj[f"atlas{gi}"] for li, gi in enumerate(shard)}
            if len(getattr(base, "scrna_seq_adj", {})) else {})
        self.discriminator_single = base.discriminator_single
        self.discriminator_spatial = base.discriminator_spatial
        for name in ["gene_embedding", "gene_embedding_pretrained", "gene_embedding_new"]:
            if hasattr(base, name):
                setattr(self, name, getattr(base, name))


def _remap_state_dict(sd, shard):
    """view (local) key names -> full-model (global) key names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("base."):
            continue
        for pref in ("encoder.atlas", "decoder.atlas", "scrna_seq_adj.atlas"):
            if k.startswith(pref):
                li, rest = k[len(pref):].split(".", 1)
                k = f"{pref}{shard[int(li)]}.{rest}"
                break
        out[k] = v
    return out


def _worker(rank, world, X_paths, args, kneighbor, input_identity, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    import dgl.dataloading as dgl_dataload
    import scanpy as sc

    from fusemap import train_model as tm
    from fusemap.config import AnchorConfig, FlagConfig, ModelType
    from fusemap.data.loaders import (CustomGraphDataLoader, CustomGraphDataset,
                                 construct_data, construct_mask,
                                 get_feature_sparse)
    from fusemap.models.network import Fuse_network
    from fusemap.data.graph import get_allunique_gene_names, preprocess_raw
    from fusemap.training.train_model import (balance_weight, pretrain_model,
                                     read_model, train_model)
    from fusemap.logger import setup_logging
    from fusemap.utils import (load_snapshot, read_cell_embedding,
                               read_gene_embedding, save_obj, seed_all)

    assert AnchorConfig.anchor_lambda == 0, "sharded v0 requires FUSEMAP_ANCHOR_LAMBDA=0"

    save_dir = args.output_save_dir
    rank_dir = os.path.join(save_dir, "_shards", f"rank{rank}")
    Path(f"{rank_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    setup_logging(rank_dir if rank else save_dir)
    seed_all(0)

    # ---- full world (identical on every rank) ----
    X_input = []
    for ind, f in enumerate(X_paths):
        X = sc.read_h5ad(f)
        if "x" not in X.obs.columns:
            if "col" in X.obs.columns and "row" in X.obs.columns:
                X.obs["x"] = X.obs["col"]; X.obs["y"] = X.obs["row"]
            elif "spatial" in X.obsm:
                X.obs["x"] = X.obsm["spatial"][:, 0]; X.obs["y"] = X.obsm["spatial"][:, 1]
            else:
                raise ValueError(f"{f}: no spatial coordinates found")
        X.obs["name"] = f"section{ind}"
        X.obs["file_name"] = os.path.basename(f)
        X_input.append(X)

    n_global = len(X_input)
    ModelType.data_pth = None
    ModelType.save_dir = save_dir
    ModelType.kneighbor = kneighbor
    ModelType.input_identity = input_identity
    ModelType.n_atlas = n_global
    preprocess_raw(X_input, kneighbor, input_identity, ModelType.use_input.value, n_global, None)
    for i in range(n_global):
        X_input[i].var.index = [v.upper() for v in X_input[i].var.index]
    adatas = X_input
    n_obs_g = [a.shape[0] for a in adatas]
    input_dim_g = [a.n_vars for a in adatas]
    var_name_g = [list(a.var.index) for a in adatas]
    all_unique_genes = sorted(list(get_allunique_gene_names(*var_name_g)))

    seed_all(0)
    model = Fuse_network(
        ModelType.pca_dim.value, input_dim_g, ModelType.hidden_dim.value,
        ModelType.latent_dim.value, ModelType.dropout_rate.value, var_name_g,
        all_unique_genes, ModelType.use_input.value, n_global, input_identity,
        n_obs_g, ModelType.n_epochs.value, use_llm_gene_embedding="false")
    model.to(device)
    ModelType.use_llm_gene_embedding = "false"

    shard = list(range(rank, n_global, world))
    view = _ShardView(model, shard).to(device)
    hooks = _DistHooks(_shared_params(model), device)
    logging.info(f"[shard] rank {rank}/{world}: atlases {shard}")

    def set_world(idx_list, sd):
        ModelType.save_dir = sd
        ModelType.snapshot_path = f"{sd}/snapshot.pt"
        ModelType.n_atlas = len(idx_list)
        ModelType.n_obs = [n_obs_g[i] for i in idx_list]
        ModelType.input_dim = [input_dim_g[i] for i in idx_list]
        ModelType.var_name = [var_name_g[i] for i in idx_list]
        ModelType.input_identity = [input_identity[i] for i in idx_list]
        ModelType.epochs_run_pretrain = 0
        ModelType.epochs_run_final = 0
        sub_adatas = [adatas[i] for i in idx_list]
        adj_all, g_all = construct_data(len(idx_list), sub_adatas, ModelType.input_identity, view)
        feats = [get_feature_sparse(device, a.obsm["spatial_input"]) for a in sub_adatas]
        ds = [CustomGraphDataset(g, a, ModelType.use_input) for g, a in zip(g_all, sub_adatas)]
        dl = CustomGraphDataLoader(ds, dgl_dataload.MultiLayerFullNeighborSampler(1),
                                   ModelType.batch_size.value, shuffle=True, n_atlas=len(idx_list),
                                   drop_last=False, feature_all=feats, adj_all=adj_all,
                                   input_identity=ModelType.input_identity)
        dl_test = CustomGraphDataLoader(ds, dgl_dataload.MultiLayerFullNeighborSampler(1),
                                        ModelType.batch_size.value, shuffle=False, n_atlas=len(idx_list),
                                        drop_last=False, feature_all=feats, adj_all=adj_all,
                                        input_identity=ModelType.input_identity)
        seed_all(0)
        tr_mask, va_mask = construct_mask(len(idx_list), ds, g_all)
        import math
        global_steps = max(math.ceil(n / ModelType.batch_size.value) for n in n_obs_g)
        dl = _FixedLengthLoader(dl, global_steps)
        return sub_adatas, adj_all, g_all, feats, dl, dl_test, tr_mask, va_mask

    import fusemap.models.losses as loss_mod

    flagconfig = FlagConfig()
    hooks.flagconfig = flagconfig
    gather_fn = _make_dist_gather(device)

    # ---------------- Phase 1: sharded pretrain ----------------
    if not os.path.exists(f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt"):
        tm.ATLAS_LABEL_MAP = shard
        loss_mod.DIST_GATHER = gather_fn
        loss_mod.DIST_GATHER_VEC = gather_fn.vec
        loss_mod.DIST_ALLSUM = lambda x: (lambda t: (dist.all_reduce(t), t.item())[1])(torch.tensor([float(x)], device=device))
        loss_mod.DIST_SCALE = float(os.environ.get('FUSEMAP_DIST_DISC_SCALE', 1.0 / world))
        _, adj_s, g_s, feat_s, dl_s, _, trm_s, vam_s = set_world(shard, rank_dir)
        logging.info(f"[shard] rank {rank}: Phase 1 pretrain ({len(shard)} atlases)")
        pretrain_model(view, dl_s, feat_s, adj_s, device, trm_s, vam_s, flagconfig,
                       dist_hooks=hooks)
        tm.ATLAS_LABEL_MAP = None
        loss_mod.DIST_GATHER = None
        loss_mod.DIST_GATHER_VEC = None
        loss_mod.DIST_ALLSUM = None
        loss_mod.DIST_SCALE = 1.0
        torch.save(view.state_dict(), f"{rank_dir}/shard_pretrain.pt")
        dist.barrier()
        if rank == 0:
            full_sd = model.state_dict()
            for r in range(world):
                sh = list(range(r, n_global, world))
                sd = torch.load(os.path.join(save_dir, "_shards", f"rank{r}", "shard_pretrain.pt"),
                                map_location="cpu")
                full_sd.update(_remap_state_dict(sd, sh))
            torch.save(full_sd, f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt")
        dist.barrier()

    # rank 0 evaluates + balance weights on the full world
    set_world(list(range(n_global)), save_dir)
    model.load_state_dict(torch.load(
        f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt", map_location=device))
    if rank == 0:
        _, adj_f, g_f, feat_f, dl_f, dlt_f, trm_f, vam_f = set_world(list(range(n_global)), save_dir)
        if not os.path.exists(f"{save_dir}/latent_embeddings_all_single_pretrain.pkl"):
            read_model(model, dlt_f, g_f, feat_f, adj_f, device, ModelType, mode="pretrain")
        if not os.path.exists(f"{save_dir}/balance_weight_single.pkl"):
            balance_weight(model, adatas, save_dir, n_global, device)
        if not os.path.exists(f"{save_dir}/lambda_disc_single.pkl"):
            save_obj(flagconfig.lambda_disc_single, f"{save_dir}/lambda_disc_single")
    dist.barrier()

    # ---------------- Phase 4: sharded final ----------------
    if not os.path.exists(f"{save_dir}/trained_model/FuseMap_final_model_final.pt"):
        with open(f"{save_dir}/balance_weight_single.pkl", "rb") as f:
            bw_s = pickle.load(f)
        with open(f"{save_dir}/balance_weight_spatial.pkl", "rb") as f:
            bw_p = pickle.load(f)
        save_obj([bw_s[i] for i in shard], f"{rank_dir}/balance_weight_single")
        save_obj([bw_p[i] for i in shard], f"{rank_dir}/balance_weight_spatial")

        tm.ATLAS_LABEL_MAP = shard
        loss_mod.DIST_GATHER = gather_fn
        loss_mod.DIST_GATHER_VEC = gather_fn.vec
        loss_mod.DIST_ALLSUM = lambda x: (lambda t: (dist.all_reduce(t), t.item())[1])(torch.tensor([float(x)], device=device))
        loss_mod.DIST_SCALE = float(os.environ.get('FUSEMAP_DIST_DISC_SCALE', 1.0 / world))
        _, adj_s, g_s, feat_s, dl_s, _, trm_s, vam_s = set_world(shard, rank_dir)
        if os.path.exists(f"{rank_dir}/snapshot.pt"):
            os.remove(f"{rank_dir}/snapshot.pt")
        logging.info(f"[shard] rank {rank}: Phase 4 final ({len(shard)} atlases)")
        train_model(view, dl_s, feat_s, adj_s, device, trm_s, vam_s, flagconfig,
                    dist_hooks=hooks)
        tm.ATLAS_LABEL_MAP = None
        loss_mod.DIST_GATHER = None
        loss_mod.DIST_GATHER_VEC = None
        loss_mod.DIST_ALLSUM = None
        loss_mod.DIST_SCALE = 1.0
        torch.save(view.state_dict(), f"{rank_dir}/shard_final.pt")
        dist.barrier()
        if rank == 0:
            full_sd = model.state_dict()
            for r in range(world):
                sh = list(range(r, n_global, world))
                sd = torch.load(os.path.join(save_dir, "_shards", f"rank{r}", "shard_final.pt"),
                                map_location="cpu")
                full_sd.update(_remap_state_dict(sd, sh))
            torch.save(full_sd, f"{save_dir}/trained_model/FuseMap_final_model_final.pt")
        dist.barrier()

    # ---------------- Phase 5: rank-0 final evaluation + outputs ----------------
    if rank == 0:
        model.load_state_dict(torch.load(
            f"{save_dir}/trained_model/FuseMap_final_model_final.pt", map_location=device))
        _, adj_f, g_f, feat_f, dl_f, dlt_f, trm_f, vam_f = set_world(list(range(n_global)), save_dir)
        if not os.path.exists(f"{save_dir}/latent_embeddings_all_single_final.pkl"):
            read_model(model, dlt_f, g_f, feat_f, adj_f, device, ModelType, mode="final")
        read_gene_embedding(model, all_unique_genes, save_dir, n_global, var_name_g)
        read_cell_embedding(adatas, save_dir, args.keep_celltype, args.keep_tissueregion,
                            use_key=ModelType.use_key.value)
        logging.info("[shard] Done!")
    dist.barrier()
    dist.destroy_process_group()


def spatial_integrate_sharded(input_data_folder_path, output_save_dir, world_size,
                              keep_celltype="", keep_tissueregion="", port=29517):
    """Multi-GPU (atlas-sharded) version of :func:`fusemap.integrate`."""
    import torch.multiprocessing as mp
    from types import SimpleNamespace

    folder = Path(input_data_folder_path)
    X_paths = sorted(str(p) for p in folder.iterdir() if p.suffix == ".h5ad" and p.is_file())
    if not X_paths:
        raise ValueError(f"no .h5ad in {folder}")
    args = SimpleNamespace(output_save_dir=str(output_save_dir), keep_celltype=keep_celltype,
                           keep_tissueregion=keep_tissueregion,
                           use_llm_gene_embedding="false", pretrain_model_path="")
    kneighbor = ["delaunay"] * len(X_paths)
    input_identity = ["ST"] * len(X_paths)
    Path(output_save_dir).mkdir(parents=True, exist_ok=True)
    mp.spawn(_worker, args=(world_size, X_paths, args, kneighbor, input_identity, port),
             nprocs=world_size, join=True)
