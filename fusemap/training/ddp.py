"""Data-parallel (DDP-style) multi-GPU integration. EXPERIMENTAL but validated.

v8 (2026-09-09) results, 2 GPUs, vs the single-GPU reference:

- correctness invariant: replica param-checksum drift = 0.000e+00 in every run
  (identical init + averaged grads keep all ranks bit-identical)
- data2, anchors off:  cell iLISI 0.341 vs single-GPU seed band 0.25-0.33
  (at/above band; the atlas-SHARDED v1-v7 approach scored 0.03-0.09), 10.8 min
- data3 4-way, anchors ON (lambda=0.3 default): iLISI 0.398 vs one single-GPU
  seed at 0.443 (seed-noise quantification ongoing); anchors need no extra
  communication because every rank holds the full data
- v8.1 "exact" mode (SyncBatchNorm + per-rank batch/world + single-GPU step
  count): 0.370 and 2x slower - for a graph NN the sampled-subgraph statistics
  change with the per-rank seed count, so "exact" is not exact; NOT recommended.

Default mode is THROUGHPUT: each rank runs 1/world of the steps per epoch on
its own shuffle order, gradients averaged every step (effective batch =
world x batch_size). FUSEMAP_DDP_MODE=exact keeps the experimental variant.

Design vs the parked atlas-sharded trainer (sharded.py): every rank holds the
FULL model and FULL data; only per-step batches differ. Each rank sees all
atlases every step, so the discriminator keeps its cross-atlas noise-std
component (the mixing mechanism sharding broke), and one averaged update per
step matches single-GPU adversarial dynamics. LR schedulers / early stopping /
lambda_disc stay in lockstep via dist_hooks.sync_value broadcast from rank 0.
"""

import logging
import math
import os
import pickle
from pathlib import Path

import torch
import torch.distributed as dist

from fusemap.training.sharded import _FixedLengthLoader

__all__ = ["spatial_integrate_ddp"]


class _DDPHooks:
    """All-parameter gradient averaging + rank-0 scalar broadcast."""

    def __init__(self, params, device, world):
        self.params = params
        self.device = device
        self.world = world
        self.flagconfig = None

    def sync_grads(self, _model=None):
        # Every rank computes the SAME loss structure on different batches, so
        # grad None-ness is identical across ranks and collectives stay aligned
        # while skipping absent grads (e.g. only disc params during the disc
        # step) - this cuts per-step comm to the params actually updated.
        for p in self.params:
            if p.grad is None:
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world)
        if self.flagconfig is not None:
            t = torch.tensor([float(self.flagconfig.lambda_disc_single)], device=self.device)
            dist.broadcast(t, src=0)
            self.flagconfig.lambda_disc_single = t.item()

    def sync_value(self, x):
        t = torch.tensor([float(x)], device=self.device)
        dist.broadcast(t, src=0)
        return t.item()


def _worker(rank, world, X_paths, args, kneighbor, input_identity, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    import dgl.dataloading as dgl_dataload
    import scanpy as sc

    from fusemap.config import ModelType, FlagConfig
    from fusemap.data.loaders import (CustomGraphDataLoader, CustomGraphDataset,
                                      construct_data, construct_mask,
                                      get_feature_sparse)
    from fusemap.models.network import Fuse_network
    from fusemap.data.graph import get_allunique_gene_names, preprocess_raw
    from fusemap.training.train_model import (balance_weight, pretrain_model,
                                              read_model, train_model)
    from fusemap.logger import setup_logging
    from fusemap.utils import (read_cell_embedding, read_gene_embedding,
                               save_obj, seed_all)

    save_dir = args.output_save_dir
    rank_dir = os.path.join(save_dir, "_ddp", f"rank{rank}")
    Path(f"{rank_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    Path(f"{save_dir}/trained_model").mkdir(parents=True, exist_ok=True)
    setup_logging(rank_dir if rank else save_dir)
    seed_all(0)

    # ---- full world, identical on every rank ----
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

    n_atlas = len(X_input)
    ModelType.data_pth = None
    ModelType.save_dir = save_dir
    ModelType.kneighbor = kneighbor
    ModelType.input_identity = input_identity
    ModelType.n_atlas = n_atlas
    preprocess_raw(X_input, kneighbor, input_identity, ModelType.use_input.value, n_atlas, None)
    for i in range(n_atlas):
        X_input[i].var.index = [v.upper() for v in X_input[i].var.index]
    adatas = X_input
    n_obs = [a.shape[0] for a in adatas]
    input_dim = [a.n_vars for a in adatas]
    var_name = [list(a.var.index) for a in adatas]
    all_unique_genes = sorted(list(get_allunique_gene_names(*var_name)))

    ModelType.n_obs = n_obs
    ModelType.input_dim = input_dim
    ModelType.var_name = var_name
    ModelType.epochs_run_pretrain = 0
    ModelType.epochs_run_final = 0
    ModelType.use_llm_gene_embedding = "false"

    # identical init on every rank (same seed), then averaged grads keep the
    # replicas identical for the rest of training
    seed_all(0)
    model = Fuse_network(
        ModelType.pca_dim.value, input_dim, ModelType.hidden_dim.value,
        ModelType.latent_dim.value, ModelType.dropout_rate.value, var_name,
        all_unique_genes, ModelType.use_input.value, n_atlas, input_identity,
        n_obs, ModelType.n_epochs.value, use_llm_gene_embedding="false")
    model.to(device)

    # exact mode (default): per-rank batch = batch_size/world with the
    # single-GPU step count, and SyncBatchNorm so normalization statistics are
    # computed over the GLOBAL per-step batch - per-step dynamics match
    # single-GPU training and the speedup comes from splitting each step's
    # compute. throughput mode: full per-rank batches, 1/world of the steps
    # (effective batch = world x batch_size - fastest, slightly softer mixing).
    ddp_mode = os.environ.get("FUSEMAP_DDP_MODE", "throughput")
    if world > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)

    hooks = _DDPHooks(list(model.parameters()), device, world)
    logging.info(f"[ddp] rank {rank}/{world}: full model, {n_atlas} atlases, "
                 f"{sum(n_obs)} cells")

    def make_world(sd, per_rank_steps):
        ModelType.save_dir = sd
        ModelType.snapshot_path = f"{sd}/snapshot.pt"
        adj_all, g_all = construct_data(n_atlas, adatas, input_identity, model)
        feats = [get_feature_sparse(device, a.obsm["spatial_input"]) for a in adatas]
        ds = [CustomGraphDataset(g, a, ModelType.use_input) for g, a in zip(g_all, adatas)]
        dl = CustomGraphDataLoader(ds, dgl_dataload.MultiLayerFullNeighborSampler(1),
                                   rank_batch, shuffle=True, n_atlas=n_atlas,
                                   drop_last=False, feature_all=feats, adj_all=adj_all,
                                   input_identity=input_identity)
        dl_test = CustomGraphDataLoader(ds, dgl_dataload.MultiLayerFullNeighborSampler(1),
                                        ModelType.batch_size.value, shuffle=False, n_atlas=n_atlas,
                                        drop_last=False, feature_all=feats, adj_all=adj_all,
                                        input_identity=input_identity)
        # masks must be IDENTICAL across ranks (rank0's val loss steers all)
        seed_all(0)
        tr_mask, va_mask = construct_mask(n_atlas, ds, g_all)
        if per_rank_steps is not None:
            dl = _FixedLengthLoader(dl, per_rank_steps)
        return adj_all, g_all, feats, dl, dl_test, tr_mask, va_mask

    if ddp_mode == "exact":
        rank_batch = max(1, ModelType.batch_size.value // world)
        steps = max(math.ceil(n / ModelType.batch_size.value) for n in n_obs)
    else:  # throughput
        rank_batch = ModelType.batch_size.value
        steps = max(math.ceil(math.ceil(n / ModelType.batch_size.value) / world) for n in n_obs)
    logging.info(f"[ddp] mode={ddp_mode}: per-rank batch {rank_batch}, {steps} steps/epoch")

    flagconfig = FlagConfig()
    hooks.flagconfig = flagconfig

    # ---------------- Phase 1: data-parallel pretrain ----------------
    if not os.path.exists(f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt"):
        adj_a, g_a, feat_a, dl_a, _, trm, vam = make_world(rank_dir, steps)
        # diverge ONLY the sampling / noise RNG streams; parameters are already
        # built identically and stay identical through averaged gradients
        seed_all(1000 + rank)
        logging.info(f"[ddp] rank {rank}: Phase 1 pretrain ({steps} steps/epoch/rank)")
        pretrain_model(model, dl_a, feat_a, adj_a, device, trm, vam, flagconfig,
                       dist_hooks=hooks)
        dist.barrier()
        if rank == 0:
            torch.save(model.state_dict(),
                       f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt")
        dist.barrier()

    # rank 0: pretrain evaluation + balance weights (identical replicas -> no merge)
    model.load_state_dict(torch.load(
        f"{save_dir}/trained_model/FuseMap_pretrain_model_final.pt", map_location=device))
    if rank == 0:
        adj_f, g_f, feat_f, dl_f, dlt_f, trm_f, vam_f = make_world(save_dir, None)
        if not os.path.exists(f"{save_dir}/latent_embeddings_all_single_pretrain.pkl"):
            read_model(model, dlt_f, g_f, feat_f, adj_f, device, ModelType, mode="pretrain")
        if not os.path.exists(f"{save_dir}/balance_weight_single.pkl"):
            balance_weight(model, adatas, save_dir, n_atlas, device)
        if not os.path.exists(f"{save_dir}/lambda_disc_single.pkl"):
            save_obj(flagconfig.lambda_disc_single, f"{save_dir}/lambda_disc_single")
    dist.barrier()

    # ---------------- Phase 4: data-parallel final ----------------
    if not os.path.exists(f"{save_dir}/trained_model/FuseMap_final_model_final.pt"):
        with open(f"{save_dir}/balance_weight_single.pkl", "rb") as f:
            bw_s = pickle.load(f)
        with open(f"{save_dir}/balance_weight_spatial.pkl", "rb") as f:
            bw_p = pickle.load(f)
        save_obj(bw_s, f"{rank_dir}/balance_weight_single")
        save_obj(bw_p, f"{rank_dir}/balance_weight_spatial")

        adj_a, g_a, feat_a, dl_a, _, trm, vam = make_world(rank_dir, steps)
        if os.path.exists(f"{rank_dir}/snapshot.pt"):
            os.remove(f"{rank_dir}/snapshot.pt")
        seed_all(2000 + rank)
        logging.info(f"[ddp] rank {rank}: Phase 4 final ({steps} steps/epoch/rank)")
        train_model(model, dl_a, feat_a, adj_a, device, trm, vam, flagconfig,
                    dist_hooks=hooks)
        dist.barrier()
        if rank == 0:
            torch.save(model.state_dict(),
                       f"{save_dir}/trained_model/FuseMap_final_model_final.pt")
        dist.barrier()

    # ---------------- replica-identity check (correctness invariant) ----------------
    # Averaged grads + identical init must keep all replicas bit-identical.
    with torch.no_grad():
        checksum = torch.stack([p.double().sum() for p in model.parameters()]).sum()
    sums = [torch.zeros_like(checksum) for _ in range(world)]
    dist.all_gather(sums, checksum)
    if rank == 0:
        drift = max(abs((s - sums[0]).item()) for s in sums)
        logging.info(f"[ddp] replica param-checksum drift across ranks: {drift:.3e}")
        if drift > 1e-6:
            logging.warning("[ddp] REPLICA DIVERGENCE DETECTED - results invalid")

    # ---------------- Phase 5: rank-0 outputs ----------------
    if rank == 0:
        model.load_state_dict(torch.load(
            f"{save_dir}/trained_model/FuseMap_final_model_final.pt", map_location=device))
        adj_f, g_f, feat_f, dl_f, dlt_f, trm_f, vam_f = make_world(save_dir, None)
        if not os.path.exists(f"{save_dir}/latent_embeddings_all_single_final.pkl"):
            read_model(model, dlt_f, g_f, feat_f, adj_f, device, ModelType, mode="final")
        read_gene_embedding(model, all_unique_genes, save_dir, n_atlas, var_name)
        read_cell_embedding(adatas, save_dir, args.keep_celltype, args.keep_tissueregion,
                            use_key=ModelType.use_key.value)
        logging.info("[ddp] Done!")
    dist.barrier()
    dist.destroy_process_group()


def spatial_integrate_ddp(input_data_folder_path, output_save_dir, world_size,
                          keep_celltype="", keep_tissueregion="", port=29533):
    """Multi-GPU data-parallel version of :func:`fusemap.integrate` (experimental)."""
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
