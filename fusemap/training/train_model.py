import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import os
from fusemap.config import *
from fusemap.data.loaders import *
from fusemap.utils import *
from fusemap.models.losses import *
import anndata as ad
import torch
from tqdm import tqdm
import scanpy as sc
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle


def _is_prepared_blocks(blocks_all):
    """True when batches were pre-sliced in DataLoader workers
    (see fusemap.dataset.CustomGraphDataLoader)."""
    first = blocks_all[0]
    return isinstance(first, dict) and "row_index" in first


def _get_data_prepared(blocks_all, train_mask, device, model):
    """Fast path for worker-prepared batches: tensors are ready on CPU
    (pinned) and only need an asynchronous host-to-device copy. Learnable
    (scrna) adjacency blocks are still sliced from the model here, in the
    main process, exactly as before."""
    row_index_all = {}
    col_index_all = {}
    batch_features_all = []
    adj_all_block = []
    adj_all_block_dis = []
    for i in range(ModelType.n_atlas):
        block = blocks_all[i]
        row_index_all[i] = block["row_index"]
        col_index_all[i] = block["col_index"]
        batch_features_all.append(block["feature"].to(device, non_blocking=True))
        if ModelType.input_identity[i] == "ST":
            adj_block = block["adj_block"].to(device, non_blocking=True)
            adj_all_block.append(adj_block)
            adj_all_block_dis.append(adj_block)
        else:
            adj_learned = model.scrna_seq_adj["atlas" + str(i)]()[
                block["row_index"], :
            ][:, block["col_index"]]
            adj_all_block.append(adj_learned)
            adj_all_block_dis.append(adj_learned.detach())

    train_mask_batch_single = [
        train_mask[i][row_index_all[i]] for i in range(ModelType.n_atlas)
    ]
    train_mask_batch_spatial = [
        train_mask[i][col_index_all[i]] for i in range(ModelType.n_atlas)
    ]

    ### discriminator flags
    flag_shape_single = [len(row_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_single = torch.cat(
        [torch.full((x,), _atlas_label(i)) for i, x in enumerate(flag_shape_single)]
    )
    flag_source_cat_single = flag_all_single.long().to(device)

    flag_shape_spatial = [len(col_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_spatial = torch.cat(
        [torch.full((x,), _atlas_label(i)) for i, x in enumerate(flag_shape_spatial)]
    )
    flag_source_cat_spatial = flag_all_spatial.long().to(device)

    return (
        batch_features_all,
        adj_all_block,
        adj_all_block_dis,
        train_mask_batch_single,
        train_mask_batch_spatial,
        flag_source_cat_single,
        flag_source_cat_spatial,
        row_index_all,
        col_index_all,
    )


# Sharded training support: maps local atlas index -> global atlas index for
# discriminator labels. None (default) = identity = original behavior.
ATLAS_LABEL_MAP = None


def _atlas_label(i):
    return i if ATLAS_LABEL_MAP is None else ATLAS_LABEL_MAP[i]


def get_data(blocks_all, feature_all, adj_all, train_mask, device, model):
    if _is_prepared_blocks(blocks_all):
        return _get_data_prepared(blocks_all, train_mask, device, model)

    row_index_all = {}
    col_index_all = {}
    for i_atlas in range(ModelType.n_atlas):
        row_index = list(blocks_all[i_atlas]["spatial"][0])
        col_index = list(blocks_all[i_atlas]["spatial"][1])
        row_index_all[i_atlas] = torch.sort(torch.vstack(row_index).flatten())[
            0
        ].tolist()
        col_index_all[i_atlas] = torch.sort(torch.vstack(col_index).flatten())[
            0
        ].tolist()

    batch_features_all = [
        torch.FloatTensor(feature_all[i][row_index_all[i], :].toarray()).to(device)
        for i in range(ModelType.n_atlas)
    ]

    # Each ST adjacency block is computed once and reused for both the AE and
    # the discriminator inputs (previously computed twice, identically).
    adj_block_st = {
        i: torch.FloatTensor(
            adj_all[i][row_index_all[i], :].tocsc()[:, col_index_all[i]].todense()
        ).to(device)
        for i in range(ModelType.n_atlas)
        if ModelType.input_identity[i] == "ST"
    }

    adj_all_block = [
        adj_block_st[i]
        if ModelType.input_identity[i] == "ST"
        else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
            :, col_index_all[i]
        ]
        for i in range(ModelType.n_atlas)
    ]
    adj_all_block_dis = [
        adj_block_st[i]
        if ModelType.input_identity[i] == "ST"
        else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
            :, col_index_all[i]
        ].detach()
        for i in range(ModelType.n_atlas)
    ]

    train_mask_batch_single = [
        train_mask_i[row_index_all[blocks_all_ind]]
        for train_mask_i, blocks_all_ind in zip(train_mask, blocks_all)
    ]
    train_mask_batch_spatial = [
        train_mask_i[col_index_all[blocks_all_ind]]
        for train_mask_i, blocks_all_ind in zip(train_mask, blocks_all)
    ]

    ### discriminator flags
    flag_shape_single = [len(row_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_single = torch.cat(
        [torch.full((x,), _atlas_label(i)) for i, x in enumerate(flag_shape_single)]
    )
    flag_source_cat_single = flag_all_single.long().to(device)

    flag_shape_spatial = [len(col_index_all[i]) for i in range(ModelType.n_atlas)]
    flag_all_spatial = torch.cat(
        [torch.full((x,), _atlas_label(i)) for i, x in enumerate(flag_shape_spatial)]
    )
    flag_source_cat_spatial = flag_all_spatial.long().to(device)

    return (
        batch_features_all,
        adj_all_block,
        adj_all_block_dis,
        train_mask_batch_single,
        train_mask_batch_spatial,
        flag_source_cat_single,
        flag_source_cat_spatial,
        row_index_all,
        col_index_all,
    )


def _mnn_pairs(x, y, k):
    """Mutual nearest-neighbor pairs between two latent matrices.

    kNN (cosine) is computed in each direction and only mutual pairs are kept.
    Brute-force (sklearn NearestNeighbors) is fine for the FuseMap scale
    (~10-40k cells x 64 dims). Returns a list of (i_in_x, j_in_y) index tuples.
    """
    n_x, n_y = x.shape[0], y.shape[0]
    if n_x == 0 or n_y == 0:
        return []
    k_xy = min(k, n_y)
    k_yx = min(k, n_x)

    def _cos_knn_gpu(q, r, kk):
        # exact cosine kNN via torch chunked matmul (GPU if available):
        # same metric/k as sklearn brute cosine, ~10-20x faster at 50-90k rows
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        qn = torch.nn.functional.normalize(
            torch.as_tensor(q, dtype=torch.float32, device=dev), dim=1)
        rn = torch.nn.functional.normalize(
            torch.as_tensor(r, dtype=torch.float32, device=dev), dim=1)
        idxs, dists = [], []
        for st in range(0, qn.shape[0], 8192):
            sims = qn[st:st + 8192] @ rn.T
            top = sims.topk(kk, dim=1, largest=True)
            idxs.append(top.indices.cpu())
            dists.append((1.0 - top.values).cpu())
        del qn, rn
        return torch.cat(dists).numpy(), torch.cat(idxs).numpy()

    dist_x2y, idx_x2y = _cos_knn_gpu(x, y, k_xy)
    idx_y2x = _cos_knn_gpu(y, x, k_yx)[1]
    y_neighbor_sets = [set(row.tolist()) for row in idx_y2x]
    sim_min = AnchorConfig.anchor_sim_threshold
    pairs = []
    n_raw = 0
    for xi in range(n_x):
        for jj, yj in enumerate(idx_x2y[xi]):
            if xi in y_neighbor_sets[int(yj)]:
                n_raw += 1
                if 1.0 - dist_x2y[xi, jj] >= sim_min:
                    pairs.append((xi, int(yj)))
    return pairs, n_raw


def _infer_full_latents(model, spatial_dataloader, feature_all, adj_all, mask, device):
    """Compute full-dataset single (z_mean) and spatial (z_spatial) latents,
    batch-wise, scattering each batch into per-atlas [n_obs_i, d] tensors by
    global cell index (same inference pattern as read_model). One pass over the
    dataloader covers every cell (largest atlas drives the loop; smaller atlases
    are cycled). Model is temporarily set to eval() and restored afterwards.
    """
    d = ModelType.latent_dim.value
    single = [
        torch.zeros((ModelType.n_obs[i], d), device=device)
        for i in range(ModelType.n_atlas)
    ]
    spatial = [
        torch.zeros((ModelType.n_obs[i], d), device=device)
        for i in range(ModelType.n_atlas)
    ]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for blocks_all in spatial_dataloader:
            (
                batch_features_all,
                adj_all_block,
                _adj_all_block_dis,
                _tmb_single,
                _tmb_spatial,
                _flag_single,
                _flag_spatial,
                row_index_all,
                col_index_all,
            ) = get_data(blocks_all, feature_all, adj_all, mask, device, model)
            for i in range(ModelType.n_atlas):
                z_i = model.encoder["atlas" + str(i)](
                    batch_features_all[i], adj_all_block[i]
                )
                ri = torch.as_tensor(row_index_all[i], device=device, dtype=torch.long)
                ci = torch.as_tensor(col_index_all[i], device=device, dtype=torch.long)
                single[i][ri] = z_i[3].detach()
                spatial[i][ci] = z_i[2].detach()
    if was_training:
        model.train()
    return single, spatial


def refresh_anchors(
    model,
    spatial_dataloader,
    feature_all,
    adj_all,
    mask,
    device,
    anchor_state,
    epoch,
    phase,
):
    """Recompute cross-sample MNN anchors in the current shared latent space.

    Computes full-dataset latents, finds MNN pairs between every atlas pair
    (only needs to work well for n_atlas=2 but is implemented pairwise), caches
    the (detached) full latents for one-sided partner lookup, and stores per-cell
    partner index arrays. Logs the number of anchor pairs found at each refresh.
    """
    if anchor_state is None or AnchorConfig.anchor_lambda == 0:
        return
    single, spatial = _infer_full_latents(
        model, spatial_dataloader, feature_all, adj_all, mask, device
    )
    anchor_state.single_cache = [s.detach() for s in single]
    anchor_state.spatial_cache = [s.detach() for s in spatial]

    partner = {}
    n_pairs_total = 0
    dist_weighted_sum = 0.0
    dist_weight = 0
    k = AnchorConfig.anchor_k
    for a in range(ModelType.n_atlas):
        for b in range(a + 1, ModelType.n_atlas):
            xa = single[a].detach().cpu().numpy()
            xb = single[b].detach().cpu().numpy()
            pairs, n_raw = _mnn_pairs(xa, xb, k)
            n_simfilt = len(pairs)
            if AnchorConfig.anchor_stable_only:
                cur = set(pairs)
                prev = anchor_state.prev_pairs.get((a, b))
                anchor_state.prev_pairs[(a, b)] = cur
                if prev is not None:
                    pairs = sorted(cur & prev)
            logging.info(
                f"[FuseMap anchor] pair filter ({a},{b}): raw={n_raw} "
                f"sim>={AnchorConfig.anchor_sim_threshold}: {n_simfilt} "
                f"stable: {len(pairs)}"
            )
            pa = torch.full(
                (ModelType.n_obs[a],), -1, dtype=torch.long, device=device
            )
            pb = torch.full(
                (ModelType.n_obs[b],), -1, dtype=torch.long, device=device
            )
            for (ia, ib) in pairs:
                pa[ia] = ib
                pb[ib] = ia
            partner[(a, b)] = pa
            partner[(b, a)] = pb
            n_pairs_total += len(pairs)
            if pairs:
                ta = torch.as_tensor(
                    [pr[0] for pr in pairs], device=device, dtype=torch.long
                )
                tb = torch.as_tensor(
                    [pr[1] for pr in pairs], device=device, dtype=torch.long
                )
                dd = (
                    torch.sqrt(((single[a][ta] - single[b][tb]) ** 2).sum(dim=1))
                    .mean()
                    .item()
                )
                dist_weighted_sum += dd * len(pairs)
                dist_weight += len(pairs)

    anchor_state.partner = partner
    anchor_state.n_pairs = n_pairs_total
    anchor_state.has_anchors = n_pairs_total > 0
    anchor_state.mean_dist = (
        dist_weighted_sum / dist_weight if dist_weight > 0 else float("nan")
    )
    logging.info(
        f"\n\n[FuseMap anchor] {phase} epoch {epoch}: found {n_pairs_total} "
        f"MNN anchor pairs (k={k}); mean cross-atlas single-latent distance "
        f"{anchor_state.mean_dist}\n"
    )


def pretrain_model(
    model,
    spatial_dataloader,
    feature_all,
    adj_all,
    device,
    train_mask,
    val_mask,
    flagconfig,
    dist_hooks=None,
):
    loss_atlas_val_best = float("inf")
    patience_counter = 0

    optimizer_dis = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.discriminator_single.parameters(),
            model.discriminator_spatial.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    optimizer_ae = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
            model.scrna_seq_adj.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    scheduler_dis = ReduceLROnPlateau(
        optimizer_dis,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )
    scheduler_ae = ReduceLROnPlateau(
        optimizer_ae,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )

    anchor_state = (
        AnchorState()
        if AnchorConfig.anchor_lambda > 0 and AnchorConfig.anchor_in_pretrain
        else None
    )

    for epoch in tqdm(
        range(ModelType.epochs_run_pretrain + 1, ModelType.n_epochs.value)
    ):
        loss_dis = 0
        loss_ae_dis = 0
        loss_all_item = 0
        if ModelType.use_llm_gene_embedding=='combine':
            loss_gene_embedding = 0
        loss_atlas_i = {}
        for i in range(ModelType.n_atlas):
            loss_atlas_i[i] = 0
        loss_atlas_val = 0
        anneal = (
            max(1 - (epoch - 1) / flagconfig.align_anneal, 0)
            if flagconfig.align_anneal
            else 0
        )

        if (
            anchor_state is not None
            and epoch >= AnchorConfig.anchor_start_epoch
            and (epoch - AnchorConfig.anchor_start_epoch)
            % AnchorConfig.anchor_refresh_epochs
            == 0
        ):
            refresh_anchors(
                model,
                spatial_dataloader,
                feature_all,
                adj_all,
                train_mask,
                device,
                anchor_state,
                epoch,
                "pretrain",
            )

        model.train()

        for blocks_all in tqdm(spatial_dataloader):
            (
                batch_features_all,
                adj_all_block,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flag_source_cat_single,
                flag_source_cat_spatial,
                row_index_all,
                col_index_all,
            ) = get_data(blocks_all, feature_all, adj_all, train_mask, device, model)

            # Train discriminator part
            loss_part1 = compute_dis_loss_pretrain(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part1["dis"].backward()
            if dist_hooks is not None:
                dist_hooks.sync_grads(model)
            optimizer_dis.step()
            loss_dis += loss_part1["dis"].item()

            # Train AE part
            loss_part2 = compute_ae_loss_pretrain(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flagconfig,
                anchor_state=anchor_state,
                row_index_all=row_index_all,
                col_index_all=col_index_all,
            )
            model.zero_grad(set_to_none=True)
            loss_part2["loss_all"].backward()
            if dist_hooks is not None:
                dist_hooks.sync_grads(model)
            optimizer_ae.step()

            if ModelType.use_llm_gene_embedding=='combine':
                # for _ in range(3):
                    loss_part3 = compute_gene_embedding_loss(model)
                    model.zero_grad(set_to_none=True)
                    loss_part3.backward()
                    if dist_hooks is not None:
                        dist_hooks.sync_grads(model)
                    optimizer_ae.step()
                    loss_gene_embedding += loss_part3.item()

            for i in range(ModelType.n_atlas):
                loss_atlas_i[i] += loss_part2["loss_AE_all"][i].item()
            loss_all_item += loss_part2["loss_all"].item()
            loss_ae_dis += loss_part2["dis_ae"].item()

        if dist_hooks is not None:
            flagconfig.lambda_disc_single = dist_hooks.sync_value(flagconfig.lambda_disc_single)
        flagconfig.align_anneal /= 2

        if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
            logging.info(
                f"\n\nTrain Epoch {epoch}/{ModelType.n_epochs}, \
            Loss dis: {loss_dis / len(spatial_dataloader)},\
            Loss AE: {[i / len(spatial_dataloader) for i in loss_atlas_i.values()]} , \
            Loss AE sum: {sum([i / len(spatial_dataloader) for i in loss_atlas_i.values()])} , \
            Loss ae dis:{loss_ae_dis / len(spatial_dataloader)},\
            Loss all:{loss_all_item / len(spatial_dataloader)}\n"
            )
            if ModelType.use_llm_gene_embedding=='combine':
                logging.info(
                    f"Train Epoch {epoch}/{ModelType.n_epochs}, \
                Loss gene embedding: {loss_gene_embedding / len(spatial_dataloader)}\n"
                )

        save_snapshot(model, epoch, ModelType.epochs_run_final, ModelType.snapshot_path,ModelType.verbose.value)

        if not os.path.exists(f"{ModelType.save_dir}/lambda_disc_single.pkl"):
            save_obj(
                flagconfig.lambda_disc_single,
                f"{ModelType.save_dir}/lambda_disc_single",
            )

        ################# validation
        if epoch > ModelType.TRAIN_WITHOUT_EVAL.value:
            model.eval()
            with torch.no_grad():
                for blocks_all in spatial_dataloader:

                    (
                        batch_features_all,
                        adj_all_block,
                        adj_all_block_dis,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        _,
                        _,
                    ) = get_data(
                        blocks_all, feature_all, adj_all, val_mask, device, model
                    )

                    # val AE part
                    loss_part2 = compute_ae_loss_pretrain(
                        model,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        anneal,
                        batch_features_all,
                        adj_all_block,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        flagconfig,
                    )

                    for i in range(ModelType.n_atlas):
                        loss_atlas_val += loss_part2["loss_AE_all"][i].item()
                        # if np.isnan(loss_part2['loss_AE_all'][i].item()):
                        #     p=0

                loss_atlas_val = (
                    loss_atlas_val / len(spatial_dataloader) / ModelType.n_atlas
                )

                if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                    logging.info(
                        f"\n\nValidation Epoch {epoch + 1}/{ModelType.n_epochs}, \
                    Loss AE validation: {loss_atlas_val} \n"
                    )

            if dist_hooks is not None:
                loss_atlas_val = dist_hooks.sync_value(loss_atlas_val)
            scheduler_dis.step(loss_atlas_val)
            scheduler_ae.step(loss_atlas_val)
            current_lr = optimizer_dis.param_groups[0]["lr"]
            if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                logging.info(f"\n\ncurrent lr:{current_lr}\n")

            # If the loss is lower than the best loss so far, save the model And reset the patience counter
            if loss_atlas_val < loss_atlas_val_best:
                loss_atlas_val_best = loss_atlas_val
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                )

            else:
                patience_counter += 1

            # If the patience counter is greater than or equal to the patience limit, stop training
            if patience_counter >= ModelType.patience_limit_pretrain.value:
                logging.info("\n\nEarly stopping due to loss not improving - patience count\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break
            if current_lr < ModelType.lr_limit_pretrain.value:
                logging.info("\n\nEarly stopping due to loss not improving - learning rate\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break

        # torch.save(model.state_dict(), f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_{epoch}.pt")

    if os.path.exists(f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt"):
        os.rename(
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model.pt",
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt",
        )
        logging.info("\n\nFile name changed in the end\n")


def train_model(
    model,
    spatial_dataloader,
    feature_all,
    adj_all,
    device,
    train_mask,
    val_mask,
    flagconfig,
    dist_hooks=None,
):
    with open(f"{ModelType.save_dir}/balance_weight_single.pkl", "rb") as openfile:
        balance_weight_single = pickle.load(openfile)
    with open(f"{ModelType.save_dir}/balance_weight_spatial.pkl", "rb") as openfile:
        balance_weight_spatial = pickle.load(openfile)
    balance_weight_single = [i.to(device) for i in balance_weight_single]
    balance_weight_spatial = [i.to(device) for i in balance_weight_spatial]

    anchor_pair_single = anchor_pair_spatial = None
    _pp = f"{ModelType.save_dir}/balance_weight_pair_single.pkl"
    if AnchorConfig.anchor_lambda > 0 and os.path.exists(_pp):
        with open(_pp, "rb") as openfile:
            anchor_pair_single = pickle.load(openfile)
        with open(
            f"{ModelType.save_dir}/balance_weight_pair_spatial.pkl", "rb"
        ) as openfile:
            anchor_pair_spatial = pickle.load(openfile)
        anchor_pair_single = {
            k: torch.as_tensor(v, dtype=torch.float32).to(device)
            for k, v in anchor_pair_single.items()
        }
        anchor_pair_spatial = {
            k: torch.as_tensor(v, dtype=torch.float32).to(device)
            for k, v in anchor_pair_spatial.items()
        }
        logging.info("[FuseMap anchor] pairwise balance-weight gating ENABLED")

    loss_atlas_val_best = float("inf")
    patience_counter = 0

    optimizer_dis = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.discriminator_single.parameters(),
            model.discriminator_spatial.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    optimizer_ae = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            model.encoder.parameters(),
            model.decoder.parameters(),
            model.scrna_seq_adj.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    scheduler_dis = ReduceLROnPlateau(
        optimizer_dis,
        mode="min",
        factor=ModelType.lr_factor_final.value,
        patience=ModelType.lr_patience_final.value,
        verbose=True,
    )
    scheduler_ae = ReduceLROnPlateau(
        optimizer_ae,
        mode="min",
        factor=ModelType.lr_factor_final.value,
        patience=ModelType.lr_patience_final.value,
        verbose=True,
    )

    anchor_state = AnchorState() if AnchorConfig.anchor_lambda > 0 else None

    for epoch in tqdm(range(ModelType.epochs_run_final + 1, ModelType.n_epochs.value)):
        loss_dis = 0
        loss_ae_dis = 0
        loss_all_item = 0
        if ModelType.use_llm_gene_embedding=='combine':
            loss_gene_embedding = 0
        loss_atlas_i = {}
        for i in range(ModelType.n_atlas):
            loss_atlas_i[i] = 0
        loss_atlas_val = 0
        anneal = (
            max(1 - (epoch - 1) / flagconfig.align_anneal, 0)
            if flagconfig.align_anneal
            else 0
        )

        if (
            anchor_state is not None
            and epoch >= AnchorConfig.anchor_start_epoch
            and (epoch - AnchorConfig.anchor_start_epoch)
            % AnchorConfig.anchor_refresh_epochs
            == 0
        ):
            refresh_anchors(
                model,
                spatial_dataloader,
                feature_all,
                adj_all,
                train_mask,
                device,
                anchor_state,
                epoch,
                "final",
            )

        model.train()

        for  blocks_all in spatial_dataloader:
            (
                batch_features_all,
                adj_all_block,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flag_source_cat_single,
                flag_source_cat_spatial,
                row_index_all,
                col_index_all,
            ) = get_data(blocks_all, feature_all, adj_all, train_mask, device, model)

            balance_weight_single_block = [
                balance_weight_single[i][row_index_all[i]]
                for i in range(ModelType.n_atlas)
            ]

            balance_weight_spatial_block = [
                balance_weight_spatial[i][col_index_all[i]]
                for i in range(ModelType.n_atlas)
            ]

            # Train discriminator part
            loss_part1 = compute_dis_loss(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                balance_weight_single_block,
                balance_weight_spatial_block,
                flagconfig,
            )
            model.zero_grad(set_to_none=True)
            loss_part1["dis"].backward()
            if dist_hooks is not None:
                dist_hooks.sync_grads(model)
            optimizer_dis.step()
            loss_dis += loss_part1["dis"].item()

            # Train AE part
            loss_part2 = compute_ae_loss(
                model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block,
                train_mask_batch_single,
                train_mask_batch_spatial,
                balance_weight_single_block,
                balance_weight_spatial_block,
                flagconfig,
                anchor_state=anchor_state,
                row_index_all=row_index_all,
                col_index_all=col_index_all,
                anchor_pair_single=anchor_pair_single,
                anchor_pair_spatial=anchor_pair_spatial,
            )
            model.zero_grad(set_to_none=True)
            loss_part2["loss_all"].backward()
            if dist_hooks is not None:
                dist_hooks.sync_grads(model)
            optimizer_ae.step()

            if ModelType.use_llm_gene_embedding=='combine':
                # for _ in range(3):
                    loss_part3 = compute_gene_embedding_loss(model)
                    model.zero_grad(set_to_none=True)
                    loss_part3.backward()
                    if dist_hooks is not None:
                        dist_hooks.sync_grads(model)
                    optimizer_ae.step()
                    loss_gene_embedding += loss_part3.item()

            for i in range(ModelType.n_atlas):
                loss_atlas_i[i] += loss_part2["loss_AE_all"][i].item()
            loss_all_item += loss_part2["loss_all"].item()
            loss_ae_dis += loss_part2["dis_ae"].item()

        if dist_hooks is not None:
            flagconfig.lambda_disc_single = dist_hooks.sync_value(flagconfig.lambda_disc_single)
        flagconfig.align_anneal /= 2

        save_snapshot(
            model, ModelType.epochs_run_pretrain, epoch, ModelType.snapshot_path,ModelType.verbose.value
        )

        if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
            logging.info(
                f"\n\nTrain Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
            Loss dis: {loss_dis / len(spatial_dataloader)},\
            Loss AE: {[i / len(spatial_dataloader) for i in loss_atlas_i.values()]} , \
            Loss AE sum: {sum([i / len(spatial_dataloader) for i in loss_atlas_i.values()])} , \
            Loss ae dis:{loss_ae_dis / len(spatial_dataloader)},\
            Loss all:{loss_all_item / len(spatial_dataloader)}\n"
            )
            if ModelType.use_llm_gene_embedding=='combine':
                logging.info(
                    f"Train Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
                Loss gene embedding: {loss_gene_embedding / len(spatial_dataloader)}\n"
                )

        ################# validation
        if epoch > ModelType.TRAIN_WITHOUT_EVAL.value:
            model.eval()
            with torch.no_grad():
                for ind, blocks_all in enumerate(spatial_dataloader):
                    # if ind not in random_numbers:
                    #     continue

                    (
                        batch_features_all,
                        adj_all_block,
                        adj_all_block_dis,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        row_index_all,
                        col_index_all,
                    ) = get_data(
                        blocks_all, feature_all, adj_all, val_mask, device, model
                    )

                    balance_weight_single_block = [
                        balance_weight_single[i][row_index_all[i]]
                        for i in range(ModelType.n_atlas)
                    ]

                    balance_weight_spatial_block = [
                        balance_weight_spatial[i][col_index_all[i]]
                        for i in range(ModelType.n_atlas)
                    ]

                    # val AE part
                    loss_part2 = compute_ae_loss(
                        model,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        anneal,
                        batch_features_all,
                        adj_all_block,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        balance_weight_single_block,
                        balance_weight_spatial_block,
                        flagconfig,
                    )

                    for i in range(ModelType.n_atlas):
                        loss_atlas_val += loss_part2["loss_AE_all"][i].item()

                loss_atlas_val = (
                    loss_atlas_val / len(spatial_dataloader) / ModelType.n_atlas
                )
                if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                    logging.info(
                        f"\n\nValidation Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
                    Loss AE validation: {loss_atlas_val} \n"
                    )

            if dist_hooks is not None:
                loss_atlas_val = dist_hooks.sync_value(loss_atlas_val)
            scheduler_dis.step(loss_atlas_val)
            scheduler_ae.step(loss_atlas_val)
            current_lr = optimizer_dis.param_groups[0]["lr"]
            if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                logging.info(f"\n\ncurrent lr:{current_lr}\n")

            # If the loss is lower than the best loss so far, save the model And reset the patience counter
            if loss_atlas_val < loss_atlas_val_best:
                loss_atlas_val_best = loss_atlas_val
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                )
            else:
                patience_counter += 1

            # If the patience counter is greater than or equal to the patience limit, stop training
            if patience_counter >= ModelType.patience_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                logging.info("\n\nEarly stopping due to loss not improving\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break
            if current_lr < ModelType.lr_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                logging.info("\n\nEarly stopping due to loss not improving - learning rate\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break

    if os.path.exists(f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt"):
        os.rename(
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model.pt",
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt",
        )
        logging.info("\n\nFile name changed in the end\n")


def read_model(
    model, spatial_dataloader_test, g_all, feature_all, adj_all, device, ModelType, mode
):
    model.load_state_dict(
        torch.load(f"{ModelType.save_dir}/trained_model/FuseMap_{mode}_model_final.pt")
    )
    
    with torch.no_grad():
        model.eval()

        learnt_scrna_seq_adj = {}
        for i in range(ModelType.n_atlas):
            if ModelType.input_identity[i] == "scrna":
                learnt_scrna_seq_adj["atlas" + str(i)] = (
                    model.scrna_seq_adj["atlas" + str(i)]().detach().cpu().numpy()
                )

        for blocks_all in tqdm(spatial_dataloader_test):
            if _is_prepared_blocks(blocks_all):
                row_index_all = {
                    i: blocks_all[i]["row_index"] for i in range(ModelType.n_atlas)
                }
                col_index_all = {
                    i: blocks_all[i]["col_index"] for i in range(ModelType.n_atlas)
                }
                batch_features_all = [
                    blocks_all[i]["feature"].to(device, non_blocking=True)
                    for i in range(ModelType.n_atlas)
                ]
                adj_all_block_dis = [
                    blocks_all[i]["adj_block"].to(device, non_blocking=True)
                    if ModelType.input_identity[i] == "ST"
                    else model.scrna_seq_adj["atlas" + str(i)]()[
                        blocks_all[i]["row_index"], :
                    ][:, blocks_all[i]["col_index"]].detach()
                    for i in range(ModelType.n_atlas)
                ]
            else:
                row_index_all = {}
                col_index_all = {}
                for i_atlas in range(ModelType.n_atlas):
                    row_index = list(blocks_all[i_atlas]["spatial"][0])
                    col_index = list(blocks_all[i_atlas]["spatial"][1])
                    row_index_all[i_atlas] = torch.sort(
                        torch.vstack(row_index).flatten()
                    )[0].tolist()
                    col_index_all[i_atlas] = torch.sort(
                        torch.vstack(col_index).flatten()
                    )[0].tolist()

                batch_features_all = [
                    torch.FloatTensor(
                        feature_all[i][row_index_all[i], :].toarray()
                    ).to(device)
                    for i in range(ModelType.n_atlas)
                ]
                adj_all_block_dis = [
                    torch.FloatTensor(
                        adj_all[i][row_index_all[i], :]
                        .tocsc()[:, col_index_all[i]]
                        .todense()
                    ).to(device)
                    if ModelType.input_identity[i] == "ST"
                    else model.scrna_seq_adj["atlas" + str(i)]()[row_index_all[i], :][
                        :, col_index_all[i]
                    ].detach()
                    for i in range(ModelType.n_atlas)
                ]

            z_all = [
                model.encoder["atlas" + str(i)](
                    batch_features_all[i], adj_all_block_dis[i]
                )
                for i in range(ModelType.n_atlas)
            ]
            z_distribution_all = [
                z_all[i][3] for i in range(ModelType.n_atlas)
            ]

            z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]

            for i in range(ModelType.n_atlas):
                g_all[i].nodes[row_index_all[i]].data["single_feat_hidden"] = (
                    z_distribution_all[i].detach().cpu()
                )
                g_all[i].nodes[col_index_all[i]].data["spatial_feat_hidden"] = (
                    z_spatial_all[i].detach().cpu()
                )

    latent_embeddings_all_single = [
        g_all[i].ndata["single_feat_hidden"].numpy() for i in range(ModelType.n_atlas)
    ]
    latent_embeddings_all_spatial = [
        g_all[i].ndata["spatial_feat_hidden"].numpy() for i in range(ModelType.n_atlas)
    ]

    save_obj(
        latent_embeddings_all_single,
        f"{ModelType.save_dir}/latent_embeddings_all_single_{mode}",
    )
    save_obj(
        latent_embeddings_all_spatial,
        f"{ModelType.save_dir}/latent_embeddings_all_spatial_{mode}",
    )


def balance_weight(model, adatas, save_dir, n_atlas, device):
    with open(
        f"{save_dir}/latent_embeddings_all_single_pretrain.pkl", "rb"
    ) as openfile:
        latent_embeddings_all_single = pickle.load(openfile)
    with open(
        f"{save_dir}/latent_embeddings_all_spatial_pretrain.pkl", "rb"
    ) as openfile:
        latent_embeddings_all_spatial = pickle.load(openfile)

    adatas_ = [
        ad.AnnData(
            obs=adatas[i].obs.copy(deep=False).assign(n=1),
            obsm={
                "single": latent_embeddings_all_single[i],
                "spatial": latent_embeddings_all_spatial[i],
            },
        )
        for i in range(n_atlas)
    ]

    if not os.path.exists(f"{save_dir}/ad_fusemap_single_leiden.pkl"):
        leiden_adata_single = []
        leiden_adata_spatial = []
        ad_fusemap_single_leiden = []
        ad_fusemap_spatial_leiden = []
        for adata_ in adatas_:
            sc.pp.neighbors(
                adata_,
                n_pcs=adata_.obsm["single"].shape[1],
                use_rep="single",
                metric="cosine",
            )
            sc.tl.leiden(
                adata_,
                resolution=float(os.environ.get("FUSEMAP_BALANCE_RES", "4")),
                key_added="fusemap_single_leiden",
            )
            ad_fusemap_single_leiden.append(list(adata_.obs["fusemap_single_leiden"]))
            leiden_adata_single.append(
                average_embeddings(adata_, "fusemap_single_leiden", "single")
            )

            sc.pp.neighbors(
                adata_,
                n_pcs=adata_.obsm["spatial"].shape[1],
                use_rep="spatial",
                metric="cosine",
            )
            sc.tl.leiden(
                adata_,
                resolution=float(os.environ.get("FUSEMAP_BALANCE_RES", "4")),
                key_added="fusemap_spatial_leiden",
            )
            ad_fusemap_spatial_leiden.append(list(adata_.obs["fusemap_spatial_leiden"]))
            leiden_adata_spatial.append(
                average_embeddings(adata_, "fusemap_spatial_leiden", "spatial")
            )

        save_obj(ad_fusemap_single_leiden, f"{save_dir}/ad_fusemap_single_leiden")
        save_obj(ad_fusemap_spatial_leiden, f"{save_dir}/ad_fusemap_spatial_leiden")
        save_obj(leiden_adata_single, f"{save_dir}/leiden_adata_single")
        save_obj(leiden_adata_spatial, f"{save_dir}/leiden_adata_spatial")

    else:
        with open(f"{save_dir}/ad_fusemap_single_leiden.pkl", "rb") as openfile:
            ad_fusemap_single_leiden = pickle.load(openfile)
        with open(f"{save_dir}/ad_fusemap_spatial_leiden.pkl", "rb") as openfile:
            ad_fusemap_spatial_leiden = pickle.load(openfile)
        try:
            with open(f"{save_dir}/leiden_adata_single.pkl", "rb") as openfile:
                leiden_adata_single = pickle.load(openfile)
            with open(f"{save_dir}/leiden_adata_spatial.pkl", "rb") as openfile:
                leiden_adata_spatial = pickle.load(openfile)
        except:
            ### need to convert
            leiden_adata_single = []
            for i in range(len(ad_fusemap_single_leiden)):
                leiden_adata_single.append(
                    sc.read_h5ad(
                        f"{save_dir}/pickle_convert/PRETRAINED_leiden_adata_single_{i}.h5ad"
                    )
                )
            leiden_adata_spatial = []
            for i in range(len(ad_fusemap_single_leiden)):
                leiden_adata_spatial.append(
                    sc.read_h5ad(
                        f"{save_dir}/pickle_convert/PRETRAINED_leiden_adata_spatial_{i}.h5ad"
                    )
                )

        for ind, adata_ in enumerate(adatas_):
            adata_.obs["fusemap_single_leiden"] = ad_fusemap_single_leiden[ind]
            adata_.obs["fusemap_spatial_leiden"] = ad_fusemap_spatial_leiden[ind]

    # joint K^n sparse product explodes at >=5-6 atlases with fine leiden
    # (measured: 6 atlases x res=4 -> 6.8e9 nnz, 304 GiB OOM); GLUE's own
    # pairwise-aggregated path scales linearly in pairs
    if len(leiden_adata_single) >= 5:
        # raise ValueError('balance weight')
        balance_weight_single = get_balance_weight_subsample(
            leiden_adata_single, adatas_, "fusemap_single_leiden"
        )
        balance_weight_spatial = get_balance_weight_subsample(
            leiden_adata_spatial, adatas_, "fusemap_spatial_leiden"
        )
    else:
        # joint (all-pairs product) weights gate the DISCRIMINATOR: keep the
        # original GLUE matching (0.5/4) — the strict env knobs apply only to
        # the pairwise anchor weights below (joint product collapses to zero
        # on heterogeneous collections when matching is strict).
        balance_weight_single = get_balance_weight(
            adatas, leiden_adata_single, adatas_, "fusemap_single_leiden",
            cutoff=0.5, power=4,
        )
        balance_weight_spatial = get_balance_weight(
            adatas, leiden_adata_spatial, adatas_, "fusemap_spatial_leiden",
            cutoff=0.5, power=4,
        )

    balance_weight_single = [torch.tensor(i).to(device) for i in balance_weight_single]
    balance_weight_spatial = [
        torch.tensor(i).to(device) for i in balance_weight_spatial
    ]

    save_obj(balance_weight_single, f"{save_dir}/balance_weight_single")
    save_obj(balance_weight_spatial, f"{save_dir}/balance_weight_spatial")

    ### pairwise weights for ANCHOR gating (strictness from env, default 0.75/8).
    ### key (a,b) -> weights of atlas a's cells for its pull toward atlas b.
    pair_single, pair_spatial = {}, {}
    for a in range(n_atlas):
        for b in range(a + 1, n_atlas):
            for pair_dict, leiden_list, key in (
                (pair_single, leiden_adata_single, "fusemap_single_leiden"),
                (pair_spatial, leiden_adata_spatial, "fusemap_spatial_leiden"),
            ):
                try:
                    ws = get_balance_weight(
                        [adatas[a], adatas[b]],
                        [leiden_list[a], leiden_list[b]],
                        [adatas_[a], adatas_[b]],
                        key,
                    )
                except ValueError:
                    ws = [
                        np.zeros(adatas_[a].n_obs),
                        np.zeros(adatas_[b].n_obs),
                    ]
                pair_dict[(a, b)] = ws[0]
                pair_dict[(b, a)] = ws[1]
    save_obj(pair_single, f"{save_dir}/balance_weight_pair_single")
    save_obj(pair_spatial, f"{save_dir}/balance_weight_pair_spatial")


def load_ref_model(
    ref_dir,
    device,
):
    PRETRAINED_MODEL_PATH = ref_dir + f"/trained_model/FuseMap_final_model_final.pt"

    if os.path.exists(PRETRAINED_MODEL_PATH):
        TRAINED_MODEL = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        TRAINED_X_NUM = sum(["decoder" in i for i in TRAINED_MODEL.keys()])

        TRAINED_GENE_EMBED = sc.read_h5ad(ref_dir + "/ad_gene_embedding.h5ad")
        TRAINED_GENE_NAME = list(TRAINED_GENE_EMBED.obs.index)
    else:
        raise ValueError("No pretrained model found!")
    return TRAINED_MODEL, TRAINED_X_NUM, TRAINED_GENE_EMBED, TRAINED_GENE_NAME


def add_pretrain_to_name(s):
    if "discriminator_single" in s:
        return s.replace("discriminator_single", "discriminator_single_pretrain")
    elif "discriminator_spatial" in s:
        return s.replace("discriminator_spatial", "discriminator_spatial_pretrain")
    else:
        return s


def transfer_weight(TRAINED_MODEL, pretrain_index, adapt_model):
    layers_to_transfer = [
        "discriminator_single.linear_0.weight",
        "discriminator_single.linear_0.bias",
        "discriminator_single.linear_1.weight",
        "discriminator_single.linear_1.bias",
        "discriminator_spatial.linear_0.weight",
        "discriminator_spatial.linear_0.bias",
        "discriminator_spatial.linear_1.weight",
        "discriminator_spatial.linear_1.bias",
    ]
    transferred_dict = {
        k: v for k, v in TRAINED_MODEL.items() if k in layers_to_transfer
    }
    transferred_dict_pretrain = {
        add_pretrain_to_name(k): v
        for k, v in TRAINED_MODEL.items()
        if k in layers_to_transfer
    }
    transferred_dict.update(transferred_dict_pretrain)

    new_model_dict = adapt_model.state_dict()
    new_model_dict.update(transferred_dict)
    adapt_model.load_state_dict(new_model_dict)

    with torch.no_grad():
        # Assuming the pretrained parameters go into the first 'n' units
        adapt_model.discriminator_single_pretrain.pred.weight = nn.Parameter(
            TRAINED_MODEL["discriminator_single.pred.weight"]
        )
        adapt_model.discriminator_single_pretrain.pred.bias = nn.Parameter(
            TRAINED_MODEL["discriminator_single.pred.bias"]
        )
        adapt_model.discriminator_spatial_pretrain.pred.weight = nn.Parameter(
            TRAINED_MODEL["discriminator_spatial.pred.weight"]
        )
        adapt_model.discriminator_spatial_pretrain.pred.bias = nn.Parameter(
            TRAINED_MODEL["discriminator_spatial.pred.bias"]
        )
        adapt_model.gene_embedding_pretrained = nn.Parameter(
            TRAINED_MODEL["gene_embedding"][:, pretrain_index]
        )

    for param in adapt_model.discriminator_single_pretrain.parameters():
        param.requires_grad = False
    for param in adapt_model.discriminator_spatial_pretrain.parameters():
        param.requires_grad = False
    adapt_model.gene_embedding_pretrained.requires_grad = False

    adapt_model.discriminator_single.pred.weight.requires_grad = True
    adapt_model.discriminator_single.pred.bias.requires_grad = True
    adapt_model.discriminator_spatial.pred.weight.requires_grad = True
    adapt_model.discriminator_spatial.pred.bias.requires_grad = True

    # Print out to verify
    # for name, param in adapt_model.named_parameters():
    #     print(name, param.requires_grad)


def load_ref_data(ref_dir, TRAINED_X_NUM, batch_size, USE_REFERENCE_PCT=0.1):
    with open(ref_dir + f"/latent_embeddings_all_single_final.pkl", "rb") as openfile:
        latent_embeddings_single = pickle.load(openfile)
    with open(ref_dir + f"/latent_embeddings_all_spatial_final.pkl", "rb") as openfile:
        latent_embeddings_spatial = pickle.load(openfile)

    ds_pretrain_single = [
        MapPretrainDataset(latent_embeddings_single[i]) for i in range(TRAINED_X_NUM)
    ]
    dataloader_pretrain_single = MapPretrainDataLoader(
        ds_pretrain_single,
        int(batch_size * USE_REFERENCE_PCT * 4),
        shuffle=True,
        n_atlas=TRAINED_X_NUM,
    )

    ds_pretrain_spatial = [
        MapPretrainDataset(latent_embeddings_spatial[i]) for i in range(TRAINED_X_NUM)
    ]
    dataloader_pretrain_spatial = MapPretrainDataLoader(
        ds_pretrain_spatial,
        int(batch_size * USE_REFERENCE_PCT),
        shuffle=True,
        n_atlas=TRAINED_X_NUM,
    )

    return dataloader_pretrain_single, dataloader_pretrain_spatial


def map_model(
    adapt_model,
    spatial_dataloader,
    feature_all,
    adj_all,
    device,
    train_mask,
    val_mask,
    ref_dir,
    dataloader_pretrain_single,
    dataloader_pretrain_spatial,
    TRAINED_X_NUM,
    flagconfig,
):
    loss_atlas_val_best = float("inf")
    patience_counter = 0

    optimizer_dis = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            adapt_model.discriminator_single.parameters(),
            adapt_model.discriminator_spatial.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    optimizer_ae = getattr(torch.optim, ModelType.optim_kw.value)(
        itertools.chain(
            adapt_model.encoder.parameters(),
            adapt_model.decoder.parameters(),
            adapt_model.scrna_seq_adj.parameters(),
        ),
        lr=ModelType.learning_rate.value,
    )
    scheduler_dis = ReduceLROnPlateau(
        optimizer_dis,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )
    scheduler_ae = ReduceLROnPlateau(
        optimizer_ae,
        mode="min",
        factor=ModelType.lr_factor_pretrain.value,
        patience=ModelType.lr_patience_pretrain.value,
        verbose=True,
    )

    dataloader_pretrain_single_cycle = itertools.cycle(dataloader_pretrain_single)
    dataloader_pretrain_spatial_cycle = itertools.cycle(dataloader_pretrain_spatial)

    for epoch in tqdm(
        range(ModelType.epochs_run_pretrain + 1, ModelType.n_epochs.value)
    ):
        loss_dis = 0
        loss_ae_dis = 0
        loss_all_item = 0
        loss_atlas_i = {}
        for i in range(ModelType.n_atlas):
            loss_atlas_i[i] = 0
        loss_atlas_val = 0
        anneal = (
            max(1 - (epoch - 1) / flagconfig.align_anneal, 0)
            if flagconfig.align_anneal
            else 0
        )

        adapt_model.train()

        for blocks_all in spatial_dataloader:
            (
                batch_features_all,
                adj_all_block,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                flag_source_cat_single,
                flag_source_cat_spatial,
                _,
                _,
            ) = get_data(
                blocks_all, feature_all, adj_all, train_mask, device, adapt_model
            )

            ### difference: add pretrain data
            pretrain_single_batch = next(dataloader_pretrain_single_cycle)
            pretrain_single_batch = [
                pretrain_single_batch[i].to(device) for i in range(TRAINED_X_NUM)
            ]
            pretrain_spatial_batch = next(dataloader_pretrain_spatial_cycle)
            pretrain_spatial_batch = [
                pretrain_spatial_batch[i].to(device) for i in range(TRAINED_X_NUM)
            ]

            ### add difference: add discriminator pretrain
            flag_shape_single_pretrain = [
                pretrain_single_batch[i].shape[0] for i in range(TRAINED_X_NUM)
            ]
            flag_all_single_pretrain = torch.cat(
                [
                    torch.full((x,), i + ModelType.n_atlas)
                    for i, x in enumerate(flag_shape_single_pretrain)
                ]
            )
            flag_source_cat_single_pretrain = flag_all_single_pretrain.long().to(device)

            flag_shape_spatial_pretrain = [
                pretrain_spatial_batch[i].shape[0] for i in range(TRAINED_X_NUM)
            ]
            flag_all_spatial_pretrain = torch.cat(
                [
                    torch.full((x,), i + ModelType.n_atlas)
                    for i, x in enumerate(flag_shape_spatial_pretrain)
                ]
            )
            flag_source_cat_spatial_pretrain = flag_all_spatial_pretrain.long().to(
                device
            )

            # Train discriminator part
            loss_part1 = compute_dis_loss_map(
                adapt_model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block_dis,
                train_mask_batch_single,
                train_mask_batch_spatial,
                pretrain_single_batch,
                pretrain_spatial_batch,
                flag_source_cat_single_pretrain,
                flag_source_cat_spatial_pretrain,
                flagconfig,
            )
            adapt_model.zero_grad(set_to_none=True)
            loss_part1["dis"].backward()
            optimizer_dis.step()
            loss_dis += loss_part1["dis"].item()

            # Train AE part
            loss_part2 = compute_ae_loss_map(
                adapt_model,
                flag_source_cat_single,
                flag_source_cat_spatial,
                anneal,
                batch_features_all,
                adj_all_block,
                train_mask_batch_single,
                train_mask_batch_spatial,
                pretrain_single_batch,
                pretrain_spatial_batch,
                flag_source_cat_single_pretrain,
                flag_source_cat_spatial_pretrain,
                flagconfig,
            )
            adapt_model.zero_grad(set_to_none=True)
            loss_part2["loss_all"].backward()
            optimizer_ae.step()

            if ModelType.use_llm_gene_embedding=='combine':
                loss_part3 = compute_gene_embedding_new_loss(adapt_model)
                adapt_model.zero_grad(set_to_none=True)
                loss_part3.backward()
                optimizer_ae.step()

            for i in range(ModelType.n_atlas):
                loss_atlas_i[i] += loss_part2["loss_AE_all"][i].item()
            loss_all_item += loss_part2["loss_all"].item()
            loss_ae_dis += loss_part2["dis_ae"].item()

        flagconfig.align_anneal /= 2

        if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
            logging.info(
                f"\n\nTrain Epoch {epoch}/{ModelType.n_epochs}, \
            Loss dis: {loss_dis / len(spatial_dataloader)},\
            Loss AE: {[i / len(spatial_dataloader) for i in loss_atlas_i.values()]} , \
            Loss ae dis:{loss_ae_dis / len(spatial_dataloader)},\
            Loss all:{loss_all_item / len(spatial_dataloader)}\n"
            )

        save_snapshot(
            adapt_model, epoch, ModelType.epochs_run_final, ModelType.snapshot_path, ModelType.verbose.value
        )

        if not os.path.exists(f"{ModelType.save_dir}/lambda_disc_single.pkl"):
            save_obj(
                flagconfig.lambda_disc_single,
                f"{ModelType.save_dir}/lambda_disc_single",
            )

        ################# validation
        if epoch > ModelType.TRAIN_WITHOUT_EVAL.value:
            adapt_model.eval()
            with torch.no_grad():
                for blocks_all in spatial_dataloader:
                    (
                        batch_features_all,
                        adj_all_block,
                        adj_all_block_dis,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        _,
                        _,
                    ) = get_data(
                        blocks_all, feature_all, adj_all, val_mask, device, adapt_model
                    )

                    ### difference: add pretrain data
                    pretrain_single_batch = next(dataloader_pretrain_single_cycle)
                    pretrain_single_batch = [
                        pretrain_single_batch[i].to(device)
                        for i in range(TRAINED_X_NUM)
                    ]
                    pretrain_spatial_batch = next(dataloader_pretrain_spatial_cycle)
                    pretrain_spatial_batch = [
                        pretrain_spatial_batch[i].to(device)
                        for i in range(TRAINED_X_NUM)
                    ]

                    ### difference: discriminator pretrain
                    flag_shape_single_pretrain = [
                        pretrain_single_batch[i].shape[0] for i in range(TRAINED_X_NUM)
                    ]
                    flag_all_single_pretrain = torch.cat(
                        [
                            torch.full((x,), i + ModelType.n_atlas)
                            for i, x in enumerate(flag_shape_single_pretrain)
                        ]
                    )
                    flag_source_cat_single_pretrain = (
                        flag_all_single_pretrain.long().to(device)
                    )

                    flag_shape_spatial_pretrain = [
                        pretrain_spatial_batch[i].shape[0] for i in range(TRAINED_X_NUM)
                    ]
                    flag_all_spatial_pretrain = torch.cat(
                        [
                            torch.full((x,), i + ModelType.n_atlas)
                            for i, x in enumerate(flag_shape_spatial_pretrain)
                        ]
                    )
                    flag_source_cat_spatial_pretrain = (
                        flag_all_spatial_pretrain.long().to(device)
                    )

                    # val AE part
                    loss_part2 = compute_ae_loss_map(
                        adapt_model,
                        flag_source_cat_single,
                        flag_source_cat_spatial,
                        anneal,
                        batch_features_all,
                        adj_all_block,
                        val_mask_batch_single,
                        val_mask_batch_spatial,
                        pretrain_single_batch,
                        pretrain_spatial_batch,
                        flag_source_cat_single_pretrain,
                        flag_source_cat_spatial_pretrain,
                        flagconfig,
                    )

                    for i in range(ModelType.n_atlas):
                        loss_atlas_val += loss_part2["loss_AE_all"][i].item()

                loss_atlas_val = loss_atlas_val / len(spatial_dataloader) / ModelType.n_atlas
                if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                    logging.info(
                        f"\n\nValidation Epoch {epoch + 1}/{ModelType.n_epochs.value}, \
                    Loss AE validation: {loss_atlas_val} \n"
                    )

            if dist_hooks is not None:
                loss_atlas_val = dist_hooks.sync_value(loss_atlas_val)
            scheduler_dis.step(loss_atlas_val)
            scheduler_ae.step(loss_atlas_val)
            current_lr = optimizer_dis.param_groups[0]["lr"]
            if ModelType.verbose.value == True or os.environ.get("FUSEMAP_VERBOSE") == "1":
                logging.info(f"\n\ncurrent lr:{current_lr}\n")

            # If the loss is lower than the best loss so far, save the model And reset the patience counter
            if loss_atlas_val < loss_atlas_val_best:
                loss_atlas_val_best = loss_atlas_val
                patience_counter = 0
                torch.save(
                    adapt_model.state_dict(),
                    f"{ModelType.save_dir}/trained_model/FuseMap_map_model.pt",
                )

            else:
                patience_counter += 1


            # If the patience counter is greater than or equal to the patience limit, stop training
            if patience_counter >= ModelType.patience_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                logging.info("\n\nEarly stopping due to loss not improving\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_map_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_map_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break
            if current_lr < ModelType.lr_limit_final.value:
                # torch.save(model.state_dict(), f"{save_dir}/trained_model/FuseMap_final_model_end.pt")
                logging.info("\n\nEarly stopping due to loss not improving - learning rate\n")
                os.rename(
                    f"{ModelType.save_dir}/trained_model/FuseMap_map_model.pt",
                    f"{ModelType.save_dir}/trained_model/FuseMap_map_model_final.pt",
                )
                logging.info("\n\nFile name changed\n")
                break

    if os.path.exists(f"{ModelType.save_dir}/trained_model/FuseMap_map_model.pt"):
        os.rename(
            f"{ModelType.save_dir}/trained_model/FuseMap_map_model.pt",
            f"{ModelType.save_dir}/trained_model/FuseMap_map_model_final.pt",
        )
        logging.info("\n\nFile name changed in the end\n")