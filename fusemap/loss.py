import logging
import torch.nn.functional as F
import sklearn
import numpy as np
import torch
import torch.distributions as D
import pandas as pd
from sparse import COO
from fusemap.config import *
import torch.nn as nn
from sklearn import preprocessing

def AE_Gene_loss(recon_x, x, z_distribution):
    """
    Compute the generator loss for the autoencoder.
    
    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed tensor.
    x : torch.Tensor
        The original tensor.
    z_distribution : torch.distributions
        The distribution of the latent variables.
    Returns
    -------
    torch.Tensor
        The gene loss.
    
    Examples
    --------
    >>> import torch
    >>> import torch.distributions as D
    >>> recon_x = torch.randn(10, 10)
    >>> x = torch.randn(10, 10)
    >>> z_distribution = D.Normal(0.0, 1.0)
    >>> AE_Gene_loss(recon_x, x, z_distribution)
    tensor(0.0)
    
    """

    if recon_x.shape[0] == 0:
        return torch.tensor(0.0, dtype=torch.float32).to(recon_x.device)

    reconstruction_loss = F.mse_loss(recon_x, x)
    kl_divergence = (
        D.kl_divergence(z_distribution, D.Normal(0.0, 1.0)).sum(dim=1).mean()
        / x.shape[1]
    )
    return reconstruction_loss + kl_divergence


def prod(x):
    """
    Compute the product of a list of numbers.
    
    Parameters
    ----------
    x : list
        The list of numbers.
    Returns
    -------
    int
        The product of the numbers.
        
    Examples
    --------
    >>> x = [1, 2, 3, 4]
    >>> prod(x)
    24
    
    """
    ########### function from GLUE: https://github.com/gao-lab/GLUE
    # try:
    # from math import prod  # pylint: disable=redefined-outer-name
    #     return np.prod(x)
    # except ImportError:
    ans = 1
    for item in x:
        ans = ans * item
    return ans


"""
pretrain loss
"""


def compute_gene_embedding_loss(
        model
        ):
    """
    Compute the gene embedding loss.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model.
    Returns
    -------
    torch.Tensor
        The gene embedding loss.
    
    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> compute_gene_embedding_loss(model)
    tensor(0.0)

    """
    # Calculate gene embedding loss
    learned_matrix = model.gene_embedding.T

    learned_matrix=learned_matrix[model.llm_ind,:]

    learned_matrix_normalized = learned_matrix / learned_matrix.norm(dim=1, keepdim=True)
    predicted_matrix = torch.matmul(learned_matrix_normalized, learned_matrix_normalized.T)

    loss_fn = nn.MSELoss()
    loss_part3 = loss_fn(predicted_matrix, model.ground_truth_rel_matrix)
    return loss_part3


def compute_gene_embedding_new_loss(
        model
        ):
    """
    Compute the gene embedding loss.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model.
    Returns
    -------
    torch.Tensor
        The gene embedding loss.
    
    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> compute_gene_embedding_loss(model)
    tensor(0.0)

    """
    # Calculate gene embedding loss
    learned_matrix = torch.hstack([model.gene_embedding_new,model.gene_embedding_pretrained]).T

    learned_matrix=learned_matrix[model.llm_ind,:]

    learned_matrix_normalized = learned_matrix / learned_matrix.norm(dim=1, keepdim=True)
    predicted_matrix = torch.matmul(learned_matrix_normalized, learned_matrix_normalized.T)

    loss_fn = nn.MSELoss()
    loss_part3 = loss_fn(predicted_matrix, model.ground_truth_rel_matrix)
    return loss_part3
                

def compute_anchor_loss(
    z_single_list,
    z_spatial_list,
    row_index_all,
    col_index_all,
    anchor_state,
    balance_weight_single=None,
    balance_weight_spatial=None,
    anchor_pair_single=None,
    anchor_pair_spatial=None,
):
    """Cross-sample MNN anchor alignment term (STAligner-style).

    For every cell in the current minibatch that participates in a stored MNN
    anchor pair, pull its latent toward the CACHED (detached) partner latent
    from the other atlas. The partner side is detached, so this is a one-sided
    pull per batch; it becomes symmetric across batches over time. Applied to
    BOTH the single latent (z_mean) and the spatial latent (z_spatial) using the
    same anchor pairs. Normalized by the number of anchored contributions.

    Returns ``None`` when disabled or when no anchors are present in the batch,
    so that ``AnchorConfig.anchor_lambda == 0`` reproduces the original loss
    exactly (the term is never added).

    Parameters
    ----------
    z_single_list : list of torch.Tensor
        Per-atlas batch single latent (z_mean); rows aligned to row_index_all.
    z_spatial_list : list of torch.Tensor
        Per-atlas batch spatial latent (z_spatial); rows aligned to col_index_all.
    row_index_all : dict or None
        atlas -> global cell indices of the single-latent rows in this batch.
    col_index_all : dict or None
        atlas -> global cell indices of the spatial-latent rows in this batch.
    anchor_state : AnchorState or None
        Runtime anchor state (caches + partner maps).

    Returns
    -------
    torch.Tensor or None
        Scalar anchor loss, or None when there is nothing to add.
    """
    if (
        anchor_state is None
        or AnchorConfig.anchor_lambda == 0
        or not getattr(anchor_state, "has_anchors", False)
        or anchor_state.partner is None
        or row_index_all is None
        or col_index_all is None
    ):
        return None

    device = z_single_list[0].device
    total = None
    count = 0
    for i in range(ModelType.n_atlas):
        row_idx = torch.as_tensor(row_index_all[i], device=device, dtype=torch.long)
        col_idx = torch.as_tensor(col_index_all[i], device=device, dtype=torch.long)
        for j in range(ModelType.n_atlas):
            if i == j:
                continue
            # asymmetric mode: never pull anyone TOWARD a query (e.g. bead)
            # dataset's latents; query datasets still get pulled toward others.
            if j in AnchorConfig.anchor_query_atlases:
                continue
            partner = anchor_state.partner.get((i, j))
            if partner is None:
                continue
            # ---- single latent (z_mean) ----
            p_row = partner[row_idx]
            m_row = p_row >= 0
            if bool(m_row.any()):
                cur = z_single_list[i][m_row]
                tgt = anchor_state.single_cache[j][p_row[m_row]].detach()
                d2 = ((cur - tgt) ** 2).sum(dim=1)
                if anchor_pair_single is not None and (i, j) in anchor_pair_single:
                    # pairwise gating: this pull is weighted by cell i's
                    # correspondence in THIS pair only (no all-pairs product,
                    # so one noisy sample cannot veto alignment in other pairs)
                    w = anchor_pair_single[(i, j)][row_idx[m_row]].detach().flatten()
                    total = (d2 * w).sum() if total is None else total + (d2 * w).sum()
                    count += float(w.sum())
                elif balance_weight_single is not None:
                    # same per-cell protection as the discriminator loss:
                    # cells from clusters without cross-sample counterparts
                    # carry low balance weight -> anchor pull fades for them
                    w = balance_weight_single[i][m_row].detach().flatten()
                    total = (d2 * w).sum() if total is None else total + (d2 * w).sum()
                    count += float(w.sum())
                else:
                    total = d2.sum() if total is None else total + d2.sum()
                    count += d2.numel()
            # ---- spatial latent (z_spatial); tissue level is worst-separated ----
            p_col = partner[col_idx]
            m_col = p_col >= 0
            if bool(m_col.any()):
                cur_s = z_spatial_list[i][m_col]
                tgt_s = anchor_state.spatial_cache[j][p_col[m_col]].detach()
                d2s = ((cur_s - tgt_s) ** 2).sum(dim=1)
                if anchor_pair_spatial is not None and (i, j) in anchor_pair_spatial:
                    w_s = anchor_pair_spatial[(i, j)][col_idx[m_col]].detach().flatten()
                    total = (d2s * w_s).sum() if total is None else total + (d2s * w_s).sum()
                    count += float(w_s.sum())
                elif balance_weight_spatial is not None:
                    w_s = balance_weight_spatial[i][m_col].detach().flatten()
                    total = (d2s * w_s).sum() if total is None else total + (d2s * w_s).sum()
                    count += float(w_s.sum())
                else:
                    total = d2s.sum() if total is None else total + d2s.sum()
                    count += d2s.numel()

    if total is None or count == 0:
        return None
    return AnchorConfig.anchor_lambda * total / count


def compute_struct_loss(z_single_list, row_index_all, anchor_state):
    """Within-dataset structure-preservation triplet loss (opt-in).

    The cross-dataset alignment forces (adversarial discriminator + MNN anchor
    pull) only PULL cells together; nothing preserves each dataset's own
    neighborhood structure, so cell-type separation gets compressed. For each
    cell in the current minibatch this term samples ONE positive (a uniformly
    random expression-space kNN neighbor from the SAME dataset, precomputed in
    ``AnchorConfig.struct_knn``) and ONE negative from the same dataset,
    fetches both latents from the DETACHED full-dataset cache
    (``anchor_state.single_cache``, refreshed every few epochs by
    ``refresh_anchors``), and penalizes

        relu(||z - z_pos||^2 - ||z - z_neg||^2 + struct_margin)

    averaged over batch cells, then averaged over atlases with valid terms and
    scaled by ``AnchorConfig.struct_lambda``.

    Negative selection (``AnchorConfig.struct_hardneg``):

    - True (default): semi-hard mining. Draw ``struct_neg_candidates`` random
      candidates per cell, invalidate candidates equal to the cell itself or
      to one of its expression-kNN members, and keep the remaining candidate
      CLOSEST to the cell's current latent (selection distances computed on
      DETACHED tensors so no graph ops are created; only the final triplet
      uses the attached ``z``). If every candidate is invalid (tiny datasets)
      the first candidate is used, so the loss never sees inf/nan.
    - False: legacy path, ONE uniformly random negative per cell; reproduces
      prior runs bit-for-bit (same number and order of RNG draws).

    Parameters
    ----------
    z_single_list : list of torch.Tensor
        Per-atlas batch single latent (z_mean); rows aligned to row_index_all.
    row_index_all : dict or None
        atlas -> global cell indices of the single-latent rows in this batch.
    anchor_state : AnchorState or None
        Runtime anchor state; ``single_cache`` supplies pos/neg latents.

    Returns
    -------
    torch.Tensor or None
        Scalar struct loss, or None when there is nothing to add.

    CRITICAL: when ``AnchorConfig.struct_lambda == 0`` this returns None
    BEFORE any torch.randint call, so RNG state is untouched and existing
    runs reproduce exactly.
    """
    if AnchorConfig.struct_lambda == 0:
        return None
    struct_knn = getattr(AnchorConfig, "struct_knn", None)
    if (
        struct_knn is None
        or anchor_state is None
        or getattr(anchor_state, "single_cache", None) is None
        or row_index_all is None
    ):
        return None

    device = z_single_list[0].device
    total = None
    count = 0
    for i in range(ModelType.n_atlas):
        knn_i = struct_knn.get(i)
        cache_i = (
            anchor_state.single_cache[i]
            if i < len(anchor_state.single_cache)
            else None
        )
        if knn_i is None or cache_i is None:
            continue
        n_obs_i = cache_i.shape[0]
        if knn_i.shape[0] != n_obs_i:
            continue
        row_idx = torch.as_tensor(row_index_all[i], dtype=torch.long).cpu()
        n_batch = int(row_idx.numel())
        if n_batch == 0:
            continue
        # gather each batch cell's kNN row on CPU (struct_knn stays a small
        # int64 numpy array; torch.as_tensor shares its memory), then move
        # only the [n_batch, k] slice to the device
        knn_rows = torch.as_tensor(knn_i, dtype=torch.long)[row_idx].to(device)
        k_i = knn_rows.shape[1]
        # ONE positive per cell: uniformly random column of its kNN row
        pos_col = torch.randint(0, k_i, (n_batch,), device=device)
        pos_idx = knn_rows.gather(1, pos_col.unsqueeze(1)).squeeze(1)
        z = z_single_list[i]
        z_pos = cache_i[pos_idx].detach()
        if AnchorConfig.struct_hardneg:
            # Semi-hard negative mining: uniformly random negatives are
            # already far apart in the compact latent, so the repulsion term
            # is inert. Draw M candidates and keep the VALID one closest to
            # the cell's CURRENT latent (hardest valid negative).
            m_cand = AnchorConfig.struct_neg_candidates
            cand_idx = torch.randint(
                0, n_obs_i, (n_batch, m_cand), device=device
            )
            cand_z = cache_i[cand_idx].detach()  # [n_batch, M, d]
            # selection must not create autograd ops on z: detach for the
            # distances used ONLY to pick the negative
            d_cand = ((z.detach().unsqueeze(1) - cand_z) ** 2).sum(dim=2)
            # invalid candidates: the cell itself, or one of its
            # expression-space kNN members (those are positives)
            invalid = cand_idx == row_idx.to(device).unsqueeze(1)
            invalid |= (
                cand_idx.unsqueeze(2) == knn_rows.unsqueeze(1)
            ).any(-1)
            d_cand = d_cand.masked_fill(invalid, float("inf"))
            best_col = d_cand.argmin(dim=1)
            # if ALL M candidates are invalid, fall back to the first
            # candidate so the loss never sees inf/nan
            best_col = torch.where(
                invalid.all(dim=1), torch.zeros_like(best_col), best_col
            )
            neg_idx = cand_idx.gather(1, best_col.unsqueeze(1)).squeeze(1)
        else:
            # legacy path (bit-for-bit reproducible with prior runs): ONE
            # negative per cell, uniformly random cell of the same atlas
            neg_idx = torch.randint(0, n_obs_i, (n_batch,), device=device)
        z_neg = cache_i[neg_idx].detach()
        d_pos = ((z - z_pos) ** 2).sum(dim=1)
        d_neg = ((z - z_neg) ** 2).sum(dim=1)
        loss_i = F.relu(d_pos - d_neg + AnchorConfig.struct_margin).mean()
        total = loss_i if total is None else total + loss_i
        count += 1

    if total is None or count == 0:
        return None
    return AnchorConfig.struct_lambda * total / count


def compute_dis_loss_pretrain(
    model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    flagconfig,
):
    """
    Compute the discriminator loss for the pretraining phase.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model.
    flag_source_cat_single : torch.Tensor
        The source category for the single-cell data.
    flag_source_cat_spatial : torch.Tensor
        The source category for the spatial data.
    anneal : float
        The annealing factor.
    batch_features_all : list
        The list of features.
    adj_all : list
        The list of adjacency matrices.
    mask_batch_single : list
        The list of masks for the single-cell data.
    mask_batch_spatial : list
        The list of masks for the spatial data.
    flagconfig : FlagConfig
        The configuration flags.
    Returns
    -------
    dict
        The discriminator loss.
    
    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> flag_source_cat_single = torch.randn(10, 10)
    >>> flag_source_cat_spatial = torch.randn(10, 10)
    >>> anneal = 0.5
    >>> batch_features_all = [torch.randn(10, 10)]
    >>> adj_all = [torch.randn(10, 10)]
    >>> mask_batch_single = [torch.randn(10, 10)]
    >>> mask_batch_spatial = [torch.randn(10, 10)]
    >>> flagconfig = fusemap.config.FlagConfig()
    >>> compute_dis_loss_pretrain(
    ...     model,
    ...     flag_source_cat_single,
    ...     flag_source_cat_spatial,
    ...     anneal,
    ...     batch_features_all,
    ...     adj_all,
    ...     mask_batch_single,
    ...     mask_batch_spatial,
    ...     flagconfig
    ... )
    {'dis': tensor(0.0)}

    """
    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_all = [
        model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]
    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single
                + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(
                0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)
            ).sample((z_mean_cat_spatial.shape[0],))
            z_mean_cat_spatial = (
                z_mean_cat_spatial
                + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )

    ### compute dis loss
    loss_dis_single = F.cross_entropy(
        F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
        flag_source_cat_single[mask_batch_single_all],
        reduction="none",
    )
    loss_dis_single = loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
        flag_source_cat_spatial[mask_batch_spatial_all],
        reduction="none",
    )
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    loss_all = {"dis": loss_dis}
    return loss_all


def compute_ae_loss_pretrain(
    model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    flagconfig,
    anchor_state=None,
    row_index_all=None,
    col_index_all=None,
):
    """
    Compute the autoencoder loss for the pretraining phase.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    flag_source_cat_single : torch.Tensor
        The source category for the single-cell data.
    flag_source_cat_spatial : torch.Tensor
        The source category for the spatial data.   
    anneal : float
        The annealing factor.
    batch_features_all : list
        The list of features.
    adj_all : list
        The list of adjacency matrices.
    mask_batch_single : list    
        The list of masks for the single-cell data.
    mask_batch_spatial : list
        The list of masks for the spatial data.
    flagconfig : FlagConfig
        The configuration flags.
    Returns
    -------
    dict
        The autoencoder loss.
    
    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> flag_source_cat_single = torch.randn(10, 10)
    >>> flag_source_cat_spatial = torch.randn(10, 10)
    >>> anneal = 0.5
    >>> batch_features_all = [torch.randn(10, 10)]
    >>> adj_all = [torch.randn(10, 10)]
    >>> mask_batch_single = [torch.randn(10, 10)]
    >>> mask_batch_spatial = [torch.randn(10, 10)]
    >>> flagconfig = fusemap.config.FlagConfig()
    >>> compute_ae_loss_pretrain(
    ...     model,
    ...     flag_source_cat_single,
    ...     flag_source_cat_spatial,
    ...     anneal,
    ...     batch_features_all,
    ...     adj_all,
    ...     mask_batch_single,
    ...     mask_batch_spatial,
    ...     flagconfig
    ... )
    {'dis_ae': tensor(0.0), 'loss_AE_all': [tensor(0.0)], 'loss_all': tensor(0.0)}
    
    """
    z_all = [
        model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]

    # z_sample_all[i], 

    decoder_all = [
        model.decoder["atlas" + str(i)](z_spatial_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]

    z_distribution_loss = [
            z_all[i][0]
        
        for i in range(ModelType.n_atlas)
    ]
    loss_AE_all = [
        ModelType.lambda_ae_single.value
        * AE_Gene_loss(
            decoder_all[i][mask_batch_single[i], :],
            batch_features_all[i][mask_batch_single[i], :],
            z_distribution_loss[i],
        )
        for i in range(ModelType.n_atlas)
    ]

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single
                + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(
                0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)
            ).sample((z_mean_cat_spatial.shape[0],))
            z_mean_cat_spatial = (
                z_mean_cat_spatial
                + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )


    ### compute dis loss
    loss_dis_single = F.cross_entropy(
        F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
        flag_source_cat_single[mask_batch_single_all],
        reduction="none",
    )
    loss_dis_single = loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
        flag_source_cat_spatial[mask_batch_spatial_all],
        reduction="none",
    )
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)


    if (
        flagconfig.lambda_disc_single == 1
    ):  # and loss_dis.item()<sum(loss_AE_all).item()/DIS_LAMDA:
        flagconfig.lambda_disc_single = (
            sum(loss_AE_all).item() / ModelType.DIS_LAMDA.value / loss_dis.item()
        )
        print(f"lambda_disc_single changed to {flagconfig.lambda_disc_single}")
        loss_dis = flagconfig.lambda_disc_single * loss_dis

    # if ModelType.use_llm_gene_embedding=='combine':
    #     loss_part3 = compute_gene_embedding_loss(model)*10000
    #     loss_all = {
    #         "dis_ae": loss_dis,
    #         "loss_AE_all": loss_AE_all,
    #         "loss_all": -loss_dis + sum(loss_AE_all)+loss_part3,
    #     }
    # else:

    anchor_term = compute_anchor_loss(
        [z_all[i][3] for i in range(ModelType.n_atlas)],
        z_spatial_all,
        row_index_all,
        col_index_all,
        anchor_state,
    )
    loss_total = -loss_dis + sum(loss_AE_all)
    if anchor_term is not None:
        loss_total = loss_total + anchor_term
    loss_all = {
        "dis_ae": loss_dis,
        "loss_AE_all": loss_AE_all,
        "loss_all": loss_total,
        "anchor": anchor_term if anchor_term is not None else 0.0,
    }
    return loss_all


"""
final train loss
"""


def compute_dis_loss(
    model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    balance_weight_single_block,
    balance_weight_spatial_block,
    flagconfig,
):
    """
    Compute the discriminator loss for the final training phase.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    flag_source_cat_single : torch.Tensor
        The source category for the single-cell data.
    flag_source_cat_spatial : torch.Tensor
        The source category for the spatial data.
    anneal : float
        The annealing factor.
    batch_features_all : list
        The list of features.
    adj_all : list
        The list of adjacency matrices.
    mask_batch_single : list
        The list of masks for the single-cell data.
    mask_batch_spatial : list
        The list of masks for the spatial data.
    balance_weight_single_block : list
        The list of balance weights for the single-cell data.
    balance_weight_spatial_block : list
        The list of balance weights for the spatial data.
    flagconfig : FlagConfig
        The configuration flags.
    Returns
    -------
    dict
        The discriminator loss.

    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> flag_source_cat_single = torch.randn(10, 10)
    >>> flag_source_cat_spatial = torch.randn(10, 10)
    >>> anneal = 0.5
    >>> batch_features_all = [torch.randn(10, 10)]
    >>> adj_all = [torch.randn(10, 10)] 
    >>> mask_batch_single = [torch.randn(10, 10)]
    >>> mask_batch_spatial = [torch.randn(10, 10)]
    >>> balance_weight_single_block = [torch.randn(10, 10)]
    >>> balance_weight_spatial_block = [torch.randn(10, 10)]
    >>> flagconfig = fusemap.config.FlagConfig()
    >>> compute_dis_loss(
    ...     model,
    ...     flag_source_cat_single,
    ...     flag_source_cat_spatial,
    ...     anneal,
    ...     batch_features_all,
    ...     adj_all,
    ...     mask_batch_single,
    ...     mask_batch_spatial,
    ...     balance_weight_single_block,
    ...     balance_weight_spatial_block,   
    ...     flagconfig
    ... )
    {'dis': tensor(0.0)}

    """
    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)
    balance_weight_single_block = torch.hstack(balance_weight_single_block)
    balance_weight_spatial_block = torch.hstack((balance_weight_spatial_block))

    z_all = [
        model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]
    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single
                + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(
                0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)
            ).sample((z_mean_cat_spatial.shape[0],))
            z_mean_cat_spatial = (
                z_mean_cat_spatial
                + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )

    ### compute dis loss
    loss_dis_single = F.cross_entropy(
        F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
        flag_source_cat_single[mask_batch_single_all],
        reduction="none",
    )
    loss_dis_single = (
        balance_weight_single_block[mask_batch_single_all] * loss_dis_single
    ).sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
        flag_source_cat_spatial[mask_batch_spatial_all],
        reduction="none",
    )
    loss_dis_spatial = (
        balance_weight_spatial_block[mask_batch_spatial_all] * loss_dis_spatial
    ).sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    loss_all = {"dis": loss_dis}
    return loss_all


def compute_ae_loss(
    model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    balance_weight_single_block,
    balance_weight_spatial_block,
    flagconfig,
    anchor_state=None,
    row_index_all=None,
    col_index_all=None,
    anchor_pair_single=None,
    anchor_pair_spatial=None,
):
    """
    Compute the autoencoder loss for the final training phase.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model.
    flag_source_cat_single : torch.Tensor
        The source category for the single-cell data.
    flag_source_cat_spatial : torch.Tensor
        The source category for the spatial data.
    anneal : float
        The annealing factor.
    batch_features_all : list
        The list of features.
    adj_all : list
        The list of adjacency matrices.
    mask_batch_single : list
        The list of masks for the single-cell data.
    mask_batch_spatial : list
        The list of masks for the spatial data.
    balance_weight_single_block : list
        The list of balance weights for the single-cell data.
    balance_weight_spatial_block : list
        The list of balance weights for the spatial data.
    flagconfig : FlagConfig
        The configuration flags.
    Returns
    -------
    dict
        The autoencoder loss.

    Examples
    --------
    >>> import torch
    >>> import fusemap
    >>> model = fusemap.model.Fuse_network()
    >>> flag_source_cat_single = torch.randn(10, 10)
    >>> flag_source_cat_spatial = torch.randn(10, 10)
    >>> anneal = 0.5
    >>> batch_features_all = [torch.randn(10, 10)]
    >>> adj_all = [torch.randn(10, 10)]
    >>> mask_batch_single = [torch.randn(10, 10)]
    >>> mask_batch_spatial = [torch.randn(10, 10)]
    >>> balance_weight_single_block = [torch.randn(10, 10)]
    >>> balance_weight_spatial_block = [torch.randn(10, 10)]
    >>> flagconfig = fusemap.config.FlagConfig()
    >>> compute_ae_loss(
    ...     model,
    ...     flag_source_cat_single,
    ...     flag_source_cat_spatial,
    ...     anneal,
    ...     batch_features_all,
    ...     adj_all,
    ...     mask_batch_single,
    ...     mask_batch_spatial,
    ...     balance_weight_single_block,
    ...     balance_weight_spatial_block,
    ...     flagconfig
    ... )
    {'dis_ae': tensor(0.0), 'loss_AE_all': [tensor(0.0)], 'loss_all': tensor(0.0)}

    """
    z_all = [
        model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]

    decoder_all = [
        model.decoder["atlas" + str(i)]( z_spatial_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]

    ### compute AE loss
    # z_distribution_loss = [
    #     D.Normal(
    #         z_all[i][0][mask_batch_single[i], :], z_all[i][1][mask_batch_single[i], :]
    #     )
    #     for i in range(ModelType.n_atlas)
    # ]
    z_distribution_loss = [
        z_all[i][0]
        for i in range(ModelType.n_atlas)
    ]
    loss_AE_all = [
        ModelType.lambda_ae_single.value
        * AE_Gene_loss(
            decoder_all[i][mask_batch_single[i], :],
            batch_features_all[i][mask_batch_single[i], :],
            z_distribution_loss[i],
        )
        for i in range(ModelType.n_atlas)
    ]

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single
                + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(
                0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)
            ).sample((z_mean_cat_spatial.shape[0],))
            z_mean_cat_spatial = (
                z_mean_cat_spatial
                + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )

    ### compute dis loss
    bw_single_per_atlas = list(balance_weight_single_block)
    bw_spatial_per_atlas = list(balance_weight_spatial_block)
    balance_weight_single_block = torch.hstack(balance_weight_single_block)
    balance_weight_spatial_block = torch.hstack((balance_weight_spatial_block))

    loss_dis_single = F.cross_entropy(
        F.softmax(model.discriminator_single(z_mean_cat_single), dim=1),
        flag_source_cat_single[mask_batch_single_all],
        reduction="none",
    )
    loss_dis_single = (
        balance_weight_single_block[mask_batch_single_all] * loss_dis_single
    ).sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(model.discriminator_spatial(z_mean_cat_spatial), dim=1),
        flag_source_cat_spatial[mask_batch_spatial_all],
        reduction="none",
    )
    loss_dis_spatial = (
        balance_weight_spatial_block[mask_batch_spatial_all] * loss_dis_spatial
    ).sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    if (
        flagconfig.lambda_disc_single == 1
    ):  # and loss_dis.item()<sum(loss_AE_all).item()/DIS_LAMDA:
        flagconfig.lambda_disc_single = (
            sum(loss_AE_all).item() / ModelType.DIS_LAMDA.value / loss_dis.item()
        )
        print(f"lambda_disc_single changed to {flagconfig.lambda_disc_single}")
        loss_dis = flagconfig.lambda_disc_single * loss_dis

    # if ModelType.use_llm_gene_embedding=='combine':
    #     loss_part3 = compute_gene_embedding_loss(model)*10000
    #     loss_all = {
    #         "dis_ae": loss_dis,
    #         "loss_AE_all": loss_AE_all,
    #         "loss_all": -loss_dis + sum(loss_AE_all)+loss_part3,
    #     }
    # else:

    anchor_term = compute_anchor_loss(
        [z_all[i][3] for i in range(ModelType.n_atlas)],
        z_spatial_all,
        row_index_all,
        col_index_all,
        anchor_state,
        balance_weight_single=bw_single_per_atlas,
        balance_weight_spatial=bw_spatial_per_atlas,
        anchor_pair_single=anchor_pair_single,
        anchor_pair_spatial=anchor_pair_spatial,
    )
    struct_term = compute_struct_loss(
        [z_all[i][3] for i in range(ModelType.n_atlas)],
        row_index_all,
        anchor_state,
    )
    loss_total = -loss_dis + sum(loss_AE_all)
    if anchor_term is not None:
        loss_total = loss_total + anchor_term
    if struct_term is not None:
        loss_total = loss_total + struct_term
    loss_all = {
        "dis_ae": loss_dis,
        "loss_AE_all": loss_AE_all,
        "loss_all": loss_total,
        "anchor": anchor_term if anchor_term is not None else 0.0,
        "struct": struct_term if struct_term is not None else 0.0,
    }
    return loss_all


"""
balance weight part
"""


def get_balance_weight_subsample(leiden_adata_single, adatas_, key_leiden_category):
    """
    Compute the balance weight for the subsample.
    
    Parameters
    ----------
    leiden_adata_single : list
        The list of single-cell data.
    adatas_ : list
        The list of data.
    key_leiden_category : str
        The key for the leiden category.
    Returns
    -------
    list
        The balance weight.

    Examples
    --------
    >>> import fusemap
    >>> leiden_adata_single = [fusemap.data.load_sample_data()]
    >>> adatas_ = [fusemap.data.load_sample_data()]
    >>> key_leiden_category = 'leiden'
    >>> get_balance_weight_subsample(leiden_adata_single, adatas_, key_leiden_category)
    [tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            ...,
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]])]
            
    
    """
    ########### function from GLUE: https://github.com/gao-lab/GLUE

    us = [
        sklearn.preprocessing.normalize(leiden.X, norm="l2")
        for leiden in leiden_adata_single
    ]
    ns = [leiden.obs["size"] for leiden in leiden_adata_single]

    power = 4
    cutoff = 0.5
    while True:
        summary_balance_dict_sum = {}
        summary_balance_dict_multiply = {}
        summary_balance_dict_num = {}
        for i, ui in enumerate(us):
            for j, uj in enumerate(us[i + 1 :], start=i + 1):
                cosine = ui @ uj.T
                cosine[cosine < cutoff] = 0
                cosine = COO.from_numpy(cosine)
                cosine = np.power(cosine, power)

                for ind in [i, j]:
                    if ind == i:
                        balancing = cosine.sum(axis=1).todense() / ns[ind]
                    else:
                        balancing = cosine.sum(axis=0).todense() / ns[ind]
                    balancing = pd.Series(
                        balancing, index=leiden_adata_single[ind].obs_names
                    )
                    balancing = balancing.loc[
                        adatas_[ind].obs[key_leiden_category]
                    ].to_numpy()
                    balancing /= balancing.sum() / balancing.size
                    if ind in summary_balance_dict_sum:
                        summary_balance_dict_sum[ind] += balancing.copy()
                        summary_balance_dict_multiply[ind] *= balancing.copy()
                        summary_balance_dict_num[ind] += 1
                    else:
                        summary_balance_dict_sum[ind] = balancing.copy()
                        summary_balance_dict_multiply[ind] = balancing.copy()
                        summary_balance_dict_num[ind] = 1
        flag = 0
        for i in range(len(summary_balance_dict_sum)):
            if sum(np.isnan(summary_balance_dict_sum[i])) > 0:
                flag = 1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(np.isnan(summary_balance_dict_multiply[i])) > 0:
                flag = 1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(summary_balance_dict_sum[i]) == 0:
                flag = 1
                break
        for i in range(len(summary_balance_dict_multiply)):
            if sum(summary_balance_dict_multiply[i]) == 0:
                flag = 1
                break
        if flag == 1:
            cutoff -= 0.1
        else:
            break
    print(f"balance weight final cutoff: {cutoff}")
    for i in range(len(summary_balance_dict_sum)):
        if (
            summary_balance_dict_sum[i][summary_balance_dict_sum[i] == np.inf].shape[0]
            > 0
        ):
            print(
                i,
                "inf:",
                summary_balance_dict_sum[i][
                    summary_balance_dict_sum[i] == np.inf
                ].shape[0],
            )
            summary_balance_dict_sum[i][summary_balance_dict_sum[i] == np.inf] = 1e308

    for i in range(len(summary_balance_dict_sum)):
        if (
            summary_balance_dict_multiply[i][
                summary_balance_dict_multiply[i] == np.inf
            ].shape[0]
            > 0
        ):
            print(
                i,
                "inf:",
                summary_balance_dict_multiply[i][
                    summary_balance_dict_multiply[i] == np.inf
                ].shape[0],
            )
            summary_balance_dict_multiply[i][
                summary_balance_dict_multiply[i] == np.inf
            ] = 1e308

    balance_weight = []
    summary_balance_dict = {}
    for i in range(len(us)):
        test1 = summary_balance_dict_sum[i] / (
            summary_balance_dict_sum[i].sum() / summary_balance_dict_sum[i].size
        )
        test2 = summary_balance_dict_multiply[i] / (
            summary_balance_dict_multiply[i].sum()
            / summary_balance_dict_multiply[i].size
        )
        test = 0.9 * test1 + 0.1 * test2
        test /= test.sum() / test.size
        summary_balance_dict[i] = test.copy()
        balance_weight.append(summary_balance_dict[i])
    return balance_weight


def get_balance_weight(adatas, leiden_adata_single, adatas_, key_leiden_category, cutoff=None, power=None):
    ########### function from GLUE: https://github.com/gao-lab/GLUE
    us = [
        preprocessing.normalize(leiden.X, norm="l2")
        for leiden in leiden_adata_single
    ]
    ns = [leiden.obs["size"] for leiden in leiden_adata_single]

    import os
    cosines = []
    # defaults = STRICT matching (2026-09-01 validated): best disease-state
    # preservation + highest mixing; use 0.5/4 for max transfer accuracy (BWR4)
    if cutoff is None:
        cutoff = float(os.environ.get("FUSEMAP_BALANCE_CUTOFF", "0.75"))
    if power is None:
        power = float(os.environ.get("FUSEMAP_BALANCE_POWER", "8"))

    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1 :], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = prod(cosines)

    if joint_cosine.coords.shape[0] == 0:
        raise ValueError(
            "Balance weight computation error! No correlation between samples or lower cutoff!"
        )
    #
    balance_weight = []
    for i, (adata, adata_, leiden, n) in enumerate(
        zip(adatas, adatas_, leiden_adata_single, ns)
    ):
        balancing = (
            joint_cosine.sum(
                axis=tuple(k for k in range(joint_cosine.ndim) if k != i)
            ).todense()
            / n
        )
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs[key_leiden_category]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        balance_weight.append(balancing)
    return balance_weight


"""
train ref data part
"""


def compute_dis_loss_map(
    adapt_model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    pretrain_single_batch,
    pretrain_spatial_batch,
    flag_source_cat_single_pretrain,
    flag_source_cat_spatial_pretrain,
    flagconfig,
):
    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_all = [
        adapt_model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]
    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]
    z_mean_cat_single = torch.vstack(
        [
            z_mean_cat_single,
            torch.cat(
                [pretrain_single_batch[i] for i in range(len(pretrain_single_batch))]
            ),
        ]
    )

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]
    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]
    z_mean_cat_spatial = torch.vstack(
        [
            z_mean_cat_spatial,
            torch.cat(
                [pretrain_spatial_batch[i] for i in range(len(pretrain_spatial_batch))]
            ),
        ]
    )

    ######### append pretrained data ##############

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single
                + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(
                0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)
            ).sample((z_mean_cat_spatial.shape[0],))
            z_mean_cat_spatial = (
                z_mean_cat_spatial
                + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )

    ### compute dis loss
    loss_dis_single = F.cross_entropy(
        F.softmax(
            torch.hstack(
                [
                    adapt_model.discriminator_single(z_mean_cat_single),
                    adapt_model.discriminator_single_pretrain(z_mean_cat_single),
                ]
            ),
            dim=1,
        ),
        torch.hstack(
            [
                flag_source_cat_single[mask_batch_single_all],
                flag_source_cat_single_pretrain,
            ]
        ),
        reduction="none",
    )
    loss_dis_single = loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(
            torch.hstack(
                [
                    adapt_model.discriminator_spatial(z_mean_cat_spatial),
                    adapt_model.discriminator_spatial_pretrain(z_mean_cat_spatial),
                ]
            ),
            dim=1,
        ),
        torch.hstack(
            [
                flag_source_cat_spatial[mask_batch_spatial_all],
                flag_source_cat_spatial_pretrain,
            ]
        ),
        reduction="none",
    )
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)
    # loss_dis = self.lambda_disc_single * (loss_dis_single )

    loss_all = {"dis": loss_dis}
    return loss_all


def compute_ae_loss_map(
    adapt_model,
    flag_source_cat_single,
    flag_source_cat_spatial,
    anneal,
    batch_features_all,
    adj_all,
    mask_batch_single,
    mask_batch_spatial,
    pretrain_single_batch,
    pretrain_spatial_batch,
    flag_source_cat_single_pretrain,
    flag_source_cat_spatial_pretrain,
    flagconfig
):
    z_all = [
        adapt_model.encoder["atlas" + str(i)](batch_features_all[i], adj_all[i])
        for i in range(ModelType.n_atlas)
    ]

    # z_distribution_all = [
    #     z_all[i][0] for i in range(ModelType.n_atlas)
    # ]
    # z_sample_all = [z_distribution_all[i].rsample() for i in range(ModelType.n_atlas)]

    z_spatial_all = [z_all[i][2] for i in range(ModelType.n_atlas)]

    decoder_all = [
        adapt_model.decoder["atlas" + str(i)](
            # z_all[i][1],
            z_spatial_all[i],
            adj_all[i],
            adapt_model.gene_embedding_pretrained,
            adapt_model.gene_embedding_new,
        )
        for i in range(ModelType.n_atlas)
    ]

    ### compute AE loss
    z_distribution_loss = [
            z_all[i][0] 
        for i in range(ModelType.n_atlas)
    ]
    loss_AE_all = [
        ModelType.lambda_ae_single.value
        * AE_Gene_loss(
            decoder_all[i][mask_batch_single[i], :],
            batch_features_all[i][mask_batch_single[i], :],
            z_distribution_loss[i],
        )
        for i in range(ModelType.n_atlas)
    ]

    mask_batch_single_all = torch.hstack(mask_batch_single)
    mask_batch_spatial_all = torch.hstack(mask_batch_spatial)

    z_mean_cat_single = torch.cat([z_all[i][3] for i in range(ModelType.n_atlas)])[
        mask_batch_single_all, :
    ]
    z_mean_cat_single = torch.vstack(
        [
            z_mean_cat_single,
            torch.cat(
                [pretrain_single_batch[i] for i in range(len(pretrain_single_batch))]
            ),
        ]
    )

    z_mean_cat_spatial = torch.cat(z_spatial_all)[mask_batch_spatial_all, :]
    z_mean_cat_spatial = torch.vstack(
        [
            z_mean_cat_spatial,
            torch.cat(
                [pretrain_spatial_batch[i] for i in range(len(pretrain_spatial_batch))]
            ),
        ]
    )

    if anneal:
        if z_mean_cat_single.shape[0] > 1:
            noise_single = D.Normal(0, z_mean_cat_single.std(axis=0)).sample(
                (z_mean_cat_single.shape[0],)
            )
            z_mean_cat_single = (
                z_mean_cat_single + (anneal * ModelType.align_noise_coef.value) * noise_single
            )
        if z_mean_cat_spatial.shape[0] > 1:
            noise_spatial = D.Normal(0, ModelType.EPS.value + z_mean_cat_spatial.std(axis=0)).sample(
                (z_mean_cat_spatial.shape[0],)
            )
            z_mean_cat_spatial = (
                z_mean_cat_spatial + (anneal * ModelType.align_noise_coef.value) * noise_spatial
            )

    ### compute dis loss
    loss_dis_single = F.cross_entropy(
        F.softmax(
            torch.hstack(
                [
                    adapt_model.discriminator_single(z_mean_cat_single),
                    adapt_model.discriminator_single_pretrain(z_mean_cat_single),
                ]
            ),
            dim=1,
        ),
        torch.hstack(
            [
                flag_source_cat_single[mask_batch_single_all],
                flag_source_cat_single_pretrain,
            ]
        ),
        reduction="none",
    )
    loss_dis_single = loss_dis_single.sum() / loss_dis_single.numel()

    loss_dis_spatial = F.cross_entropy(
        F.softmax(
            torch.hstack(
                [
                    adapt_model.discriminator_spatial(z_mean_cat_spatial),
                    adapt_model.discriminator_spatial_pretrain(z_mean_cat_spatial),
                ]
            ),
            dim=1,
        ),
        torch.hstack(
            [
                flag_source_cat_spatial[mask_batch_spatial_all],
                flag_source_cat_spatial_pretrain,
            ]
        ),
        reduction="none",
    )
    loss_dis_spatial = loss_dis_spatial.sum() / loss_dis_spatial.numel()

    loss_dis = flagconfig.lambda_disc_single * (loss_dis_single + loss_dis_spatial)

    if (
        flagconfig.lambda_disc_single == 1
    ):  # and loss_dis.item()<sum(loss_AE_all).item()/DIS_LAMDA:
        flagconfig.lambda_disc_single = sum(loss_AE_all).item() / ModelType.DIS_LAMDA.value / loss_dis.item()
        logging.info(f"\n\nlambda_disc_single changed to {flagconfig.lambda_disc_single}\n")
        loss_dis = flagconfig.lambda_disc_single * loss_dis

    loss_all = {
        "dis_ae": loss_dis,
        "loss_AE_all": loss_AE_all,
        "loss_all": -loss_dis + sum(loss_AE_all),
    }
    return loss_all
