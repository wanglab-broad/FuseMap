"""Shared Stage-B objective for integration and frozen-reference mapping."""

import logging

import numpy as np
import scipy.sparse as sp
import torch


def fit_mixtures(expression, signatures, gene_mask=None, entropy_weight=5e-4,
                 n_steps=800, chunk_size=4096, device=None):
    """Fit simplex weights and positive per-gene platform terms.

    Minimizes mean squared error + entropy_weight * mean entropy, with
    prediction ``softplus(beta) * (softmax(logits) @ signatures) + softplus(alpha)``.
    Signatures are fixed. Only bead logits and platform terms are optimized.
    Cache at most 512 MiB of dense expression on the device. Larger inputs are
    transferred one chunk at a time to bound GPU memory.
    """
    x = sp.csr_matrix(expression, dtype=np.float32)
    s = np.asarray(signatures, dtype=np.float32)
    if (x.shape[0] == 0 or s.ndim != 2 or s.shape[0] == 0
            or s.shape[1] != x.shape[1]):
        raise ValueError("Expression and signatures must have matching, nonempty dimensions")
    if not np.isfinite(x.data).all() or not np.isfinite(s).all():
        raise ValueError("Expression and signatures must be finite")
    if (x.data < 0).any() or (s < 0).any():
        raise ValueError("Stage-B requires nonnegative expression and signatures")
    mask = np.ones(x.shape[1], dtype=bool) if gene_mask is None else np.asarray(gene_mask, dtype=bool)
    if mask.shape != (x.shape[1],) or not mask.any():
        raise ValueError("No shared genes between query and reference signatures")
    if not np.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError("entropy_weight must be finite and nonnegative")
    if n_steps < 1 or chunk_size < 1:
        raise ValueError("n_steps and chunk_size must be positive")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Genes outside the reference panel do not enter the objective.
    x = x[:, mask].tocsr()
    s = torch.as_tensor(s[:, mask], device=device)
    n_bead, n_gene = x.shape
    logits = torch.zeros(n_bead, s.shape[0], device=device, requires_grad=True)
    beta_raw = torch.full((n_gene,), float(np.log(np.expm1(1.0))),
                          device=device, requires_grad=True)
    alpha_raw = torch.full((n_gene,), float(np.log(np.expm1(0.01))),
                           device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits, beta_raw, alpha_raw], lr=0.05)
    # Keep small/medium runs as fast as the original Stage-B implementation;
    # repeated sparse-to-dense conversion can dominate the optimization time.
    cached_chunks = None
    if n_bead * n_gene * np.dtype(np.float32).itemsize <= 512 * 1024**2:
        cached_chunks = {start: torch.as_tensor(x[start:start + chunk_size].toarray(), device=device)
                         for start in range(0, n_bead, chunk_size)}

    def expression_chunk(start, stop):
        if cached_chunks is not None:
            return cached_chunks[start]
        return torch.as_tensor(x[start:stop].toarray(), device=device)

    previous_loss = None
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        total_se = total_ent = 0.0
        for start in range(0, n_bead, chunk_size):
            stop = min(start + chunk_size, n_bead)
            xb = expression_chunk(start, stop)
            pi = torch.softmax(logits[start:stop], dim=1)
            pred = torch.nn.functional.softplus(beta_raw) * (pi @ s)
            pred = pred + torch.nn.functional.softplus(alpha_raw)
            se = (xb - pred).square().sum()
            ent = -(pi * torch.log(pi + 1e-8)).sum()
            loss = se / (n_bead * n_gene) + entropy_weight * ent / n_bead
            loss.backward()
            total_se += se.item()
            total_ent += ent.item()
        total_loss = total_se / (n_bead * n_gene) + entropy_weight * total_ent / n_bead
        if not np.isfinite(total_loss):
            raise ValueError("Non-finite Stage-B loss; check expression normalization")
        optimizer.step()
        if step % 100 == 0:
            logging.info("[Stage-B] step=%d mse=%.6g entropy=%.6g", step,
                         total_se / (n_bead * n_gene), total_ent / n_bead)
        if previous_loss is not None and abs(previous_loss - total_loss) < 1e-7 and step > 200:
            break
        previous_loss = total_loss

    with torch.no_grad():
        pi = torch.softmax(logits, dim=1)
        beta = torch.nn.functional.softplus(beta_raw)
        alpha = torch.nn.functional.softplus(alpha_raw)
        squared_error = np.empty(n_bead, dtype=np.float32)
        for start in range(0, n_bead, chunk_size):
            stop = min(start + chunk_size, n_bead)
            xb = expression_chunk(start, stop)
            pred = beta * (pi[start:stop] @ s) + alpha
            squared_error[start:stop] = (xb - pred).square().mean(1).cpu().numpy()
    return dict(pi=pi.cpu().numpy(), beta=beta.cpu().numpy(), alpha=alpha.cpu().numpy(),
                gene_mask=mask, reconstruction_mse=squared_error, n_steps=step + 1)
