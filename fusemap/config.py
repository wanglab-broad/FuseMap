import argparse
import os
from enum import Enum


def parse_input_args():
    parser = argparse.ArgumentParser(description="FuseMap")

    parser.add_argument(
        "--input_data_folder_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_save_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--keep_celltype",
        type=str,
        default="",
    )
    parser.add_argument(
        "--keep_tissueregion",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use_llm_gene_embedding",
        default='false',
    )
    parser.add_argument(
        "--pretrain_model_path",
        default="",
    )
    args = parser.parse_args()
    return args


class FlagConfig:
    lambda_disc_single = 1
    align_anneal = 1e10


class ModelType(Enum):
    pca_dim = 50
    hidden_dim = 512
    latent_dim = 64
    dropout_rate = 0.2
    n_epochs = 16
    batch_size = 64
    learning_rate = 0.001
    optim_kw = "RMSprop"
    use_input = "norm"
    lambda_ae_single = 1
    lambda_disc_spatial = 1
    lambda_ae_spatial = 1
    align_noise_coef = 1.5
    lr_patience_pretrain = 2
    lr_factor_pretrain = 0.5
    lr_limit_pretrain = 0.00001
    patience_limit_final = 5
    lr_patience_final = 3
    lr_factor_final = 0.5
    lr_limit_final = 0.00001
    patience_limit_pretrain = 3
    EPS = 1e-10
    DIS_LAMDA = 2
    TRAIN_WITHOUT_EVAL = 10
    USE_REFERENCE_PCT = 0.02
    verbose = False
    use_key='final'


class AnchorConfig:
    """Cross-sample MNN anchor alignment (STAligner-style), loss-level only.

    This is a plain class (NOT a ``ModelType`` Enum member; the Enum has
    value-aliasing quirks). Setting ``anchor_lambda = 0`` disables every anchor
    code path and reproduces the original FuseMap behavior exactly.
    """

    # default 0.3 = validated optimal (2026-09-01); override via env var
    anchor_lambda = float(os.environ.get("FUSEMAP_ANCHOR_LAMBDA", "0.3"))
    anchor_refresh_epochs = 2    # recompute MNN anchors every N epochs
    anchor_k = 15                # kNN used per direction for MNN
    anchor_start_epoch = int(os.environ.get("FUSEMAP_ANCHOR_START", "2"))
    # quality filters (2026-09-01): drop low-similarity MNN pairs and pairs not
    # stable across consecutive refreshes; disable anchors during pretrain by
    # default (immature latent locks in wrong anchors).
    anchor_sim_threshold = float(os.environ.get("FUSEMAP_ANCHOR_SIM", "0.5"))
    # comma-separated file-name substrings marking QUERY datasets (e.g. bead
    # data): they get pulled toward others, but nothing is pulled toward them.
    anchor_query_files = os.environ.get("FUSEMAP_ANCHOR_QUERY", "")
    anchor_query_atlases = set()  # resolved at runtime from input file names
    anchor_stable_only = os.environ.get("FUSEMAP_ANCHOR_STABLE", "1") == "1"
    anchor_in_pretrain = os.environ.get("FUSEMAP_ANCHOR_PRETRAIN", "0") == "1"

    # Within-dataset structure-preservation triplet loss (opt-in).
    # struct_lambda = 0 keeps the feature fully OFF and reproduces current
    # behavior EXACTLY (no extra RNG draws, no extra compute anywhere).
    # When > 0, spatial_integrate precomputes per-atlas expression-space kNN
    # (stored at runtime as ``AnchorConfig.struct_knn``) and
    # ``fusemap.loss.compute_struct_loss`` adds a triplet term in the final
    # training phase that keeps each cell's expression neighbors closer in
    # latent space than random same-dataset cells.
    struct_lambda = float(os.environ.get("FUSEMAP_STRUCT_LAMBDA", "0"))
    struct_k = int(os.environ.get("FUSEMAP_STRUCT_K", "10"))
    struct_margin = float(os.environ.get("FUSEMAP_STRUCT_MARGIN", "1.0"))
    # Semi-hard negative mining (default ON when struct loss is enabled):
    # uniformly random negatives are already far apart in the compact latent,
    # so the repulsion term is inert and the triplet degenerates into extra
    # compression (measured: type purity 0.578 -> 0.502). Instead mine the
    # hardest VALID negative among struct_neg_candidates random candidates.
    # Set FUSEMAP_STRUCT_HARDNEG=0 to reproduce the legacy random-negative
    # path bit-for-bit (same number and order of RNG draws).
    struct_hardneg = os.environ.get("FUSEMAP_STRUCT_HARDNEG", "1") == "1"
    struct_neg_candidates = int(os.environ.get("FUSEMAP_STRUCT_NEGC", "16"))


class AnchorState:
    """Runtime container for anchor alignment state.

    Populated by ``fusemap.train_model.refresh_anchors`` and consumed by the AE
    loss (``fusemap.loss.compute_anchor_loss``). Kept dependency-free (plain
    attributes) so it can live in the config module.

    Attributes
    ----------
    single_cache : list of torch.Tensor or None
        Per-atlas full-dataset single latent (z_mean), detached, [n_obs_i, d].
    spatial_cache : list of torch.Tensor or None
        Per-atlas full-dataset spatial latent (z_spatial), detached.
    partner : dict
        (i, j) -> int64 tensor of length n_obs_i giving, for each cell in atlas
        i, the index of its MNN partner cell in atlas j (or -1 if none).
    n_pairs : int
        Total number of MNN anchor pairs found at the last refresh.
    mean_dist : float
        Mean cross-atlas single-latent distance of anchored pairs at refresh.
    has_anchors : bool
        Whether any anchors are currently stored.
    """

    def __init__(self):
        self.prev_pairs = {}
        self.single_cache = None
        self.spatial_cache = None
        self.partner = {}
        self.n_pairs = 0
        self.mean_dist = float("nan")
        self.has_anchors = False
