import argparse
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
