from fusemap.model import Fuse_network
from fusemap.preprocess import *
from fusemap.dataset import *
from fusemap.loss import *
from fusemap.config import *
from fusemap.utils import *
from fusemap.train_model import *
from pathlib import Path
import dgl.dataloading as dgl_dataload
import os
import torch
import logging
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle





def train(X_input, save_dir, kneighbor, input_identity, 
          data_pth=None, preprocess_save=False):
    """
    A function to train FuseMap model.

    Parameters
    ----------
    X_input : list
        A list of anndata objects, each representing a spatial section.
    save_dir : str
        The path to save the data.
    kneighbor : list
        A list of strings, each representing the method to calculate the k-nearest neighbors.
    input_identity : list
        A list of strings, each representing the identity of the input data.
    data_pth : str
        The path to save the data.

        
    Examples
    --------
    >>> import fusemap
    >>> import scanpy as sc
    >>> import os
    >>> spatial_integrate(
    ...     [sc.read_h5ad(f) for f in os.listdir('data') if f.endswith('.h5ad')],
    ...     'data',
    ...     ['delaunay']*len(X_input),
    ...     ['ST']*len(X_input),
    ...     'data'
    ... )


    """
    # ModelType = parse_ModelType()

    ModelType.preprocess_save = preprocess_save
    ModelType.data_pth = data_pth
    ModelType.save_dir = save_dir
    ModelType.kneighbor = kneighbor
    ModelType.input_identity = input_identity

    ### preprocess
    ModelType.snapshot_path = f"{ModelType.save_dir}/snapshot.pt"
    Path(f"{ModelType.save_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{ModelType.save_dir}/trained_model").mkdir(parents=True, exist_ok=True)

    ModelType.n_atlas = len(X_input)
    if ModelType.preprocess_save == False:
        preprocess_raw(
            X_input,
            ModelType.kneighbor,
            ModelType.input_identity,
            ModelType.use_input.value,
            ModelType.n_atlas,
            ModelType.data_pth,
        )
    for i in range(ModelType.n_atlas):
        X_input[i].var.index = [i.upper() for i in X_input[i].var.index]
    adatas = X_input
    ModelType.n_obs = [adatas[i].shape[0] for i in range(ModelType.n_atlas)]
    ModelType.input_dim = [adatas[i].n_vars for i in range(ModelType.n_atlas)]
    ModelType.var_name = [list(i.var.index) for i in adatas]

    all_unique_genes = sorted(list(get_allunique_gene_names(*ModelType.var_name)))
    logging.info(
        f"\n\nnumber of genes in each section:{[len(i) for i in ModelType.var_name]}, Number of all genes: {len(all_unique_genes)}\n"
    )

    ### model
    model = Fuse_network(
        ModelType.pca_dim.value,
        ModelType.input_dim,
        ModelType.hidden_dim.value,
        ModelType.latent_dim.value,
        ModelType.dropout_rate.value,
        ModelType.var_name,
        all_unique_genes,
        ModelType.use_input.value,
        ModelType.harmonized_gene,
        ModelType.n_atlas,
        ModelType.input_identity,
        ModelType.n_obs,
        ModelType.n_epochs.value,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ModelType.epochs_run_pretrain = 0
    ModelType.epochs_run_final = 0
    if os.path.exists(ModelType.snapshot_path):
        logging.info("\n\nLoading snapshot\n")
        load_snapshot(model, ModelType.snapshot_path, device)

    ### construct graph and data
    adj_all, g_all = construct_data(ModelType.n_atlas, adatas, ModelType.input_identity, model)
    feature_all = [
        get_feature_sparse(device, adata.obsm["spatial_input"]) for adata in adatas
    ]
    spatial_dataset_list = [
        CustomGraphDataset(i, j, ModelType.use_input) for i, j in zip(g_all, adatas)
    ]
    spatial_dataloader = CustomGraphDataLoader(
        spatial_dataset_list,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        ModelType.batch_size.value,
        shuffle=True,
        n_atlas=ModelType.n_atlas,
        drop_last=False,
    )
    spatial_dataloader_test = CustomGraphDataLoader(
        spatial_dataset_list,
        dgl_dataload.MultiLayerFullNeighborSampler(1),
        ModelType.batch_size.value,
        shuffle=False,
        n_atlas=ModelType.n_atlas,
        drop_last=False,
    )
    train_mask, val_mask = construct_mask(ModelType.n_atlas, spatial_dataset_list, g_all)

    ### train
    flagconfig=FlagConfig()
    if os.path.exists(f"{ModelType.save_dir}/lambda_disc_single.pkl"):
        with open(f"{ModelType.save_dir}/lambda_disc_single.pkl", "rb") as openfile:
            flagconfig.lambda_disc_single = pickle.load(openfile)

    
    if not os.path.exists(
        f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt"
    ):
        logging.info(
            "\n\n---------------------------------- Phase 1. Pretrain FuseMap model ----------------------------------\n"
        )
        pretrain_model(
            model,
            spatial_dataloader,
            feature_all,
            adj_all,
            device,
            train_mask,
            val_mask,
            flagconfig,
        )

    if not os.path.exists(f"{ModelType.save_dir}/latent_embeddings_all_single_pretrain.pkl"):
        logging.info(
            "\n\n---------------------------------- Phase 2. Evaluate pretrained FuseMap model ----------------------------------\n"
        )
        if os.path.exists(
            f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt"
        ):
            read_model(
                model,
                spatial_dataloader_test,
                g_all,
                feature_all,
                adj_all,
                device,
                ModelType,
                mode="pretrain",
            )
        else:
            raise ValueError("No pretrained model!")

    if not os.path.exists(f"{ModelType.save_dir}/balance_weight_single.pkl"):
        logging.info(
            "\n\n---------------------------------- Phase 3. Estimate_balancing_weight ----------------------------------\n"
        )
        balance_weight(model, adatas, ModelType.save_dir, ModelType.n_atlas, device)

    if not os.path.exists(
        f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt"
    ):
        model.load_state_dict(
            torch.load(f"{ModelType.save_dir}/trained_model/FuseMap_pretrain_model_final.pt")
        )
        logging.info(
            "\n\n---------------------------------- Phase 4. Train final FuseMap model ----------------------------------\n"
        )
        train_model(
            model,
            spatial_dataloader,
            feature_all,
            adj_all,
            device,
            train_mask,
            val_mask,
            flagconfig,
        )

    if not os.path.exists(f"{ModelType.save_dir}/latent_embeddings_all_single_final.pkl"):
        logging.info(
            "\n\n---------------------------------- Phase 5. Evaluate final FuseMap model ----------------------------------\n"
        )
        if os.path.exists(
            f"{ModelType.save_dir}/trained_model/FuseMap_final_model_final.pt"
        ):
            read_model(
                model,
                spatial_dataloader_test,
                g_all,
                feature_all,
                adj_all,
                device,
                ModelType,
                mode="final",
            )
        else:
            raise ValueError("No final model!")

    logging.info(
        "\n\n---------------------------------- Finish ----------------------------------\n"
    )

    ### read out gene embedding
    read_gene_embedding(
        model, all_unique_genes, ModelType.save_dir, ModelType.n_atlas, ModelType.var_name
    )

    ### read out cell embedding
    annotation_transfer(
        adatas,
        ModelType.save_dir,
    )

    return
