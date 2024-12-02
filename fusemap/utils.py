from torch.utils.data import DataLoader, TensorDataset
import logging
import os
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle
import torch
import anndata as ad
import pandas as pd
import numpy as np
import dgl
import random
from fusemap.model import NNTransfer
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle



def seed_all(seed_value, cuda_deterministic=True):
    logging.info(
        "\n\n---------------------------------- SEED ALL: {seed_value}  ----------------------------------\n"
    )

    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    dgl.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(
            seed_value
        )  # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic:  # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True



def save_obj(objt, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(objt, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_snapshot(model, snapshot_path, loc):
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run_pretrain = snapshot["EPOCHS_RUN_pretrain"]
    epochs_run_final = snapshot["EPOCHS_RUN_final"]
    logging.info(
        f"\n\nResuming training from snapshot at pretrain Epoch {epochs_run_pretrain}, final epoch {epochs_run_final}\n"
    )


def save_snapshot(model, epoch_pretrain, epoch_final, snapshot_path,verbose):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "EPOCHS_RUN_pretrain": epoch_pretrain,
        "EPOCHS_RUN_final": epoch_final,
    }
    torch.save(snapshot, snapshot_path)
    if verbose == True:
        logging.info(
            f"\n\nPretrain Epoch {epoch_pretrain}, final Epoch {epoch_final} | Training snapshot saved at {snapshot_path}\n"
        )


def average_embeddings(adata, category, obsm_latent):
    """
    Calculate the average embeddings for each category in the AnnData object

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the embeddings
    category : str
        Column name in adata.obs containing the category information
    obsm_latent : str
        Key in adata.obsm containing the latent embeddings

    Returns
    -------
    new_adata : AnnData
        Anndata object containing the average embeddings

    Examples    
    --------
    >>> adata = sc.read_h5ad("path/to/adata.h5ad")
    >>> new_adata = average_embeddings(adata, "cell_type", "latent")
    """
    latent_df = pd.DataFrame(adata.obsm[obsm_latent], index=adata.obs[category])
    mean_embeddings = latent_df.groupby(level=0).mean()

    # Calculate the number of cells in each category
    num_cells = latent_df.groupby(level=0).size()

    # Create a new AnnData object with the average embeddings
    new_adata = ad.AnnData(mean_embeddings)
    new_adata.obs["size"] = num_cells

    return new_adata


def read_gene_embedding(model, all_unique_genes, save_dir, n_atlas, var_name):
    if not os.path.exists(f"{save_dir}/ad_gene_embedding.h5ad"):
        ad_gene_embedding = ad.AnnData(X=model.gene_embedding.detach().cpu().numpy().T)
        ad_gene_embedding.obs.index = all_unique_genes
        for i in range(n_atlas):
            ad_gene_embedding.obs["sample" + str(i)] = ""
            for gene in var_name[i]:
                ad_gene_embedding.obs.loc[gene, "sample" + str(i)] = f"sample_{str(i)}"
        ad_gene_embedding.obs["type"] = ad_gene_embedding.obs[
            [f"sample{i}" for i in range(n_atlas)]
        ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        
        ad_gene_embedding.obs['type'] = 'type'+ad_gene_embedding.obs['type'].astype('str')
        ad_gene_embedding.write_h5ad(f"{save_dir}/ad_gene_embedding.h5ad")
    return


def read_gene_embedding_map(model,  new_train_gene, PRETRAINED_GENE, save_dir, n_atlas, var_name):
    if not os.path.exists(f"{save_dir}/ad_gene_embedding.h5ad"):
        emb_new = model.gene_embedding_new.detach().cpu().numpy().T
        emb_pretrain = model.gene_embedding_pretrained.detach().cpu().numpy().T
        gene_emb=np.vstack((emb_pretrain, emb_new))
        ad_gene_embedding = ad.AnnData(X=gene_emb)
        ad_gene_embedding.obs.index = new_train_gene+PRETRAINED_GENE
        for i in range(n_atlas):
            ad_gene_embedding.obs["sample" + str(i)] = ""
            for gene in var_name[i]:
                ad_gene_embedding.obs.loc[gene, "sample" + str(i)] = f"sample_{str(i)}"
        ad_gene_embedding.obs["type"] = ad_gene_embedding.obs[
            [f"sample{i}" for i in range(n_atlas)]
        ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)
        ad_gene_embedding.obs['type'] = 'type'+ad_gene_embedding.obs['type'].astype('str')

        ad_gene_embedding.write_h5ad(f"{save_dir}/ad_gene_embedding.h5ad")
    return


def generate_ad_embed(save_dir, X_input, keep_label, ttype, use_key="final"):
    with open(
        save_dir + f"/latent_embeddings_all_{ttype}_{use_key}.pkl", "rb"
    ) as openfile:
        latent_embeddings_all = pickle.load(openfile)
    ad_list = []
    for ind, (X_input_i, latent_embeddings_all_i) in enumerate(
        zip(X_input, latent_embeddings_all)
    ):
        ad_embed_1 = ad.AnnData(X=latent_embeddings_all_i)
        ad_embed_1.obs["x"] = list(X_input_i.obs["x"])
        ad_embed_1.obs["y"] = list(X_input_i.obs["y"])
        ad_embed_1.obs["name"] = list(X_input_i.obs["name"])  # f'sample{ind}'
        ad_embed_1.obs["batch"] = f"sample{ind}"
        ad_embed_1.obs["file_name"] = list(X_input_i.obs["file_name"])
        if keep_label!="":
            try:
                ad_embed_1.obs[keep_label] = list(X_input_i.obs[keep_label])
            except:
                ad_embed_1.obs[keep_label] = 'NA'

        ad_list.append(ad_embed_1)
    ad_embed = ad.concat(ad_list)

    try:
        origin_concat=ad.concat(X_input,join='outer').obs
    except:
        origin_concat_old=ad.concat(X_input).obs
        origin_concat=origin_concat_old.copy()
        for i in range(len(X_input)):
            X_input_i=X_input[i]
            for j in X_input_i.obs.columns:
                if j not in origin_concat_old.columns:
                    origin_concat.loc[origin_concat['file_name']==X_input_i.obs['file_name'].unique()[0],j]=list(X_input_i.obs[j].values)

    ad_embed.obs.index = origin_concat.index
    for i in origin_concat.columns:
        if i not in ad_embed.obs.columns:
            ad_embed.obs[i]=origin_concat[i]
    return ad_embed


def read_cell_embedding(X_input, save_dir,keep_celltype, keep_tissueregion, use_key="final"):
    if not os.path.exists(f"{save_dir}/ad_celltype_embedding.h5ad"):
        ad_embed = generate_ad_embed(save_dir, X_input, keep_celltype, ttype="single", use_key=use_key)
        for i in ad_embed.obs.columns:
            ad_embed.obs[i]=ad_embed.obs[i].astype(str)
        ad_embed.write_h5ad(save_dir + "/ad_celltype_embedding.h5ad")

    if not os.path.exists(f"{save_dir}/ad_tissueregion_embedding.h5ad"):
        ad_embed = generate_ad_embed(
            save_dir, X_input, keep_tissueregion, ttype="spatial", use_key=use_key
        )
        for i in ad_embed.obs.columns:
            ad_embed.obs[i]=ad_embed.obs[i].astype(str)
        ad_embed.write_h5ad(save_dir + "/ad_tissueregion_embedding.h5ad")


def transfer_annotation(X_input, save_dir, molccf_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### transfer cell type
    # Load the .pkl file
    ad_embed = ad.read_h5ad(save_dir + "/ad_celltype_embedding.h5ad")
    if 'fusemap_celltype' not in ad_embed.obs.columns:
        with open('/home/jialiulab/disk1/yichun/FuseMap/molCCF/transfer/le_gt_cell_type_main_STARmap.pkl', 'rb') as file:
            le_gt_cell_type_main_STARmap = pickle.load(file)
        
        NNmodel = NNTransfer(input_dim=64,output_dim=len(le_gt_cell_type_main_STARmap))
        NNmodel.load_state_dict(torch.load(molccf_path+"/transfer/NNtransfer_cell_type_main_STARmap.pt"))
    
        dataset = TensorDataset(torch.Tensor(ad_embed.X))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        NNmodel.to(device)
        NNmodel.eval()
        all_predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(device)
                outputs = NNmodel(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.detach().cpu().numpy())
        ad_embed.obs['fusemap_celltype']=[le_gt_cell_type_main_STARmap[i] for i in all_predictions]
        ad_embed.write_h5ad(save_dir + "/ad_celltype_embedding.h5ad")



    ### transfer tissue niche
    # Load the .pkl file
    ad_embed = ad.read_h5ad(save_dir + "/ad_tissueregion_embedding.h5ad")
    if 'fusemap_tissueregion' not in ad_embed.obs.columns:
        with open('/home/jialiulab/disk1/yichun/FuseMap/molCCF/transfer/le_gt_tissue_region_main_STARmap.pkl', 'rb') as file:
            le_gt_tissue_region_main_STARmap = pickle.load(file)
        
        NNmodel = NNTransfer(input_dim=64,output_dim=len(le_gt_tissue_region_main_STARmap))
        NNmodel.load_state_dict(torch.load(molccf_path+"/transfer/NNtransfer_tissue_region_main_STARmap.pt"))

        dataset = TensorDataset(torch.Tensor(ad_embed.X))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        NNmodel.to(device)
        NNmodel.eval()
        all_predictions = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(device)
                outputs = NNmodel(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.detach().cpu().numpy())
        ad_embed.obs['fusemap_tissueregion']=[le_gt_tissue_region_main_STARmap[i] for i in all_predictions]
        ad_embed.write_h5ad(save_dir + "/ad_tissueregion_embedding.h5ad")
