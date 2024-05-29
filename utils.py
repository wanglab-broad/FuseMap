import os
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle
import torch
import anndata as ad
import pandas as pd


def save_obj(objt, name):
    with(open(name + '.pkl', 'wb')) as f:
        pickle.dump(objt, f, pickle.HIGHEST_PROTOCOL)
        f.close()


def load_snapshot(model, snapshot_path, loc):
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epochs_run_pretrain = snapshot["EPOCHS_RUN_pretrain"]
    epochs_run_final = snapshot["EPOCHS_RUN_final"]
    print(
        f" Resuming training from snapshot at pretrain Epoch {epochs_run_pretrain}, final epoch {epochs_run_final}\n")



def save_snapshot(model, epoch_pretrain, epoch_final, snapshot_path):
    snapshot = {
        "MODEL_STATE": model.state_dict(),
        "EPOCHS_RUN_pretrain": epoch_pretrain,
        "EPOCHS_RUN_final": epoch_final,
    }
    torch.save(snapshot, snapshot_path)
    print(
        f"Pretrain Epoch {epoch_pretrain}, final Epoch {epoch_final} | Training snapshot saved at {snapshot_path}\n")


def average_embeddings(adata, category, obsm_latent):
    latent_df = pd.DataFrame(adata.obsm[obsm_latent], index=adata.obs[category])
    mean_embeddings = latent_df.groupby(level=0).mean()

    # Calculate the number of cells in each category
    num_cells = latent_df.groupby(level=0).size()

    # Create a new AnnData object with the average embeddings
    new_adata = ad.AnnData(mean_embeddings)
    new_adata.obs['size'] = num_cells

    return new_adata

def read_gene_embedding(model, all_unique_genes, save_dir, n_atlas, var_name):

    ad_gene_embedding = ad.AnnData(X=model.gene_embedding.detach().cpu().numpy().T)
    ad_gene_embedding.obs.index = all_unique_genes
    for i in range(n_atlas):
        ad_gene_embedding.obs['sample' + str(i)] = ''
        for gene in var_name[i]:
            ad_gene_embedding.obs.loc[gene, 'sample' + str(i)] = f'sample_{str(i)}'
    ad_gene_embedding.obs['type'] = ad_gene_embedding.obs[[f'sample{i}' for i in range(n_atlas)]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    ad_gene_embedding.write_h5ad(f'{save_dir}/ad_gene_embedding.h5ad')
    return


def generate_ad_embed(save_dir,X_input, ttype, use_key='final'):
    with (open(save_dir+f"/latent_embeddings_all_{ttype}_{use_key}.pkl", "rb")) as openfile:
        latent_embeddings_all = pickle.load(openfile)
    ad_list = []
    for ind, (X_input_i, latent_embeddings_all_i) in enumerate(zip(X_input, latent_embeddings_all)):
        ad_embed_1 = ad.AnnData(X=latent_embeddings_all_i)
        ad_embed_1.obs['x'] = list(X_input_i.obs['x'])
        ad_embed_1.obs['y'] = list(X_input_i.obs['y'])
        ad_embed_1.obs['name'] = list(X_input_i.obs['name'])  # f'sample{ind}'
        ad_embed_1.obs['batch'] = f'sample{ind}'
        ad_list.append(ad_embed_1)
    ad_embed = ad.concat(ad_list)
    return ad_embed


def annotation_transfer(X_input, save_dir):

    if not os.path.exists(f"{save_dir}/ad_celltype_embedding.h5ad"):
        ad_embed = generate_ad_embed(save_dir,X_input,ttype = 'single')
        ad_embed.write_h5ad(save_dir + "/ad_celltype_embedding.h5ad")

    if not os.path.exists(f"{save_dir}/ad_tissueregion_embedding.h5ad"):
        default_label_key_list_tissueregion = ['gt_tissue_region_main','gt_tissue_region_sub']
        ad_embed = generate_ad_embed(save_dir,X_input, ttype = 'spatial')
        ad_embed.write_h5ad(save_dir + "/ad_tissueregion_embedding.h5ad")

