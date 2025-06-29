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
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split
from torch import optim, nn
import sklearn
import numpy as np
import pandas as pd
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

def NNTransferTrain(model, criterion, optimizer, train_loader,val_loader, device, 
                    save_pth=None, epochs=200):
    eval_accuracy_mini=0#np.inf
    patience_count=0
    for epoch in range(epochs):
        model.train()
        loss_all=0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_all+=loss.item()
        eval_loss, eval_accuracy = NNTransferEvaluate(model, val_loader, criterion, device)
        if eval_accuracy_mini<eval_accuracy:
            eval_accuracy_mini=eval_accuracy
#             torch.save(model.state_dict(), save_pth)
            # print(f"Epoch {epoch}/{epochs} - Train Loss: {loss_all / len(train_loader)}, Accuracy: {eval_accuracy}")
            patience_count=0
        else:
            patience_count+=1
        if patience_count>10:
            p=0
            # print(f"Epoch {epoch}/{epochs} - early stopping due to patience count")
            break
            
def NNTransferEvaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return total_loss/len(dataloader), accuracy

def NNTransferPredictWithUncertainty(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_uncertainties = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.max(outputs, dim=1)[0]
            uncertainty = 1 - confidence
            all_predictions.extend(predicted.detach().cpu().numpy())
            all_uncertainties.extend(uncertainty.detach().cpu().numpy())

    return np.vstack(all_predictions), np.vstack(all_uncertainties)


def transfer_celltype(ad_cell_subset, label_key, cell_emb_sample, assign_key = 'predicted_celltype'):
    sample1_embeddings = ad_cell_subset.X
    sample1_labels = list(ad_cell_subset.obs[label_key])

    le = preprocessing.LabelEncoder()
    le.fit(sample1_labels)

    sample1_labels = le.transform(sample1_labels)
    sample1_labels = sample1_labels.astype('str').astype('int')

    dataset1 = TensorDataset(torch.Tensor(sample1_embeddings), torch.Tensor(sample1_labels).long())
    train_size = int(0.8 * len(dataset1))  # Use 80% of the data for training
    val_size = len(dataset1) - train_size
    train_dataset, val_dataset = random_split(dataset1, [train_size, val_size])

    val_size = int(0.5 * len(val_dataset))  # Use 10% of the data for val and 10% for testing
    test_size = len(val_dataset) - val_size
    val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weight = torch.Tensor(sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                classes=np.unique(sample1_labels),
                                                                                y=sample1_labels))
    model = NNTransfer(input_dim=sample1_embeddings.shape[1],
                    output_dim=len(np.unique(sample1_labels)))
    model.to(device)  # Move the model to GPU if available
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NNTransferTrain(model, criterion, optimizer, train_loader, val_loader, device,epochs=50)

    sample2_embeddings = cell_emb_sample.X
    dataset2 = TensorDataset(torch.Tensor(sample2_embeddings))
    dataloader2 = DataLoader(dataset2, batch_size=256, shuffle=False)
    sample2_predictions, sample2_uncertainty = NNTransferPredictWithUncertainty(model, dataloader2, device)
    sample2_predictions = le.inverse_transform(sample2_predictions)

    cell_emb_sample.obs[assign_key] = sample2_predictions
    return cell_emb_sample

