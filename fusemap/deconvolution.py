# this script performs cell deconvolution based on Fusemap on starmap and stereomap
import scanpy as sc
import torch
from torch import optim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from anndata import AnnData
from tqdm import tqdm
from argparse import ArgumentParser
import json
import tangram as tg
from time import time

# Astrocytes: The five “Astr” types in list 2 were grouped to represent astrocytes.
# Dentate gyrus granule neurons: “GN DG” was chosen as the counterpart.
# Inhibitory and excitatory neurons: The various “EX…” and “IN…” subtypes in list 2 are grouped to cover several of the broad classes in list 1 (for example, “Di- and mesencephalon inhibitory neurons” and “Telencephalon inhibitory interneurons” are both mapped to subsets of “IN…” types).
# Missing matches: Some cell types from list 1 (for example, “Cerebellum neurons”, “Choroid epithelial cells”, “Enteric glia”, etc.) do not have clear counterparts in list 2 and are left with empty mappings.
# Extra types in list 2: For example, “Erythrocyte” in list 2 was not used because there was no matching entry in list 1.
cell_type_mapping_starmap_stereomap = {
    "Astrocytes": ["Astr1", "Astr2", "Astr3", "Astr4", "Astr5"],
    "Cerebellum neurons": [],
    "Cholinergic and monoaminergic neurons": ["DA neuron"],
    "Choroid epithelial cells": [],
    "Dentate gyrus granule neurons": ["GN DG"],
    "Dentate gyrus radial glia-like cells": [],
    "Di- and mesencephalon excitatory neurons": ["EX", "EX Mb", "EX thalamus"],
    "Di- and mesencephalon inhibitory neurons": [
        "IN Pvalb+",
        "IN Pvalb+Gad1+",
        "IN Sst+",
        "IN Vip+",
        "IN thalamus",
    ],
    "Enteric glia": [],
    "Ependymal cells": ["Ependymal"],
    "Glutamatergic neuroblasts": [],
    "Hindbrain neurons": [],
    "Microglia": ["Microglia"],
    "Non-glutamatergic neuroblasts": [],
    "Olfactory ensheathing cells": [],
    "Olfactory inhibitory neurons": [],
    "Oligodendrocyte precursor cells": ["OPC"],
    "Oligodendrocytes": ["Olig"],
    "Peptidergic neurons": [],
    "Pericytes": [],
    "Perivascular macrophages": [],
    "Spinal cord excitatory neurons": ["EX CA", "EX L2/3", "EX L4", "EX L5/6", "EX L6"],
    "Spinal cord inhibitory neurons": [
        "IN Pvalb+",
        "IN Pvalb+Gad1+",
        "IN Sst+",
        "IN Vip+",
        "IN thalamus",
    ],
    "Subcommissural organ hypendymal cells": [],
    "Subventricular zone radial glia-like cells": [],
    "Telencephalon inhibitory interneurons": [
        "IN Pvalb+",
        "IN Pvalb+Gad1+",
        "IN Sst+",
        "IN Vip+",
    ],
    "Telencephalon projecting excitatory neurons": [
        "EX CA",
        "EX L2/3",
        "EX L4",
        "EX L5/6",
        "EX L6",
    ],
    "Telencephalon projecting inhibitory neurons": ["IN thalamus"],
    "Vascular and leptomeningeal cells": ["Meninge"],
    "Vascular endothelial cells": ["Endothelium"],
    "Vascular smooth muscle cells": ["Smooth muscle cells"],
    "nan": ["Unknown"],
}


def evaluate_spot_topk(M_final, spot_id: int, cell_type_mapping: dict, img_cate: list, ground_truth: list, k: int = 3):
    # Get the top k predicted indices (sorted in descending order by score)
    topk_indices = np.argsort(M_final[:, spot_id])[::-1][:k]
    # Map these indices to their corresponding cell types
    topk_predicted_types = [img_cate[i] for i in topk_indices]
    
    # Combine the mappings for all top k predicted cell types
    union_mapping = set()
    for pred in topk_predicted_types:
        # Use .get() to avoid KeyError if the prediction is not in the mapping
        union_mapping.update(cell_type_mapping.get(pred, []))
    
    # Get the ground truth cell type for the spot
    cell_type_main = ground_truth[spot_id]
    
    # Output the results
    # print("Top-k predicted cell types:", topk_predicted_types)
    # print("Union of corresponding ground truth mappings:", list(union_mapping))
    # print("Ground truth cell type:", cell_type_main)
    # print("Top-k correct:", cell_type_main in union_mapping)
    
    if len(union_mapping) == 0 or cell_type_main == 'Unknown' or cell_type_main == 'Erythrocyte':
        return None
    
    return cell_type_main in union_mapping
    
def evaluate_topk_accuracy(M_final, cell_type_mapping, img_cate, ground_truth, n_spots, k: int = 1):
    n_all, n_cor = 0, 0
    for spot_id in tqdm(range(n_spots)):
        res = evaluate_spot_topk(M_final, spot_id, cell_type_mapping, img_cate, ground_truth, k=k)
        if res is None:
            continue

        n_all += 1
        if res:
            n_cor += 1

    acc = n_cor / n_all
    print(f"Top-{k} accuracy: {acc:.1%}, {n_cor}/{n_all}")
    
    return acc
def get_cell_spot_embedding(ad_cell_embd: AnnData, cell_or_spot_column: str, cell_type_column: str = 'gtTaxonomyRank4'):
    """
    Extract embeddings and cell type information for cells and spots from an AnnData object.

    Parameters:
    - ad_cell_embd: AnnData
        The AnnData object containing embeddings and metadata for cells and spots.
    - cell_or_spot_column: str
        The column name in `ad_cell_embd.obs` that indicates whether a row corresponds to a 'cell' or 'spot'.
    - cell_type_column: str, optional (default: 'gtTaxonomyRank4')
        The column name in `ad_cell_embd.obs` that contains cell type annotations for 'cell' rows.

    Returns:
    - cell_embd: np.array
        The embedding matrix for rows corresponding to 'cell'.
    - spot_embd: np.array
        The embedding matrix for rows corresponding to 'spot'.
    - cell_type: pd.Series
        The cell type annotations for rows corresponding to 'cell'.
    """
    # Extract the embeddings for rows labeled as 'cell' in the specified column
    cell_embd = ad_cell_embd.X[ad_cell_embd.obs[cell_or_spot_column] == 'cell']
    
    # Extract the embeddings for rows labeled as 'spot' in the specified column
    spot_embd = ad_cell_embd.X[ad_cell_embd.obs[cell_or_spot_column] == 'spot']
    
    # Extract the cell type annotations for rows labeled as 'cell' in the specified column
    cell_type = ad_cell_embd.obs[cell_type_column][ad_cell_embd.obs[cell_or_spot_column] == 'cell']
    
    return cell_embd, spot_embd, cell_type

def get_representative_embeddings(Z_cells, cell_labels, n_types=None, n_prototypes=3, method="kmeans"):
    """
    Obtain representative embeddings (prototypes) for each cell type using clustering.

    Parameters:
    - Z_cells: np.array, shape [n_cells, C], the embedding matrix of all cells.
    - cell_labels: np.array, shape [n_cells], the type labels for each cell.
    - n_types: int, the total number of cell types.
    - n_prototypes: int, the number of prototypes (representative embeddings) per cell type.
    - method: str, clustering method, supports "kmeans".

    Returns:
    - Z_representative: np.array, shape [n_types, n_prototypes, C],
                        where each element Z_representative[k, :, :] corresponds to
                        the prototypes of cell type `k`.
    """
    C = Z_cells.shape[1]
    n_types = np.max(cell_labels) + 1 if n_types is None else n_types

    Z_representative = np.zeros((n_types, n_prototypes, C))  # Initialize the output array

    for k in range(n_types):
        # Get indices of cells belonging to the current cell type
        indices = np.where(cell_labels == k)[0]
        embeddings = Z_cells[indices, :]  # Extract embeddings for this cell type

        if embeddings.shape[0] == 0:
            # If no cells belong to this type, fill with zeros
            Z_representative[k, :, :] = np.zeros((n_prototypes, C))
            continue

        if method == "kmeans":
            # Use K-Means clustering to find prototypes
            kmeans = KMeans(n_clusters=n_prototypes, random_state=0, n_init=10)
            kmeans.fit(embeddings)
            Z_representative[k, :, :] = kmeans.cluster_centers_  # Shape: [n_prototypes, C]
        else:
            raise ValueError("Unsupported clustering method: choose 'kmeans'")

    return Z_representative

# Define the sparse regularization term
def sparse_regularization(M):
    # Count the number of non-zero elements along columns (L0 approximation)
    # l0_norm = torch.sum(M != 0, dim=0)
    # return torch.sum(l0_norm) / n_spots
    
    return -torch.mean(M * torch.log(M + 1e-8))
  
def cosine_loss(pred, target):
    # Normalize the vectors
    pred_norm = pred / torch.norm(pred, dim=1, keepdim=True)
    target_norm = target / torch.norm(target, dim=1, keepdim=True)
    # Calculate cosine similarity for each row
    cos_sim = torch.sum(pred_norm * target_norm, dim=1)
    # Convert similarity to distance (1 - similarity) and take mean
    return torch.mean(1 - cos_sim)

def optimize_cell_spot_assignment(Z_prototypes, Z_spots, lambda_reg=10, lr=0.03, num_epochs=2000, device='cpu'):
    """
    Optimize the cell-spot assignment matrix with sparse regularization.

    Parameters:
    - Z_prototypes: torch.Tensor
        The embedding matrix of cell prototypes, shape (n_cells, embedding_dim).
    - Z_spots: torch.Tensor
        The embedding matrix of spots, shape (n_spots, embedding_dim).
    - n_types: int
        The number of cell types.
    - n_prototypes: int
        The number of prototypes per cell type.
    - lambda_reg: float, optional (default: 10)
        The regularization coefficient for sparsity.
    - lr: float, optional (default: 0.03)
        Learning rate for the optimizer.
    - num_epochs: int, optional (default: 2000)
        Number of optimization iterations.
    - device: str, optional (default: 'cpu')
        The device to use ('cpu' or 'cuda').

    Returns:
    - M_final: np.array
        The final cell-spot assignment matrix, shape (n_types, n_spots).
    """
    # n_cells, n_spots = Z_prototypes.shape[0], Z_spots.shape[0]
    n_types, n_prototypes, C = Z_prototypes.shape
    n_spots = Z_spots.shape[0]
    
    Z_prototypes = Z_prototypes.view(n_types * n_prototypes, C)

    # Initialize the cell-spot assignment matrix M randomly
    M_raw = torch.rand((n_types * n_prototypes, n_spots), requires_grad=True, device=device)

    # Define the optimizer
    optimizer = optim.Adam([M_raw], lr=lr)

    # Define the loss function
    # loss_fn = nn.MSELoss()

    # Optimization loop
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        
        # Ensure that the columns of M sum to 1
        M = torch.softmax(M_raw, dim=0)
        
        # Reconstruction loss
        reconstruction_loss = cosine_loss(torch.matmul(M.T, Z_prototypes), Z_spots)
        
        # Sparse regularization loss
        reg_loss = sparse_regularization(M)
        
        # Total loss
        total_loss = reconstruction_loss + lambda_reg * reg_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Print loss for monitoring
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                  f"Reconstruction Loss: {reconstruction_loss.item():.4f}, "
                  f"Regularization Loss: {reg_loss.item():.4f}")

    # Final optimized assignment matrix
    M_opt = M.cpu().detach().numpy()
    # n_types, n_prototypes, n_spots = M_opt.shape
    # Aggregate by summing over prototypes for each cell type
    M_final = M_opt.reshape(n_types, n_prototypes, n_spots).sum(1)

    print("Optimization completed!")
    return M_final

def get_args():
    parser = ArgumentParser(description="Process number of prototypes and regularization parameter.")

    parser.add_argument(
        "--n_prototypes",
        type=int,
        default=5,
        help="Number of prototypes (default: 5)"
    )

    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Regularization parameter lambda (default: 0.1)"
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000,
    )
    
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Use baseline method (implementing Tangram)'
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Load the cell embedding data
    start_time = time()
    args = get_args()
    device = torch.device('mps')
    torch.manual_seed(0)
    
    ad_cell_embd = sc.read('/Users/mingzeyuan/Workspace/fusemap_deconvolution/raw_data/ad_celltype_embedding.h5ad')
    ad_sc = sc.read_h5ad('/Users/mingzeyuan/Workspace/fusemap_deconvolution/raw_data/starmap.h5ad')
    ad_sp = sc.read_h5ad('/Users/mingzeyuan/Workspace/fusemap_deconvolution/raw_data/stereoseq_mousebrain.h5ad')
    # Create a new column 'cell_or_spot' with a default value (e.g., 'unknown')
    
    if not args.baseline:
        ad_cell_embd.obs['cell_or_spot'] = 'unknown'
        # Assign values conditionally based on the 'batch' column
        ad_cell_embd.obs.loc[ad_cell_embd.obs['batch'] == 'sample0', 'cell_or_spot'] = 'cell'
        ad_cell_embd.obs.loc[ad_cell_embd.obs['batch'] == 'sample1', 'cell_or_spot'] = 'spot'

        # Extract embeddings and cell type information
        cell_embd, spot_embd, cell_type = get_cell_spot_embedding(ad_cell_embd, cell_or_spot_column='cell_or_spot', cell_type_column='gtTaxonomyRank4')
        cell_label = pd.Categorical(cell_type).codes # Cell types for merscope

        # Obtain representative embeddings for each cell type
        Z_representative = get_representative_embeddings(cell_embd, cell_label, n_prototypes=args.n_prototypes, method="kmeans")

        # Convert numpy arrays to torch tensors
        Z_prototypes = torch.tensor(Z_representative, dtype=torch.float32, device=device)
        Z_spots = torch.tensor(spot_embd, dtype=torch.float32, device=device)
    
    else:
        tg.pp_adatas(ad_sc, ad_sp, genes=None)
        training_genes = ad_sc.uns['training_genes']
        Z_cells = np.array(ad_sc[:, training_genes].X.toarray(), dtype="float32",)
        Z_spots = np.array(ad_sp[:, training_genes].X.toarray(), dtype="float32",)
        cell_types = pd.Categorical(ad_sc.obs['gtTaxonomyRank4']).codes # Cell types for merscope
        
        Z_prototypes = get_representative_embeddings(Z_cells, cell_types, n_prototypes=args.n_prototypes).astype(np.float32)
        Z_prototypes = torch.tensor(Z_prototypes).to(device)
        Z_spots = torch.tensor(Z_spots).to(device)
        
    # Optimize the cell-spot assignment matrix
    M_final = optimize_cell_spot_assignment(Z_prototypes, Z_spots, lambda_reg=args.lambda_reg, lr=0.02, num_epochs=args.n_epochs, device=device)

    # Save the final assignment matrix
    suffix = f'{args.n_prototypes}_{args.lambda_reg}' if not args.baseline else f'baseline_{args.n_prototypes}_{args.lambda_reg}'
    np.save(f'/Users/mingzeyuan/Workspace/fusemap_deconvolution/result/cell_spot_assignment_{suffix}.npy', M_final)
    
    img_cate = pd.Categorical(ad_cell_embd.obs[ad_cell_embd.obs['batch'] == 'sample0']['gtTaxonomyRank4']).categories.tolist()
    
    ground_truth = ad_sp.obs['gt_cell_type_main'].tolist()
    
    res = {}
    for k in [1, 2, 3, 4, 5]:
        res[k] = evaluate_topk_accuracy(M_final, cell_type_mapping_starmap_stereomap, img_cate, ground_truth, ad_sp.X.shape[0], k=k)
        
    res['time'] = time() - start_time
        
    with open(f'/Users/mingzeyuan/Workspace/fusemap_deconvolution/result/accuracy_{suffix}.json', 'w') as f:
        json.dump(res, f, indent=4)
