import numpy as np
import scipy.sparse as sp
import scipy
from scipy.spatial import Delaunay
from scipy.sparse.csr import csr_matrix
import scanpy as sc
# from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import logging

def preprocess_raw(
    X_input, kneighbor, input_identity, use_input, n_atlas, data_pth=None
):
    """A function to preprocess raw data.
    
    Parameters
    ----------
    X_input : list
        A list of anndata objects, each representing a spatial section.
    kneighbor : list
        A list of strings, each representing the method to calculate the k-nearest neighbors.
    input_identity : list
        A list of strings, each representing the identity of the input data.
    use_input : str
        The method to use the input data.
    n_atlas : int
        The number of atlases.
    data_pth : str
        The path to save the data.

    Examples
    --------
    >>> import fusemap
    >>> import scanpy as sc
    >>> import os
    >>> preprocess_raw(
    ...     [sc.read_h5ad(f) for f in os.listdir('data') if f.endswith('.h5ad')],
    ...     ['delaunay']*len(X_input),
    ...     ['ST']*len(X_input),
    ...     'pca',
    ...     len(X_input),
    ...     'data'
    ... )
    """

    logging.info(
        "\n\n---------------------------------- Preprocess adata ----------------------------------\n"
    )
    X_input = preprocess_adata(X_input, n_atlas)

    logging.info(
        "\n\n---------------------------------- Construct graph adata ----------------------------------\n"
    )
    construct_graph(X_input, n_atlas, kneighbor, input_identity)

    logging.info(
        "\n\n---------------------------------- Process graph adata ----------------------------------\n"
    )
    preprocess_adj_sparse(X_input, n_atlas, input_identity)

    get_spatial_input(X_input, n_atlas, use_input)

    if data_pth is not None:
        for pth_i, data_i in zip(data_pth, X_input[: len(data_pth)]):
            logging.info(f"\n\nSaving processed data in {pth_i}\n")
            data_i.write_h5ad(pth_i)


def contains_only_integers(arr):
    # Check if all values are whole numbers, even if they are floats
    return np.all(arr % 1 == 0)


def preprocess_adata(X_input, n_atlas):
    for i in range(n_atlas):
        # if not "spatial_input" in X_input[i].obsm:

            ### unify genes
            X_input[i].var.index = [i.upper() for i in X_input[i].var.index]

            ### keep unique genes
            _, indices = np.unique(X_input[i].var.index, return_index=True)
            X_input[i] = X_input[i][:, indices]

            ### filter genes
            if isinstance(X_input[i].X, np.ndarray):
                if contains_only_integers(X_input[i].X):
                    X_input[i] = X_input[i][:, np.sum(X_input[i].X, axis=0) > 1]
                    X_input[i] = X_input[i][:, np.max(X_input[i].X, axis=0) > 1]
                    ### filter cells
                    X_input[i] = X_input[i][np.sum(X_input[i].X, axis=1) > 5]

                    X_input[i].layers["counts"] = X_input[i].X.copy()

                    sc.pp.normalize_total(X_input[i])  # , target_sum=1e4)
                    sc.pp.log1p(X_input[i])
                    sc.pp.scale(X_input[i], zero_center=False, max_value=10)
                    
            if scipy.sparse.issparse(X_input[i].X):
                if contains_only_integers(X_input[i].X.toarray()):
                    X_input[i] = X_input[i][:, np.sum(X_input[i].X.toarray(), axis=0) > 1]
                    X_input[i] = X_input[i][:, np.max(X_input[i].X.toarray(), axis=0) > 1]
                    ### filter cells
                    X_input[i] = X_input[i][np.sum(X_input[i].X, axis=1) > 5]

                    X_input[i].layers["counts"] = X_input[i].X.copy()

                    sc.pp.normalize_total(X_input[i])  # , target_sum=1e4)
                    sc.pp.log1p(X_input[i])
                    sc.pp.scale(X_input[i], zero_center=False, max_value=10)
            
            # sc.pp.scale(X_input[i], zero_center=False, max_value=10)


            # if isinstance(X_input[i].X, np.ndarray):
            #     X_input[i].X = csr_matrix(X_input[i].X)

            ### unify genes
            X_input[i].var.index = [i.upper() for i in X_input[i].var.index]

    return X_input


def construct_graph(adatas, n_atlas, kneighbor, input_identity):
    for i_atlas in range(n_atlas):
        if input_identity[i_atlas] == "ST":
            if not "adj_normalized" in adatas[i_atlas].obsm:
                adata = adatas[i_atlas]
                k = kneighbor[i_atlas]
                data = np.array(adata.obs[["x", "y"]])
                if k == "delaunay":
                    tri = Delaunay(data)
                    indptr, indices = tri.vertex_neighbor_vertices
                    adjacency_matrix = csr_matrix(
                        (np.ones_like(indices, dtype=np.float64), indices, indptr),
                        shape=(data.shape[0], data.shape[0]),
                    )
                if k == "delaunay3d":
                    data = np.array(adata.obs[["x", "y", "z"]])
                    tri = Delaunay(data)
                    indptr, indices = tri.vertex_neighbor_vertices
                    adjacency_matrix = csr_matrix(
                        (np.ones_like(indices, dtype=np.float64), indices, indptr),
                        shape=(data.shape[0], data.shape[0]),
                    )
                if "knn" in k:
                    if "3d" in k:
                        data = np.array(adata.obs[["x", "y", "z"]])

                    knn_k = 10
                    # nbrs = NearestNeighbors(
                    #     n_neighbors=knn_k + 1, algorithm="auto"
                    # ).fit(data)
                    # distances, indices = nbrs.kneighbors(data)

                    tree = cKDTree(data)
                    # Query the tree for the k-nearest neighbors
                    distances, indices = tree.query(data, k=knn_k + 1)

                    # Create an adjacency matrix
                    num_spots = data.shape[0]
                    adjacency_matrix = np.zeros((num_spots, num_spots))

                    for i in range(num_spots):
                        # indices[i, 1:] to exclude the point itself (the first nearest neighbor)
                        for j in indices[i, 1:]:
                            adjacency_matrix[i, j] = 1
                            adjacency_matrix[
                                j, i
                            ] = 1  # Because it's an undirected graph

                adata.obsm["adj"] = adjacency_matrix


def preprocess_adj_sparse(adatas, n_atlas, input_identity):
    for i in range(n_atlas):
        if input_identity[i] == "ST":
            if not "adj_normalized" in adatas[i].obsm:
                adata = adatas[i]
                adj = sp.coo_matrix(adata.obsm["adj"])
                adj_ = adj + sp.eye(adj.shape[0])
                rowsum = np.array(adj_.sum(1))
                degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
                adj_normalized = (
                    adj_.dot(degree_mat_inv_sqrt)
                    .transpose()
                    .dot(degree_mat_inv_sqrt)
                    .tocoo()
                )
                adata.obsm[
                    "adj_normalized"
                ] = adj_normalized  # sparse_mx_to_torch_sparse_tensor(adj_normalized)
                adata.obsm["adj_normalized"] = adata.obsm["adj_normalized"].tocsr()


def get_spatial_input(adatas, n_atlas, use_input):
    for i_atlas in range(n_atlas):
        adata = adatas[i_atlas]
        # if use_input == "pca":
        #     adata.obsm["spatial_input"] = adata.obsm["X_pca"]
        # if use_input == "raw":
        #     adata.obsm["spatial_input"] = adata.layers["counts"]
        if use_input == "norm":
            if isinstance(adata.X, np.ndarray):
                adata.obsm["spatial_input"]= csr_matrix(adata.X)
            else:
                adata.obsm["spatial_input"] = adata.X
        else:
            raise ValueError("use_input not implemented")


def get_unique_gene_indices(gene_list):
    unique_genes, indices = np.unique(gene_list, return_index=True)
    return indices


def get_allunique_gene_names(*sample_gene_lists):
    unique_genes = set()
    for gene_list in sample_gene_lists:
        unique_genes.update(gene_list)
    return unique_genes
