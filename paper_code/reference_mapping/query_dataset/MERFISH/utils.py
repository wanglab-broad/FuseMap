try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle
import random
import os
import anndata as ad
import pandas as pd
try:
    import torch
except ModuleNotFoundError:
    pass
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from sklearn import preprocessing
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
import scanpy as sc
from matplotlib.patches import PathPatch
from scipy.optimize import linear_sum_assignment
import dgl

eps=1e-100





def get_filenames(directory, prefix):
    """Get filenames in 'directory' that start with 'prefix'."""
    return [f for f in os.listdir(directory) if f.startswith(prefix)]



'''
plot figures
'''
def insert_columns(array1, array2, location_indices, column_label_now, column_label_rest):
    sorted_indices = np.argsort(location_indices)
    sorted_array2 = array2[:, sorted_indices]
    sorted_rest = column_label_rest[:, sorted_indices]

    sorted_locations = np.array(location_indices)[sorted_indices]

    offset = 0
    for i, loc in enumerate(sorted_locations):
        array1 = np.insert(array1, loc + offset, sorted_array2[:, i], axis=1)
        column_label_now = np.insert(column_label_now, loc + offset, sorted_rest[:, i], axis=1)
        offset += 1

    return array1, column_label_now


def custom_annot(data, fmt_func):
    """Return formatted annotations."""
    annot = np.vectorize(fmt_func)(data)
    return annot


# Custom annotation function
def fmt(x):
    return '' if x == 0 else '{:.0f}'.format(x)


def map_diagonal_line(cross_tab_normalized):
    cost_matrix = -np.array(cross_tab_normalized)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    reordered_matrix = cost_matrix[row_indices][:, col_indices]

    row_label = cross_tab_normalized.index
    column_label = cross_tab_normalized.columns
    row_label = row_label[row_indices]
    column_label_now = np.array(column_label[col_indices]).reshape(1, -1)
    column_label_rest = np.array(column_label[list(set(np.arange(cost_matrix.shape[1])) - set(col_indices))]).reshape(1,
                                                                                                                      -1)

    rest_matrix = -cost_matrix[row_indices][:, list(set(np.arange(cost_matrix.shape[1])) - set(col_indices))]

    position_column = []
    for i in range(rest_matrix.shape[1]):
        position_column.append(np.argmax(rest_matrix[:, i]) + 1)

    location_indices = position_column
    array1 = reordered_matrix
    array2 = -rest_matrix

    result, column_label = insert_columns(reordered_matrix, -rest_matrix, position_column,
                                          column_label_now, column_label_rest)

    cross_tab_normalized = cross_tab_normalized.loc[row_label, column_label[0]]

    return cross_tab_normalized







'''
Metrics Part
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.
'''

#### general

def save_obj(objt, name):
    with(open(name + '.pkl', 'wb')) as f:
        pickle.dump(objt, f, pickle.HIGHEST_PROTOCOL)
        f.close()

def seed_all(seed_value, cuda_deterministic=True):
    print('---------------------------------- SEED ALL ---------------------------------- ')
    print(f'                           Seed Num :   {seed_value}                                ')
    print('---------------------------------- SEED ALL ---------------------------------- ')
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    dgl.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic: # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else: # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True




def add_leiden(adata_raw):
    adata = adata_raw.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.pca(adata,n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=10)
    sc.tl.umap(adata)
    sc.tl.leiden(adata,resolution=1)
    sc.pl.umap(adata,color='leiden',legend_loc='on data')
    plt.show()
    adata_raw.obs['leiden'] = adata.obs['leiden']


def batch_cosine_similarity(matrixA, matrixB, batch_size=100):
    """
    Compute the cosine similarity between two matrices in batches.

    Parameters:
    - matrixA: NumPy array of shape (N, D), where N is the number of vectors and D is the dimensionality.
    - matrixB: NumPy array of shape (M, D), where M is the number of vectors and D is the dimensionality.
    - batch_size: The number of rows from matrixA to process in each batch.

    Returns:
    - A NumPy array of shape (N, M) containing the cosine similarity between each pair of vectors.
    """
    # Normalize the rows of both matrices first to simplify the cosine similarity calculation
    normA = np.linalg.norm(matrixA, axis=1, keepdims=True)
    normB = np.linalg.norm(matrixB, axis=1, keepdims=True)
    normalizedA = matrixA / np.where(normA == 0, 1, normA)
    normalizedB = matrixB / np.where(normB == 0, 1, normB)

    # Preallocate the result matrix
    similarities = np.zeros((matrixA.shape[0], matrixB.shape[0]))

    for start_idx in range(0, matrixA.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, matrixA.shape[0])
        batch = normalizedA[start_idx:end_idx]

        # Compute the dot product between the batch and all vectors in matrixB
        sim_batch = np.dot(batch, normalizedB.T)

        # Store the results
        similarities[start_idx:end_idx, :] = sim_batch

    return similarities



def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])






####### visualize

def plot_each_group(labels, adata, x, y, min_cell=0):
    new_labels = np.array(labels)

    label_counts = Counter(new_labels)
    unique_label = []

    for i in np.unique(new_labels):  # label_counts.keys():
        if label_counts[i] > min_cell:
            unique_label.append(i)

    fig, axs = plt.subplots(int(np.ceil(len(unique_label) / 5)), 5,
                            figsize=(12, 2.5 * int(np.ceil(len(unique_label) / 5))), dpi=150)

    for ax, label, i in zip(axs.flat, unique_label, np.arange(len(unique_label))):
        ax.scatter(adata.obs[x],
                   adata.obs[y],
                   s=0.1, cmap='tab20',
                   c='lightgrey')
        ax.scatter(adata[new_labels == label, :].obs[x],
                   adata[new_labels == label, :].obs[y],
                   s=0.1, cmap='tab20',
                   c='red')
        ax.axis('off')
        ax.set_title(label)

def transfer_labels(cells_coords, cells_sample_ids, cells_labels, n_neighbors=5):
    # Separate the coordinates of cells from sample 1 and sample 2
    sample1_coords = cells_coords[cells_sample_ids == 'sample1']
    sample2_coords = cells_coords[cells_sample_ids == 'sample2']

    # Also separate the labels of cells from sample 2
    sample2_labels = cells_labels[cells_sample_ids == 'sample2']

    # Fit the NearestNeighbors model to sample 2
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(sample2_coords)

    # Find indices of nearest neighbors in sample 2 for each cell in sample 1
    distances, indices = nbrs.kneighbors(sample1_coords)

    # Prepare an empty array to hold the new labels for sample 1 cells
    new_labels = []

    # For each cell in sample 1, find the most common label among its nearest neighbors in sample 2
    for i in range(sample1_coords.shape[0]):
        # Get the labels of the nearest neighbors
        neighbor_labels = sample2_labels[indices[i]]

        # Find the most common label
        label_counts = Counter(neighbor_labels)
        most_common_label = label_counts.most_common(1)[0][0]

        # Assign the most common label to the cell in sample 1
        new_labels.append(most_common_label)

    return new_labels





####### metrics

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.
'''

from typing import Optional, Union

RandomState = Optional[Union[np.random.RandomState, int]]
from scipy.sparse.csgraph import connected_components


def mean_average_precision(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.
'''
def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def avg_silhouette_width(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_score`

    Returns
    -------
    asw
        Cell type average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    return (sklearn.metrics.silhouette_score(x, y, **kwargs).item() + 1) / 2

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def neighbor_conservation(
        x: np.ndarray, y: np.ndarray, batch: np.ndarray,
        neighbor_frac: float = 0.01, **kwargs
) -> float:
    r"""
    Neighbor conservation score

    Parameters
    ----------
    x
        Cooordinates after integration
    y
        Coordinates before integration
    b
        Batch
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    nn_cons
        Neighbor conservation score
    """
    nn_cons_per_batch = []
    for b in np.unique(batch):
        mask = batch == b
        x_, y_ = x[mask], y[mask]
        k = max(round(x.shape[0] * neighbor_frac), 1)
        nnx = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(x_.shape[0], k + 1), **kwargs
        ).fit(x_).kneighbors_graph(x_)
        nny = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(y_.shape[0], k + 1), **kwargs
        ).fit(y_).kneighbors_graph(y_)
        nnx.setdiag(0)  # Remove self
        nny.setdiag(0)  # Remove self
        n_intersection = nnx.multiply(nny).sum(axis=1).A1
        n_union = (nnx + nny).astype(bool).sum(axis=1).A1
        nn_cons_per_batch.append((n_intersection / n_union).mean())
    return np.mean(nn_cons_per_batch).item()


'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def batch_entropy_umap(embedding, batch_indices, n_neighbors=100):
    r"""
    how to use
    batch_entropies1 = batch_entropy_umap(adata_concat.obsm['X_umap'],
                                         np.array(adata_concat.obs['sample']),
                                         n_neighbors=100)
    """

    le = preprocessing.LabelEncoder()
    le.fit(batch_indices)
    batch_indices = le.transform(batch_indices)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(embedding)

    batch_entropies_all = []
    for random_seed in tqdm(range(100)):
        np.random.seed(random_seed)
        #         batch_entropies_idx=0
        for batch_label in np.unique(batch_indices):
            batch_i = np.where(batch_indices == batch_label)[0]
            idx = batch_i[np.random.permutation(batch_i.shape[0])[:n_neighbors]]
            _, indices = nn.kneighbors(embedding[idx, :])

            input_array = batch_indices[indices[:, 1:]]
            prob_array = np.zeros((n_neighbors, le.classes_.shape[0]))
            for i in range(n_neighbors):
                for j in range(le.classes_.shape[0]):
                    prob_array[i, j] = np.count_nonzero(input_array[i] == j) / n_neighbors

            prob_array += eps
            batch_entropies_all.append(-np.sum(prob_array * np.log(prob_array)))
    #         break
    #         batch_entropies_all.append(batch_entropies_idx)
    max_value = -(1 / le.classes_.shape[0]) * np.log(1 / le.classes_.shape[0]) * n_neighbors * le.classes_.shape[0]
    print(f'mean batch entropy is: {np.mean(batch_entropies_all) / max_value}')
    return [i / max_value for i in batch_entropies_all]

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def seurat_alignment_score(
        x: np.ndarray, y: np.ndarray, neighbor_frac: float = 0.01,
        n_repeats: int = 4, random_state: int = 0, **kwargs
) -> float:
    r"""
    Seurat alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=k + 1, **kwargs
        ).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
                subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(
            min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def avg_silhouette_width_batch(
        x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs
) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct).item()

'''
Metrics
Citation: Metric function code from GLUE: https://github.com/gao-lab/GLUE
Reference: Multi-omics single-cell data integration and regulatory inference with graph-linked embedding.

'''
def graph_connectivity(
        x: np.ndarray, y: np.ndarray, **kwargs
) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = ad.AnnData(X=x, dtype=x.dtype)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.value_counts(c)
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()




def map_sample(adata, coord_arg='X_umap',  type_arg='gtTaxonomyRank4'):
    map_per_sample = mean_average_precision(adata.obsm[coord_arg],
                                            np.array(adata.obs[type_arg]))
    print(f' average precision over samples is: {np.mean(map_per_sample)}')
    return map_per_sample


def ave_sw_sample_all(adata, coord_arg='X_umap', type_arg='gtTaxonomyRank4'):
    map_per_sample = avg_silhouette_width(adata.obsm[coord_arg], np.array(adata.obs[type_arg]))
    print(f'mean avg_silhouette_width is: {np.mean(map_per_sample)}')
    return map_per_sample


def nc_sample_all(adata, coord_arg='X_umap', coord_before_arg='Unintegrated',
                  batch_arg='domain'):
    map_per_sample = neighbor_conservation(adata.obsm[coord_arg], adata.obsm[coord_before_arg],
                                           np.array(adata.obs[batch_arg]), neighbor_frac=0.1)
    print(f'mean Neighbor conservation is: {np.mean(map_per_sample)}')
    return map_per_sample


def batch_entropy_sample_all(adata, coord_arg='X_umap', batch_arg='domain', ):
    map_per_sample = batch_entropy_umap(adata.obsm[coord_arg],
                                        np.array(adata.obs[batch_arg]),
                                        n_neighbors=100)
    return map_per_sample


def sas_sample_all(adata, coord_arg='X_umap',
                   batch_arg='domain',  random_seed=0):
    map_per_sample = seurat_alignment_score(adata.obsm[coord_arg],
                                            np.array(adata.obs[batch_arg]),
                                            random_state=random_seed)
    print(f' seurat_alignment_score is: {np.mean(map_per_sample)}')
    return map_per_sample


def aswb_sample_all(adata, coord_arg='X_umap',
                    batch_arg='domain', type_arg='gtTaxonomyRank4'):
    map_per_sample = avg_silhouette_width_batch(adata.obsm[coord_arg],
                                                np.array(adata.obs[batch_arg]),
                                                np.array(adata.obs[type_arg]))
    print(f' avg_silhouette_width_batch is: {np.mean(map_per_sample)}')
    return map_per_sample


def gc_sample_all(adata, coord_arg='X_umap', type_arg='gtTaxonomyRank4'):
    map_per_sample = graph_connectivity(adata.obsm[coord_arg],
                                        np.array(adata.obs[type_arg]))
    print(f' graph_connectivity is: {np.mean(map_per_sample)}')
    return map_per_sample
