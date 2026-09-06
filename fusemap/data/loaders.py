import torch
import scipy.sparse as sp
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import logging
import os

def get_feature_sparse(device, feature):
    return feature.copy()  # .to(device)


def construct_mask(n_atlas, spatial_dataset_list, g_all):
    """
    Construct mask for training and validation

    Parameters
    ----------
    n_atlas : int
        Number of atlases
    spatial_dataset_list : list
        List of spatial datasets
    g_all : list
        List of graphs

    Returns
    -------
    train_mask : list
        List of training masks
    val_mask : list
        List of validation masks

    Examples
    --------
    >>> n_atlas = 2
    >>> spatial_dataset_list = [CustomGraphDataset(i, j, ModelType.use_input) for i, j in zip(g_all, adatas)]
    >>> g_all = [dgl.graph((adj_coo.row, adj_coo.col)) for adj_coo in adj_all]
    >>> train_mask, val_mask = construct_mask(n_atlas, spatial_dataset_list, g_all)

    """
    train_pct = 0.85
    # np.random.seed(0)
    num_train = [int(len(i) * train_pct) for i in spatial_dataset_list]
    nodes_order = [np.random.permutation(g_i.number_of_nodes()) for g_i in g_all]
    train_id = [
        nodes_order_i[:num_train_i]
        for nodes_order_i, num_train_i in zip(nodes_order, num_train)
    ]
    # val_mask=[nodes_order_i[num_train_i:] for nodes_order_i,num_train_i in zip(nodes_order,num_train)]
    train_mask = [
        torch.zeros(
            len(i),
        )
        for i in spatial_dataset_list
    ]
    for i in range(n_atlas):
        train_mask[i][train_id[i]] = 1
        train_mask[i] = train_mask[i].bool()
    val_mask = [~i for i in train_mask]
    return train_mask, val_mask


def construct_data(n_atlas, adatas, input_identity, model):
    adj_all = []
    g_all = []
    for i in range(n_atlas):
        adata = adatas[i]
        if input_identity[i] == "ST":
            adj_coo = adata.obsm["adj_normalized"].tocoo()
            # adj_all.append(adj_coo.todense())
            adj_all.append(adata.obsm["adj_normalized"])
        else:
            adj_raw = model.scrna_seq_adj["atlas" + str(i)]()  # .weight
            adj_coo = sp.coo_matrix(adj_raw.detach().cpu().numpy())
            adj_all.append(adj_raw)
        g_all.append(dgl.graph((adj_coo.row, adj_coo.col)))
    return adj_all, g_all


class CustomGraphDataset(Dataset):
    def __init__(self, g, adata, useinput):
        self.g = g
        self.n_nodes = g.number_of_nodes()

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, idx):
        # return  X[idx], batch_idx[idx], library_size[idx], x_input[idx], idx
        return idx


### ----------------------------------------------------------------------
### Scalable parallel batch-preparation pipeline.
###
### The per-batch heavy CPU work (DGL block sampling, sorted row/col index
### extraction, scipy feature slicing and adjacency-block densification) is
### executed inside torch DataLoader worker processes so that batch N+1 is
### prepared while batch N trains on the GPU. Only per-batch blocks are ever
### densified: the full adjacency stays sparse and features are only cached
### densely when small (see DENSE_FEATURE_CACHE_MAX_BYTES), so the pipeline
### scales to millions of cells.
### ----------------------------------------------------------------------

DENSE_FEATURE_CACHE_MAX_BYTES = 2 * 1024**3  # 2 GB


def _build_feature_cache(feature, atlas_idx):
    """Adaptive feature cache with an explicit scalability gate.

    If the dense float32 copy of the feature matrix would occupy less than
    DENSE_FEATURE_CACHE_MAX_BYTES of CPU memory, keep a dense torch tensor so
    per-batch slicing becomes fast torch row indexing. Otherwise keep the
    scipy CSR matrix and slice sparsely per batch (scalable path). Features
    are never placed on the GPU.
    """
    if isinstance(feature, np.ndarray):
        logging.info(
            f"[FuseMap loader] atlas {atlas_idx}: features already dense "
            f"({feature.shape[0]} x {feature.shape[1]}); using dense torch cache "
            f"with torch row indexing."
        )
        return torch.from_numpy(np.ascontiguousarray(feature, dtype=np.float32))
    n_cells, n_genes = feature.shape
    dense_bytes = int(n_cells) * int(n_genes) * 4
    if dense_bytes < DENSE_FEATURE_CACHE_MAX_BYTES:
        logging.info(
            f"[FuseMap loader] atlas {atlas_idx}: dense CPU feature cache ENABLED "
            f"({n_cells} x {n_genes} = {dense_bytes / 1024**3:.3f} GB < "
            f"{DENSE_FEATURE_CACHE_MAX_BYTES / 1024**3:.1f} GB limit); per-batch "
            f"slicing uses torch row indexing."
        )
        return torch.from_numpy(
            np.ascontiguousarray(feature.toarray(), dtype=np.float32)
        )
    logging.info(
        f"[FuseMap loader] atlas {atlas_idx}: dense CPU feature cache DISABLED "
        f"({n_cells} x {n_genes} = {dense_bytes / 1024**3:.3f} GB >= "
        f"{DENSE_FEATURE_CACHE_MAX_BYTES / 1024**3:.1f} GB limit); keeping "
        f"scalable scipy CSR slicing."
    )
    return feature.tocsr()


class _MultiAtlasIndexBatchSampler(Sampler):
    """Yields one dict {atlas_id: seed-node index tensor} per training step.

    Reproduces EXACTLY the multi-atlas cycling semantics of the original
    CustomGraphDataLoader.__iter__: the largest atlas drives the iteration
    length while every other atlas's index DataLoader is wrapped in
    itertools.cycle. The per-atlas index DataLoaders (with their original
    shuffle semantics) are iterated here, in the main process, so batch
    composition is unchanged.
    """

    def __init__(self, index_dataloaders, max_value_index, n_atlas):
        self.index_dataloaders = index_dataloaders
        self.max_value_index = max_value_index
        self.n_atlas = n_atlas

    def __iter__(self):
        dataloader_iter_before = {}
        dataloader_iter_after = {}
        for i in np.arange(0, self.max_value_index):
            dataloader_iter_before[i] = itertools.cycle(self.index_dataloaders[i])
        for i in np.arange(self.max_value_index + 1, self.n_atlas):
            dataloader_iter_after[i] = itertools.cycle(self.index_dataloaders[i])

        for indices_max in self.index_dataloaders[self.max_value_index]:
            indices_all = {}
            for i in np.arange(0, self.max_value_index):
                indices_all[i] = next(dataloader_iter_before[i])
            indices_all[self.max_value_index] = indices_max
            for i in np.arange(self.max_value_index + 1, self.n_atlas):
                indices_all[i] = next(dataloader_iter_after[i])
            yield indices_all

    def __len__(self):
        return max([len(i) for i in self.index_dataloaders])


class _PreparedBatchDataset(Dataset):
    """Map-style dataset whose "index" is a dict {atlas_id: seed-node tensor}.

    __getitem__ performs all heavy per-batch CPU work and returns ready CPU
    torch tensors (float32 features / adjacency blocks, int64 sorted row and
    col indices). It runs inside DataLoader worker processes.

    Learnable (scrna) adjacencies come from the model and are therefore NOT
    computed here; workers return adj_block=None for those atlases and the
    main process slices the model adjacency in get_data (as before).
    """

    def __init__(self, graphs, sampler, feature_cache_all, adj_st_all, input_identity):
        self.graphs = graphs
        self.sampler = sampler
        self.feature_cache_all = feature_cache_all
        self.adj_st_all = adj_st_all
        self.input_identity = input_identity

    def __len__(self):
        return max([g.number_of_nodes() for g in self.graphs])

    def __getitem__(self, indices_all):
        out = {}
        for i, indices in indices_all.items():
            sample_result = self.sampler.sample_blocks(self.graphs[i], indices)
            row_index = torch.sort(sample_result[0].flatten())[0]
            col_index = torch.sort(sample_result[1].flatten())[0]

            feature = self.feature_cache_all[i]
            if isinstance(feature, torch.Tensor):
                batch_feature = feature[row_index]
            else:
                batch_feature = torch.from_numpy(
                    np.ascontiguousarray(
                        feature[row_index.numpy(), :].toarray(), dtype=np.float32
                    )
                )

            adj = self.adj_st_all[i]
            if adj is not None:
                adj_block = torch.from_numpy(
                    np.ascontiguousarray(
                        adj[row_index.numpy(), :]
                        .tocsc()[:, col_index.numpy()]
                        .todense(),
                        dtype=np.float32,
                    )
                )
            else:
                adj_block = None

            out[i] = {
                "single": indices,
                "row_index": row_index,
                "col_index": col_index,
                "feature": batch_feature,
                "adj_block": adj_block,
            }
        return out


class CustomGraphDataLoader:
    """Multi-atlas graph dataloader.

    When feature_all / adj_all / input_identity are provided, batches are
    fully prepared (block sampling + slicing + densification of per-batch
    blocks) inside torch DataLoader worker processes with prefetching and
    pinned memory, and each yielded per-atlas dict contains ready CPU tensors
    ("row_index", "col_index", "feature", "adj_block") in addition to
    "single". Otherwise the original single-process behavior is kept and each
    per-atlas dict contains "single" and "spatial" (legacy mode).
    """

    def __init__(
        self,
        dataset_all,
        sampler,
        batch_size,
        shuffle,
        n_atlas,
        drop_last,
        feature_all=None,
        adj_all=None,
        input_identity=None,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=None,
        persistent_workers=True,
    ):
        self.dataset_all = dataset_all
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_atlas = n_atlas

        self.dataloader = []
        for i in range(n_atlas):
            self.dataloader.append(
                DataLoader(
                    self.dataset_all[i],
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                )
            )
        cell_num = [len(i) for i in self.dataset_all]
        self.max_value_index = np.argmax(cell_num)

        self.prepare_in_workers = (
            feature_all is not None
            and adj_all is not None
            and input_identity is not None
        )
        if self.prepare_in_workers:
            num_workers = int(
                os.environ.get("FUSEMAP_LOADER_WORKERS", num_workers)
            )
            num_workers = max(0, min(num_workers, os.cpu_count() or 1))
            if pin_memory is None:
                pin_memory = torch.cuda.is_available()

            # Only static, CPU-resident, sparse objects are handed to workers:
            # learnable (scrna) adjacencies live in the model and stay in the
            # main process.
            adj_st_all = [
                adj_all[i].tocsr() if input_identity[i] == "ST" else None
                for i in range(n_atlas)
            ]
            feature_cache_all = [
                _build_feature_cache(feature_all[i], i) for i in range(n_atlas)
            ]
            self._prep_dataset = _PreparedBatchDataset(
                [d.g for d in dataset_all],
                sampler,
                feature_cache_all,
                adj_st_all,
                list(input_identity),
            )
            self._batch_sampler = _MultiAtlasIndexBatchSampler(
                self.dataloader, self.max_value_index, n_atlas
            )
            prep_loader_kwargs = dict(
                batch_size=None,
                sampler=self._batch_sampler,
                pin_memory=pin_memory,
                # A dedicated generator so that creating worker iterators does
                # not consume the global torch RNG stream (keeps the shuffled
                # batch sequence identical to the original implementation for
                # a given global seed).
                generator=torch.Generator(),
            )
            if num_workers > 0:
                prep_loader_kwargs.update(
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                )
            self._prep_loader = DataLoader(self._prep_dataset, **prep_loader_kwargs)
            logging.info(
                f"[FuseMap loader] parallel batch preparation ENABLED: "
                f"num_workers={num_workers}, "
                f"prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}, "
                f"pin_memory={pin_memory}, "
                f"persistent_workers={persistent_workers if num_workers > 0 else 'n/a'}."
            )
        else:
            self._prep_loader = None

    def __iter__(self):
        if self.prepare_in_workers:
            return iter(self._prep_loader)
        return self._legacy_iter()

    def _legacy_iter(self):
        dataloader_iter_before = {}
        dataloader_iter_after = {}
        for i in np.arange(0, self.max_value_index):
            dataloader_iter_before[i] = itertools.cycle(self.dataloader[i])
        for i in np.arange(self.max_value_index + 1, self.n_atlas):
            dataloader_iter_after[i] = itertools.cycle(self.dataloader[i])

        for indices_max in self.dataloader[self.max_value_index]:
            blocks = {}
            for i in np.arange(0, self.max_value_index):
                indices_i = next(dataloader_iter_before[i])
                blocks[i] = {
                    "single": indices_i,
                    "spatial": self.sampler.sample_blocks(
                        self.dataset_all[i].g, indices_i
                    ),
                }
            blocks[self.max_value_index] = {
                "single": indices_max,
                "spatial": self.sampler.sample_blocks(
                    self.dataset_all[self.max_value_index].g, indices_max
                ),
            }
            for i in np.arange(self.max_value_index + 1, self.n_atlas):
                indices_i = next(dataloader_iter_after[i])
                blocks[i] = {
                    "single": indices_i,
                    "spatial": self.sampler.sample_blocks(
                        self.dataset_all[i].g, indices_i
                    ),
                }
            yield blocks

    def __len__(self):
        return max([len(i) for i in self.dataloader])
        # return 100


class MapPretrainDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class MapPretrainDataLoader:
    def __init__(self, dataset_all, batch_size, shuffle, n_atlas):
        self.dataset_all = dataset_all
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_atlas = n_atlas
        self.dataloader = []
        for i in range(n_atlas):
            self.dataloader.append(
                DataLoader(
                    self.dataset_all[i],
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=False,
                )
            )
        cell_num = [len(i) for i in self.dataset_all]
        self.max_value_index = np.argmax(cell_num)

    def __iter__(self):
        dataloader_iter_before = {}
        dataloader_iter_after = {}
        for i in np.arange(0, self.max_value_index):
            dataloader_iter_before[i] = itertools.cycle(self.dataloader[i])
        for i in np.arange(self.max_value_index + 1, self.n_atlas):
            dataloader_iter_after[i] = itertools.cycle(self.dataloader[i])

        for atlasdata_max in self.dataloader[self.max_value_index]:
            blocks = {}
            for i in np.arange(0, self.max_value_index):
                atlasdata_i = next(dataloader_iter_before[i])
                blocks[i] = atlasdata_i

            blocks[self.max_value_index] = atlasdata_max

            for i in np.arange(self.max_value_index + 1, self.n_atlas):
                atlasdata_i = next(dataloader_iter_after[i])
                blocks[i] = atlasdata_i
            yield blocks

    def __len__(self):
        return max([len(i) for i in self.dataloader])
