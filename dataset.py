import torch
import scipy.sparse as sp
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader, DataLoader, TensorDataset
import itertools


def get_feature_sparse(device, feature):
    return feature.copy()  #.to(device)


def construct_mask(n_atlas, spatial_dataset_list, g_all):
    train_pct=0.85
    # np.random.seed(0)
    num_train= [int(len(i)*train_pct) for i in spatial_dataset_list]
    nodes_order= [np.random.permutation(g_i.number_of_nodes()) for g_i in g_all]
    train_id=[nodes_order_i[:num_train_i] for nodes_order_i,num_train_i in zip(nodes_order,num_train)]
    # val_mask=[nodes_order_i[num_train_i:] for nodes_order_i,num_train_i in zip(nodes_order,num_train)]
    train_mask=[torch.zeros(len(i),) for i in spatial_dataset_list]
    for i in range(n_atlas):
        train_mask[i][train_id[i]]=1
        train_mask[i] = train_mask[i].bool()
    val_mask = [~i for i in train_mask]
    return train_mask, val_mask

def construct_data(n_atlas, adatas, input_identity, model):
    adj_all=[]
    g_all=[]
    for i in range(n_atlas):
            adata = adatas[i]
            if input_identity[i] == 'ST':
                adj_coo = adata.obsm['adj_normalized'].tocoo()
                # adj_all.append(adj_coo.todense())
                adj_all.append(adata.obsm['adj_normalized'])
            else:
                adj_raw = model.scrna_seq_adj['atlas'+str(i)]()#.weight
                adj_coo = sp.coo_matrix(adj_raw.detach().cpu().numpy())
                adj_all.append(adj_raw)
            g_all.append( dgl.graph((adj_coo.row, adj_coo.col)))
    return adj_all,g_all

class CustomGraphDataset(Dataset):
    def __init__(self, g, adata, useinput):
        self.g = g
        self.n_nodes = g.number_of_nodes()

    def __len__(self):
        return self.n_nodes

    def __getitem__(self, idx):
        # return  X[idx], batch_idx[idx], library_size[idx], x_input[idx], idx
        return idx



class CustomGraphDataLoader:
    def __init__(self, dataset_all, sampler, batch_size, shuffle, n_atlas, drop_last):
        self.dataset_all = dataset_all
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_atlas = n_atlas

        self.dataloader = []
        for i in range(n_atlas):
            self.dataloader.append(DataLoader(self.dataset_all[i], batch_size=batch_size, shuffle=shuffle,drop_last = drop_last))
        cell_num = [len(i) for i in self.dataset_all]
        self.max_value_index = np.argmax(cell_num)

    def __iter__(self):
        dataloader_iter_before={}
        dataloader_iter_after={}
        for i in np.arange(0,self.max_value_index):
                dataloader_iter_before[i] =  itertools.cycle(self.dataloader[i])
        for i in np.arange(self.max_value_index+1,self.n_atlas):
                dataloader_iter_after[i] = itertools.cycle(self.dataloader[i])

        for indices_max in self.dataloader[self.max_value_index]:
            blocks = {}
            for i in np.arange(0, self.max_value_index):
                indices_i = next(dataloader_iter_before[i])
                blocks[i] = {'single':indices_i,
                             'spatial':self.sampler.sample_blocks(self.dataset_all[i].g, indices_i)}
            blocks[self.max_value_index] = {'single':indices_max,
                                            'spatial':self.sampler.sample_blocks(self.dataset_all[self.max_value_index].g, indices_max)}
            for i in np.arange(self.max_value_index+1,self.n_atlas):
                indices_i = next(dataloader_iter_after[i])
                blocks[i] = {'single':indices_i,
                             'spatial':self.sampler.sample_blocks(self.dataset_all[i].g, indices_i)}
            yield blocks


    def __len__(self):
        return max([len(i) for i in self.dataloader])
