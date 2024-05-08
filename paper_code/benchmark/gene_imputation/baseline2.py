#!/usr/bin/env python
# coding: utf-8

import sys, os
import scanpy.external as sce

tile = int(sys.argv[1])
print(f'tile:{tile}')

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad



impt_gene=['ABCC9',
 'ADRA1B',
 'AQP4',
 'CALCR',
 'CASR',
 'CHAT',
 'CHRM2',
 'CHRM3',
 'CLDN5',
 'CNR1',
 'DGKK',
 'DLK1',
 'DRD1',
 'DRD2',
 'DRD3',
 'DRD5',
 'EOMES',
 'GALR1',
 'GFAP',
 'GPR101',
 'GPR139',
 'GPR83',
 'GPR88',
 'GRM1',
 'GRM3',
 'GRPR',
 'HCRTR2',
 'HTR1B',
 'HTR1D',
 'HTR2A',
 'IGFBPL1',
 'KCNJ8',
 'KIT',
 'MAN1A',
 'NPBWR1',
 'NPSR1',
 'NPY2R',
 'OPRD1',
 'OPRK1',
 'OXTR',
 'PTH2R',
 'RET',
 'RXFP1',
 'RXFP3',
 'SLC17A6',
 'SLC17A7',
 'SLC17A8',
 'SLC32A1',
 'TACR1',
 'TACR3',
 'TH',
 'TRHR']

focus_gene=impt_gene[int(tile)]
for focus_sample in ['section4']:
 print(focus_sample, focus_gene)


 X1 = sc.read_h5ad(f'./data/version3/impt/{focus_sample}_{focus_gene}.h5ad')
 # X1=X1[range(1000),:]
 X1.X = X1.X.toarray()

 X1.var.index = X1.var.index.str.upper()
 sc.pp.normalize_total(X1)
 sc.pp.log1p(X1)
 sc.pp.scale(X1)


 scrna = sc.read_h5ad('./data/version3/scrna.h5ad')
 scrna.X = scrna.X.toarray()
 scrna.var.index = scrna.var.index.str.upper()
 scrna.var_names_make_unique()

 list_of_variable_names = X1.var.index.intersection(scrna.var.index)
 X1_subset = X1[:, list_of_variable_names]
 scrna_subset = scrna[:, list_of_variable_names]

 sc.pp.normalize_total(scrna_subset)
 sc.pp.log1p(scrna_subset)
 sc.pp.calculate_qc_metrics(scrna_subset, percent_top=None, inplace=True)
 sc.pp.regress_out(scrna_subset, ['total_counts'])
 sc.pp.scale(scrna_subset)


 combine_adata = X1_subset.concatenate(scrna_subset, batch_key='dataset', batch_categories=['st', 'scrna'])
 sc.tl.pca(combine_adata, n_comps=100)
 sce.pp.harmony_integrate(combine_adata, 'dataset')
 print('harmony finished')
 sc.pp.neighbors(combine_adata, n_neighbors=50, use_rep='X_pca_harmony')
 sc.tl.umap(combine_adata)

 combine_adata.write_h5ad(f'{focus_sample}_{focus_gene}.h5ad')

 ad_merged_raw = scrna.copy()
 sc.pp.normalize_total(ad_merged_raw)
 sc.pp.log1p(ad_merged_raw)
 sc.pp.scale(ad_merged_raw)
 gene_raw = ad_merged_raw[:, ad_merged_raw.var.index == focus_gene].X.toarray().flatten()

 import numpy as np
 from sklearn.neighbors import NearestNeighbors

 ad_merged_starmap = combine_adata[combine_adata.obs['dataset'] == 'st']
 ad_merged_other = combine_adata[combine_adata.obs['dataset'] != 'st']
 n1_cells = np.array(ad_merged_starmap.obsm['X_umap'])  # Replace with your n1 cells data
 n2_cells = np.array(ad_merged_other.obsm['X_umap'])  # Replace with your n2 cells data

 neigh = NearestNeighbors(n_neighbors=50)
 neigh.fit(n2_cells)
 distances, indices = neigh.kneighbors(n1_cells)

 average_features = []
 for index_array in indices:
  neighbors = gene_raw[index_array]
  mean_feature = np.mean(neighbors, axis=0)
  average_features.append(mean_feature)
 average_features = np.array(average_features)



 from scipy import stats
 x, y = X1[:, X1.var.index == focus_gene].X.reshape(-1, ), average_features

 res = stats.pearsonr(x, y)

 dic={focus_gene:res[0]}
 pd.DataFrame(dic,index=range(1)).to_csv(f'source_data/baseline2/{focus_sample}_{focus_gene}.csv')