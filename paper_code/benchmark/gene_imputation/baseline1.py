#!/usr/bin/env python
# coding: utf-8

import sys, os
import scanpy.external as sce
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad

tile = int(sys.argv[1])
print(f'tile:{tile}')





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

 X1_raw = sc.read_h5ad(f'./data/version3/{focus_sample}.h5ad')
 sc.pp.normalize_total(X1_raw)
 sc.pp.log1p(X1_raw)
 sc.pp.scale(X1_raw, zero_center=False, max_value=10)
 X1_raw.X=X1_raw.X.toarray()

 if focus_sample=='section4':
  X1 = sc.read_h5ad('./data/version3/section1.h5ad')
  X2 = sc.read_h5ad('./data/version3/section2.h5ad')
  X3 = sc.read_h5ad('./data/version3/section3.h5ad')
  X4 = sc.read_h5ad(f'./data/version3/impt/section4_{focus_gene}.h5ad')
  X4.X = X4.X.toarray()
  ad_merged_raw=ad.concat([X1,X2,X3])

 X1.obs['name'] = 'section1'
 X2.obs['name'] = 'section2'
 X3.obs['name'] = 'section3'
 X4.obs['name'] = 'section4'

 sc.pp.normalize_total(ad_merged_raw)
 sc.pp.log1p(ad_merged_raw)
 sc.pp.scale(ad_merged_raw)


 X1_processed = X1.copy()
 X2_processed = X2.copy()
 X3_processed = X3.copy()
 X4_processed = X4.copy()


 sc.pp.normalize_total(X1_processed)
 sc.pp.log1p(X1_processed)
 sc.pp.scale(X1_processed)

 sc.pp.normalize_total(X2_processed)
 sc.pp.log1p(X2_processed)
 sc.pp.scale(X2_processed)

 sc.pp.normalize_total(X3_processed)
 sc.pp.log1p(X3_processed)
 sc.pp.scale(X3_processed)

 sc.pp.normalize_total(X4_processed)
 sc.pp.log1p(X4_processed)
 sc.pp.scale(X4_processed)

 ad_merged = ad.concat([X1_processed,X2_processed,X3_processed,X4_processed])

 atlasname=['section1','section2','section3','section4']
 atlas_adata = [X1_processed,X2_processed,X3_processed,X4_processed]


 sc.tl.pca(ad_merged,n_comps=50)
 ad_merged.obs['domain']=np.concatenate([[j]*i.shape[0] for i,j in zip(atlas_adata,atlasname)])
 sce.pp.harmony_integrate(ad_merged,'domain')
 sc.pp.neighbors(ad_merged,n_neighbors=50,use_rep='X_pca_harmony')
 sc.tl.umap(ad_merged)



 gene_raw = ad_merged_raw[:,ad_merged_raw.var.index==focus_gene].X.flatten()


 ad_merged_starmap = ad_merged[ad_merged.obs['name'] == focus_sample]
 ad_merged_other = ad_merged[ad_merged.obs['name'] != focus_sample]

 n1_cells = np.array(ad_merged_starmap.obsm['X_umap'])  # Replace with your n1 cells data
 n2_cells = np.array(ad_merged_other.obsm['X_umap'])  # Replace with your n2 cells data

 # Initialize Nearest Neighbors model
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
 x, y = X1_raw[:,X1_raw.var.index==focus_gene].X.reshape(-1,), average_features
 res = stats.pearsonr(x, y)

 dic={focus_gene:res[0]}
 pd.DataFrame(dic,index=range(1)).to_csv(f'source_data/baseline1/{focus_sample}_{focus_gene}.csv')