#!/usr/bin/env python
# coding: utf-8

import sys, os


import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import scipy
from scipy import stats


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
# focus_gene=impt_gene[int(tile)]
for focus_sample in ['section1']:


 # home_addr = "/home/jialiulab/disk1/yichun/spatial_glue/scripts/FuseMap/"

 dic={}
 for focus_gene in impt_gene:
  save_dir= f"save_data_0319/rerun_version3_impt_{focus_sample}_{focus_gene}/"

  celltype_all_o=sc.read_h5ad(f"{save_dir}/transfer_celltype.h5ad")
  gene_embed = sc.read_h5ad(f"{save_dir}/ad_gene_embedding.h5ad")

  celltype_impt=celltype_all_o[celltype_all_o.obs['name']==focus_sample]
  starmapimpt=celltype_impt.X@gene_embed.X.T

  X1 = sc.read_h5ad(f'./data/version3/{focus_sample}.h5ad')
  sc.pp.normalize_total(X1)#, target_sum=1e4)
  sc.pp.log1p(X1)
  sc.pp.scale(X1, zero_center=False, max_value=10)


  x, y = X1.X[:,np.where(X1.var.index==focus_gene)[0][0]], \
         starmapimpt[:,np.where(gene_embed.obs.index==focus_gene)[0][0]]
  print(x.shape)
  print(y.shape)
  if scipy.sparse.isspmatrix(x):
    x = x.toarray().flatten()
  if scipy.sparse.isspmatrix(y):
    y = y.toarray().flatten()
  res = stats.pearsonr(x, y)

  dic[focus_gene]=res[0]

 pd.DataFrame(dic,index=range(len(dic))).to_csv(f'nospatial_{focus_sample}_fusemap.csv')