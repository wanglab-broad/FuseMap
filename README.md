# FuseMap 
Integrate spatial transcripomics with universal gene, cell, and tissue embeddings.

<p align="center">
  <img src="/img/framework.png" width="600" >
</p>

For more details, please check out our publication.


## Manuscript code and data
### Reproducibity
For code and data in the manuscript 'Towards a universal spatial molecular atlas of the mouse brain', please go to [paper_code](paper_code/).

### Exploratory analysis
We provide an interactive online database of the [molCCF](http://fusemap.spatial-atlas.net/).

## System Requirements
### Hardware requirements
`FuseMap` package requires a standard computer with optional GPU to support the in-memory operations.

### Software requirements
#### OS Requirements
This package is supported for *Linux*. The package has been tested on the following system:
+ Linux: Ubuntu 20.04

#### Python Dependencies
`FuseMap` mainly depends on the Python scientific stack.

```
dgl
numpy
scipy
scikit-learn
pandas
pytorch
scanpy
seaborn
```

## Installation
```
conda env create -f fusemap_environment.yml
```


## Tutorial
- Read a [tutorial](./tutorial_sample_data.ipynb) on [sample data](https://drive.google.com/drive/folders/1DKfP5awTUa5gaL0WB-csD0M8v-COiBfY?usp=sharing) .


## Citation

If you find FuseMap useful for your work, please cite our paper: 

> Yichun He, Hao Sheng, Hailing Shi, Wendy Xueyi Wang, Zefang Tang, Jia Liu, Xiao Wang. Towards a universal spatial molecular atlas of the mouse brain. Preprint at bioRxiv https://www.biorxiv.org/content/10.1101/2024.05.27.594872v1 (2024).
