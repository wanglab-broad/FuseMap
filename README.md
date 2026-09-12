# FuseMap 
Integrate spatial transcriptomics with universal gene, cell, and tissue embeddings.

<p align="center">
  <img src="https://raw.githubusercontent.com/wanglab-broad/FuseMap/main/docs/_static/framework.png" width="600" >
</p>

Current manuscript: **An agentic AI spatial molecular foundation model of the brain**.
See [release notes](CHANGELOG.md) for the distinction between the manuscript workflows
and the current software, including anchor alignment and Agent v2.


## Manuscript code and data
### Reproducibity
For manuscript analysis code, see [paper_code](paper_code/). The current manuscript
describes a frozen code and analysis archive at [Zenodo](https://doi.org/10.5281/zenodo.22103552).
Use that archive and its run configurations for paper reproduction.

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


## Installation and Tutorial

Use Python 3.9–3.11 (3.10 recommended). Install this checkout with:

```bash
python -m pip install .
```

For notebook dependencies, use `python -m pip install ".[tutorials]"`.
The PyPI package is installed with `python -m pip install fusemap`; see the
[release notes](CHANGELOG.md) for this checkout's version.

```python
import fusemap
fusemap.integrate("./data", "./output")
```

The classic Agent uses `python -m pip install ".[agent]"` from this repository.
Agent v2 uses a separate environment and a registered scientific kernel; follow
[the Agent setup instructions](agent_setup/README.md#agent-v2).

- Read the FuseMap tutorial [here](https://fusemap.readthedocs.io/en/latest/).
- FuseMap-Agent set up tutorial [here](https://github.com/wanglab-broad/FuseMap/tree/main/agent_setup).


## Citation

The current manuscript is titled **An agentic AI spatial molecular foundation model of the brain**.
The earlier public preprint can be cited as:

> Yichun He, Hao Sheng, Hailing Shi, Wendy Xueyi Wang, Zefang Tang, Jia Liu, Xiao Wang. Towards a universal spatial molecular atlas of the mouse brain. Preprint at bioRxiv https://www.biorxiv.org/content/10.1101/2024.05.27.594872v1 (2024).
