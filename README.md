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

For bead mapping with deconvolution, install the 1.2.0 source release:

```bash
python -m pip install "fusemap[tutorials] @ git+https://github.com/wanglab-broad/FuseMap.git@v1.2.0"
```

```python
fusemap.map_to_reference(
    "./new_beads", "./mapped", "./reference_model",
    bead_files="slideseq", sig_ref="merfish,starmap",
    reference_data_folder_path="./reference_data",
)
```

This trains query adaptation and fits bead mixtures against frozen reference
archetypes without retraining the reference model. The original reference
expression files are read once to build signatures. Subsequent queries can use
`reference_signatures_path="./mapped/reference_signatures.npz"` instead of
`reference_data_folder_path` and `sig_ref`. Mixtures are written to `stageB_pi.npz`;
the original mapped embeddings are preserved as `_nodeconv`. See
[Tutorial 2.1](docs/notebooks/4_map_new_dataset_customized.ipynb) for the readouts
and their interpretation. Archetype weights are not calibrated cell counts.

The classic Agent uses `python -m pip install ".[agent]"` from this repository.
Agent v2 uses a separate environment and a registered scientific kernel; follow
[the Agent setup instructions](agent_setup/README.md#agent-v2).

- Read the FuseMap tutorial [here](https://fusemap.readthedocs.io/en/latest/).
- FuseMap-Agent set up tutorial [here](https://github.com/wanglab-broad/FuseMap/tree/main/agent_setup).

### Recently added tutorials

These notebooks are included in the repository and linked from the
[tutorial index](https://fusemap.readthedocs.io/en/latest/tutorials.html).
You can also open them directly on GitHub:

| Tutorial | Notebook | Prerequisite |
|---|---|---|
| 2.3 | [Cross-species mapping: marmoset to mouse molCCF](docs/notebooks/12_cross_species_marmoset.ipynb) | molCCF reference; marmoset data available from the corresponding authors upon reasonable request |
| 3.3 | [Subregion discovery and spatially associated cell states](docs/notebooks/11_subregion_discovery.ipynb) | Tutorial 1.2 cell, tissue, and gene embeddings |
| 3.4 | [Targeted gene panel selection](docs/notebooks/10_gene_panel_selection.ipynb) | Tutorial 1.2 data and gene embedding; uses Leiden modules as a teaching adaptation of the paper's WGCNA workflow |


## Citation

The current manuscript is titled **An agentic AI spatial molecular foundation model of the brain**.
The earlier public preprint can be cited as:

> Yichun He, Hao Sheng, Hailing Shi, Wendy Xueyi Wang, Zefang Tang, Jia Liu, Xiao Wang. Towards a universal spatial molecular atlas of the mouse brain. Preprint at bioRxiv https://www.biorxiv.org/content/10.1101/2024.05.27.594872v1 (2024).
