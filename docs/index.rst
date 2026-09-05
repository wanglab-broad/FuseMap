FuseMap
=====================================

Spatial integration and mapping with universal gene, cell, and tissue embeddings
---------------------------------------------------------------------------------

FuseMap is a deep-learning framework for spatial transcriptomics
that (1) bridges single-cell or single-spot gene expression within spatial contexts
and (2) consolidates various gene panels across technologies, organs, and species.
Trained across atlases, FuseMap yields **universal gene, cell, and tissue embeddings**
in which datasets from different technologies, conditions, and resolutions
can be analyzed jointly.

.. container:: fusemap-hero

   .. image:: _static/framework.png
       :width: 100%
       :align: center

|

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 🔧 Installation
      :link: install
      :link-type: doc

      Install the ``fusemap`` package and download pretrained models.

   .. grid-item-card:: 🚀 Getting started
      :link: getting_started
      :link-type: doc

      Run your first integration in minutes and understand every output file.

   .. grid-item-card:: 📚 Tutorials
      :link: tutorials
      :link-type: doc

      Step-by-step notebooks organized by FuseMap's core capabilities.

   .. grid-item-card:: 🎛️ Parameters
      :link: userguide/parameters
      :link-type: doc

      Every CLI argument, model hyperparameter, and environment variable.

   .. grid-item-card:: 🧩 API reference
      :link: api/index
      :link-type: doc

      Documentation of the public functions and classes.

   .. grid-item-card:: 🤖 FuseMap Agent
      :link: agent/index
      :link-type: doc

      Chat with the mouse brain atlas through a multi-agent AI interface.

------------------------------------------

Quick start
------------------------------------------

Organize your sections as ``.h5ad`` files (spatial coordinates in
``obs['x']``/``obs['y']`` or ``obsm['spatial']``) inside one folder, then:

.. code-block:: bash

    conda create -n fusemap python=3.10.16 && conda activate fusemap
    pip install fusemap

    # integrate all sections in ./data/ into shared embeddings
    python main.py \
        --input_data_folder_path ./data/ \
        --output_save_dir ./output/ \
        --mode integrate

You get three AnnData files —
``ad_celltype_embedding.h5ad`` (cell embedding :math:`Z_c`),
``ad_tissueregion_embedding.h5ad`` (tissue embedding :math:`Z_T`),
and ``ad_gene_embedding.h5ad`` (gene embedding) —
ready for scanpy clustering, UMAP, and label transfer.
See :doc:`getting_started` for a full walk-through.

------------------------------------------

Citation
------------------------------------------

If FuseMap is useful for your research, please cite:

    He Y. et al. *Towards a universal spatial molecular atlas of the mouse brain.*
    bioRxiv (2024). `doi:10.1101/2024.05.27.594872 <https://doi.org/10.1101/2024.05.27.594872>`__


.. toctree::
   :hidden:
   :maxdepth: 2

    Installation <install>
    Getting started <getting_started>
    Tutorials <tutorials>
    Example data <data>
    Parameters <userguide/parameters>
    API reference <api/index>
    FuseMap Agent <agent/index>
    About <about>
