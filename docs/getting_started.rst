.. _GettingStarted:

Getting started
================================================================================

This page takes you from a folder of ``.h5ad`` files to your first integrated
UMAP in three steps. For conceptual background see the
`FuseMap paper <https://doi.org/10.1101/2024.05.27.594872>`__;
for task-specific walk-throughs see :doc:`tutorials`.

1. Prepare the input folder
--------------------------------------------------------------------------------

Place one ``.h5ad`` file (`AnnData <https://anndata.readthedocs.io>`__) per
spatial section in a single folder:

::

    data/
    ├── section_A.h5ad
    ├── section_B.h5ad
    └── section_C.h5ad

Each file must satisfy:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Spatial coordinates
     - ``obs['x']`` and ``obs['y']`` (or ``obsm['spatial']``,
       or ``obs['col']``/``obs['row']``).
   * - Expression matrix
     - Raw counts in ``X`` recommended — FuseMap normalizes internally
       (``normalize_total`` → ``log1p`` → ``scale``). Already-normalized input is
       used as-is. **Use the same convention for all sections.**
   * - Gene names
     - ``var_names`` as gene symbols. Panels do **not** need to overlap fully;
       shared genes anchor the gene-embedding space.
   * - (Optional) annotations
     - Cell-type / tissue-region labels in ``obs``, carried into the outputs via
       ``--keep_celltype`` / ``--keep_tissueregion``.

2. Run the integration
--------------------------------------------------------------------------------

.. code-block:: bash

    python main.py \
        --input_data_folder_path ./data/ \
        --output_save_dir ./output/ \
        --mode integrate \
        --keep_celltype celltype_anno \
        --keep_tissueregion tissueregion_anno

.. note::

    A GPU is strongly recommended (select one with ``CUDA_VISIBLE_DEVICES``).
    Runtime scales roughly linearly with total cell number; two ~10k-cell
    sections finish in tens of minutes on a single GPU.
    Training snapshots are saved each epoch — re-running the same command
    resumes automatically from ``snapshot.pt``.

3. Understand the outputs
--------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - File in ``output/``
     - Content
   * - ``ad_celltype_embedding.h5ad``
     - **Cell embedding** :math:`Z_c` (one row per cell, 64-dim ``X``).
       Use for cell-type clustering, UMAP, and cross-sample label transfer.
   * - ``ad_tissueregion_embedding.h5ad``
     - **Tissue embedding** :math:`Z_T` (spatial-neighborhood aggregation of
       :math:`Z_c`). Use for tissue-region identification and niche analysis.
   * - ``ad_gene_embedding.h5ad``
     - **Gene embedding** (one row per gene). Use for gene-program discovery
       and transcriptome-wide imputation.
   * - ``trained_model/``
     - Model weights (final and intermediate checkpoints).
   * - ``snapshot.pt``
     - Auto-resume training snapshot.
   * - ``config.csv``, ``*.log``
     - Run configuration and logs.
   * - ``latent_embeddings_all_*.pkl``, ``balance_weight*.pkl``, ...
     - Internal artifacts (raw latents per atlas, adversarial balance weights);
       not needed for downstream analysis.

Both embedding files keep ``obs['name']`` (section), ``obs['file_name']``,
``obs['x']``/``obs['y']``, and any label columns you passed via
``--keep_celltype`` / ``--keep_tissueregion``.

4. First look at the result
--------------------------------------------------------------------------------

.. code-block:: python

    import scanpy as sc

    ad = sc.read_h5ad("output/ad_celltype_embedding.h5ad")

    sc.pp.neighbors(ad, use_rep="X", n_neighbors=50)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=0.5)

    # sections should interleave; clusters should follow cell types
    sc.pl.umap(ad, color=["name", "leiden"])

A well-integrated result shows sections mixing within clusters while
biological structure (cell types, tissue regions) stays separated.
Repeat with ``ad_tissueregion_embedding.h5ad`` to inspect tissue regions —
its Leiden clusters projected back to ``obs['x']/obs['y']`` should form
spatially coherent anatomical domains.

Next steps
--------------------------------------------------------------------------------

- :doc:`tutorials` — cross-technology integration, imputation, mapping to molCCF.
- :doc:`userguide/parameters` — what you can tune (and what you should not).
- :doc:`api/index` — call :func:`fusemap.spatial_integrate.spatial_integrate`
  directly from Python.
