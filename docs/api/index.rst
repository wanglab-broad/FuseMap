.. _API:

API reference
================================================================================

Import FuseMap as::

    import fusemap

Pipelines
--------------------------------------------------------------------------------

High-level entry points (also exposed through ``main.py``).

.. currentmodule:: fusemap

.. autosummary::
   :toctree: generated/
   :nosignatures:

   integrate
   map_to_reference
   prepare_reference_signatures
   deconvolve_beads
   transfer_labels
   read_input_folder
   training.integrate.spatial_integrate
   training.map.spatial_map

Preprocessing
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   data.graph.preprocess_raw
   data.graph.preprocess_adata
   data.graph.construct_graph
   data.graph.preprocess_adj_sparse
   data.graph.get_spatial_input

Model
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   models.network.Fuse_network
   models.network.FuseMapEncoder
   models.network.FuseMapDecoder
   models.network.FuseMapAdaptDecoder
   models.network.Discriminator
   models.network.Adj_model
   models.network.NNTransfer

Training
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   training.train_model.pretrain_model
   training.train_model.train_model
   training.train_model.balance_weight
   training.train_model.refresh_anchors
   training.train_model.map_model
   training.train_model.read_model

Losses
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   models.losses.compute_ae_loss
   models.losses.compute_dis_loss
   models.losses.compute_ae_loss_pretrain
   models.losses.compute_dis_loss_pretrain
   models.losses.compute_anchor_loss
   models.losses.compute_struct_loss
   models.losses.get_balance_weight
   models.losses.get_balance_weight_subsample
   models.losses.AE_Gene_loss

Data handling
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   data.loaders.CustomGraphDataset
   data.loaders.CustomGraphDataLoader
   data.loaders.construct_data
   data.loaders.construct_mask

Configuration
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   config.parse_input_args
   config.ModelType
   config.AnchorConfig

Utilities
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.seed_all
   utils.generate_ad_embed
   utils.average_embeddings
   utils.transfer_annotation
   utils.transfer_celltype
   utils.save_snapshot
   utils.load_snapshot

Post-processing scripts
--------------------------------------------------------------------------------

Bead mapping can reuse frozen reference signatures without retraining the reference:

.. code-block:: python

    fusemap.map_to_reference(
        "./new_beads", "./mapped", "./reference_model",
        bead_files="slideseq",
        sig_ref="merfish,starmap",
        reference_data_folder_path="./reference_data",
    )

The first call prepares ``./mapped/reference_signatures.npz`` from the declared
single-cell sections' saved embeddings and original expression data. Subsequent
calls can use ``reference_signatures_path="./mapped/reference_signatures.npz"``
instead of ``sig_ref`` and ``reference_data_folder_path``. The saved reference
checkpoint and embeddings remain unchanged. Query adaptation still runs.
Use :func:`fusemap.prepare_reference_signatures` to prepare this artifact separately.

The mixture objective and spatial readout match integration Stage-B. Reference
archetypes are fixed, so results need not equal a joint integration that retrains
the reference and changes its embedding space. Select only single-cell sections
as ``sig_ref``; in mapping these sections define both prototypes and signatures.
Without ``bead_files``, mapping retains its ordinary embedding-only behavior.

For each declared bead dataset, canonical embeddings contain ``obsm['stageB_pi']``
and per-bead reconstruction errors. ``stageB_pi.npz`` also records observation IDs,
gene coverage and the reference checkpoint fingerprint. Mixture weights describe
reference archetypes; they are not calibrated cell counts. The expression panel's
coverage and ability to distinguish archetypes limit composition inference.

Two post-processing modules also operate on a finished integration run
(see :doc:`../userguide/parameters` for their environment variables):

``fusemap.postprocess.stage_b_script``
    Bead deconvolution for spot/bead-resolution datasets (Slide-seq, Visium HD):
    decomposes each bead into a mixture over cell archetypes and rebuilds its
    cell/tissue embeddings. Prefer the high-level entry points — it runs
    automatically when ``bead_files``/``sig_ref`` are declared to
    :func:`fusemap.integrate`, or call :func:`fusemap.deconvolve_beads`.
    Script form:

    .. code-block:: bash

        FUSEMAP_BEAD_FILES=slideseq FUSEMAP_SIG_REF=starmap \
        STAGEB_DATA_DIR=./data STAGEB_OUT_DIR=./output \
        python -m fusemap.postprocess.stage_b_script

``fusemap.postprocess.niche_align``
    Computes spatial-niche composition vectors for quantitative region-level
    comparison across samples:

    .. code-block:: bash

        python -m fusemap.postprocess.niche_align
