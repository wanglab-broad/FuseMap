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

   spatial_integrate.spatial_integrate
   spatial_map.spatial_map

Preprocessing
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   preprocess.preprocess_raw
   preprocess.preprocess_adata
   preprocess.construct_graph
   preprocess.preprocess_adj_sparse
   preprocess.get_spatial_input

Model
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   model.Fuse_network
   model.FuseMapEncoder
   model.FuseMapDecoder
   model.FuseMapAdaptDecoder
   model.Discriminator
   model.Adj_model
   model.NNTransfer

Training
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   train_model.pretrain_model
   train_model.train_model
   train_model.balance_weight
   train_model.refresh_anchors
   train_model.map_model
   train_model.read_model

Losses
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   loss.compute_ae_loss
   loss.compute_dis_loss
   loss.compute_ae_loss_pretrain
   loss.compute_dis_loss_pretrain
   loss.compute_anchor_loss
   loss.compute_struct_loss
   loss.get_balance_weight
   loss.get_balance_weight_subsample
   loss.AE_Gene_loss

Data handling
--------------------------------------------------------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   dataset.CustomGraphDataset
   dataset.CustomGraphDataLoader
   dataset.construct_data
   dataset.construct_mask

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

Two standalone scripts operate on a finished integration run
(see :doc:`../userguide/parameters` for their environment variables):

``stage_b_deconv.py``
    Bead deconvolution for spot/bead-resolution datasets (Slide-seq, Visium HD):
    decomposes each bead into a mixture over cell archetypes and rebuilds its
    cell/tissue embeddings.

    .. code-block:: bash

        FUSEMAP_BEAD_FILES=slideseq FUSEMAP_SIG_REF=starmap \
        STAGEB_DATA_DIR=./data STAGEB_OUT_DIR=./output \
        python stage_b_deconv.py

``niche_align.py``
    Computes spatial-niche composition vectors for quantitative region-level
    comparison across samples.
