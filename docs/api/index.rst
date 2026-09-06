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
