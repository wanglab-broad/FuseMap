.. _Parameters:

Parameters
================================================================================

FuseMap runs with validated defaults — a standard integration needs **no tuning**.
This page documents every knob for completeness, grouped by how likely you are
to need it.

Command-line arguments
--------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Argument
     - Default
     - Meaning
   * - ``--input_data_folder_path``
     - (required)
     - Folder containing one ``.h5ad`` per spatial section.
   * - ``--output_save_dir``
     - (required)
     - Output directory (created if missing).
   * - ``--mode``
     - (required)
     - ``integrate`` — train shared embeddings across all sections;
       ``map`` — map each section onto a pretrained reference model.
   * - ``--keep_celltype``
     - ``""``
     - Name of an ``obs`` column with cell-type labels to carry into
       ``ad_celltype_embedding.h5ad``.
   * - ``--keep_tissueregion``
     - ``""``
     - Name of an ``obs`` column with tissue-region labels to carry into
       ``ad_tissueregion_embedding.h5ad``.
   * - ``--use_llm_gene_embedding``
     - ``false``
     - Initialize the gene embedding from a language-model gene representation
       (``combine`` mode); requires the GenePT pickle.
   * - ``--pretrain_model_path``
     - ``""``
     - Path to pretrained weights (used with ``--mode map``).

Model hyperparameters
--------------------------------------------------------------------------------

Set in :class:`fusemap.config.ModelType`. These are the values used for all
results in the paper; we recommend leaving them unchanged.

.. list-table::
   :header-rows: 1
   :widths: 26 14 60

   * - Parameter
     - Value
     - Meaning
   * - ``pca_dim``
     - 50
     - Input PCA dimension.
   * - ``hidden_dim``
     - 512
     - Encoder/decoder hidden width.
   * - ``latent_dim``
     - 64
     - Dimension of the universal embeddings (:math:`Z_c`, :math:`Z_T`).
   * - ``n_epochs``
     - 16
     - Training epochs per phase. Changing this alters results.
   * - ``batch_size``
     - 64
     - Mini-batch size. Changing this alters results.
   * - ``learning_rate``
     - 0.001
     - RMSprop learning rate (with plateau decay).
   * - ``dropout_rate``
     - 0.2
     - Encoder dropout.

Environment variables
--------------------------------------------------------------------------------

Advanced switches read from the environment at launch, e.g.
``FUSEMAP_ANCHOR_LAMBDA=0 python main.py ...``.

Cross-sample anchor alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Improves mixing across technologies and conditions via mutual-nearest-neighbor
anchors, gated per sample pair so that dissimilar populations are never forced
together.

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Variable
     - Default
     - Meaning
   * - ``FUSEMAP_ANCHOR_LAMBDA``
     - ``0.3``
     - Weight of the anchor alignment loss.
       **Set to** ``0`` **to reproduce the original FuseMap behavior exactly.**
   * - ``FUSEMAP_ANCHOR_START``
     - ``2``
     - First epoch (of the final phase) at which anchors are used.
   * - ``FUSEMAP_ANCHOR_SIM``
     - ``0.5``
     - Minimum cosine similarity for an MNN pair to count as an anchor.
   * - ``FUSEMAP_ANCHOR_STABLE``
     - ``1``
     - Keep only anchors stable across consecutive refreshes.
   * - ``FUSEMAP_ANCHOR_PRETRAIN``
     - ``0``
     - Also apply anchors during pretraining (not recommended).
   * - ``FUSEMAP_ANCHOR_QUERY``
     - ``""``
     - Comma-separated file-name substrings marking query-only datasets:
       they are pulled toward others, never the reverse.
   * - ``FUSEMAP_BALANCE_RES``
     - ``4``
     - Leiden resolution for the pairwise balance weights that gate anchors.
   * - ``FUSEMAP_BALANCE_CUTOFF`` / ``FUSEMAP_BALANCE_POWER``
     - ``0.75`` / ``8``
     - Similarity cutoff and sharpening power of the anchor gate.

Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Variable
     - Default
     - Meaning
   * - ``FUSEMAP_LOADER_WORKERS``
     - ``8``
     - Parallel batch-prefetch workers.
   * - ``CUDA_VISIBLE_DEVICES``
     - —
     - GPU selection.
   * - ``OMP_NUM_THREADS`` etc.
     - —
     - CPU thread caps for BLAS/OpenMP (set to 32 on large machines).

Bead deconvolution (Stage-B)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Post-processing for bead/spot-resolution data (Slide-seq, Visium HD) after an
integration run: ``python stage_b_deconv.py``. Each bead is decomposed into a
mixture over cell archetypes learned from the single-cell sections; its cell
and tissue embeddings are rebuilt from the mixture. Dataset roles are **defined
by you**, not auto-detected.

.. list-table::
   :header-rows: 1
   :widths: 32 12 56

   * - Variable
     - Default
     - Meaning
   * - ``FUSEMAP_BEAD_FILES``
     - (required)
     - Comma-separated file-name substrings, each resolving to exactly one
       input file: the bead-resolution dataset(s) to deconvolve.
   * - ``FUSEMAP_SIG_REF``
     - (required)
     - Comma-separated substrings selecting the single-cell reference
       dataset(s) whose expression builds the archetype signatures.
   * - ``STAGEB_OUT_DIR`` / ``STAGEB_DATA_DIR``
     - —
     - Integration output dir / input data dir.
   * - ``STAGEB_ENT_W``
     - ``5e-4``
     - Entropy regularization of the mixture weights.
   * - ``STAGEB_SIG``
     - ``empirical``
     - Signature mode (``empirical`` = mean member expression; recommended).

Experimental (default off)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``FUSEMAP_STRUCT_LAMBDA`` (``0``) enables a within-dataset structure-preservation
triplet loss with knobs ``FUSEMAP_STRUCT_K``, ``FUSEMAP_STRUCT_MARGIN``,
``FUSEMAP_STRUCT_NEGC``, ``FUSEMAP_STRUCT_HARDNEG``. In our benchmarks it did
not improve results; it is kept for research use.
