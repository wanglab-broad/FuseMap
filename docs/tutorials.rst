.. _Tutorials:

Tutorials
================================================================================

Explanatory notebooks organized by FuseMap's core capabilities. Every tutorial follows the
same three steps: **download the data** (one cell) → **run FuseMap** (one function call) →
**analyze the results**. Tutorials that need long training ship precomputed outputs so you
can skip straight to the analysis.

.. seealso::

    For the conversational AI interface to the mouse brain atlas,
    see :doc:`FuseMap Agent <agent/index>`.

Integrate spatial atlases
--------------------------------------------------------------------------------

FuseMap trains one encoder per section against shared **cell** (:math:`Z_c`) and
**tissue** (:math:`Z_T`) embedding spaces, so sections with different gene panels,
technologies, and conditions become jointly analyzable.

.. nbgallery::

    Integrate imaging-based data <notebooks/1_spatial_integration_imaging>
    Integrate imaging- and sequencing-based data <notebooks/2_spatial_integration_cross_tech>
    Integrate across conditions (healthy × disease) <notebooks/7_cross_condition_integration>

Deconvolve bead-resolution data
--------------------------------------------------------------------------------

Bead/spot technologies (Slide-seq, Visium HD) measure mixtures of cells. Stage-B
deconvolution decomposes each bead into cell archetypes and rebuilds its embeddings.

.. nbgallery::

    Deconvolve Slide-seq beads (Stage-B) <notebooks/8_bead_deconvolution>

Impute transcriptome-wide expression
--------------------------------------------------------------------------------

The universal **gene embedding** ties every panel to a shared gene space,
letting FuseMap impute genes that were never measured in a section.

.. nbgallery::

    Spatially impute transcriptome-wide genes <notebooks/3_gene_spatial_imputation>

Map new data to a reference
--------------------------------------------------------------------------------

Project new sections onto a pretrained FuseMap model and transfer annotations.
Mapping works best against large, diverse references such as molCCF; for small custom
references, consider integrating jointly instead (see the note in Tutorial 4).

.. nbgallery::

    Map to a customized pretrained model <notebooks/4_map_new_dataset_customized>
    Map to the molCCF mouse brain atlas <notebooks/5_map_new_dataset_molCCF>

Beyond the brain
--------------------------------------------------------------------------------

The same pipeline generalizes to any tissue — here, whole mouse embryos across
resolutions and technologies.

.. nbgallery::

    Whole-embryo cross-resolution integration <notebooks/9_embryo_cross_resolution>

Downstream analysis
--------------------------------------------------------------------------------

.. nbgallery::

    Infer cell-cell communication <notebooks/6_cell_to_cell_interaction>
