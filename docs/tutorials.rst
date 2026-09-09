.. _Tutorials:

Tutorials
================================================================================

Explanatory notebooks organized by FuseMap's core capabilities. Every tutorial follows the
same three steps: **download the data** (one cell) → **run FuseMap** (one function call) →
**analyze the results**. Tutorials that need long training ship precomputed outputs so you
can skip straight to the analysis. Every notebook opens directly in **Google Colab**
(badge at the top of each page) — free GPU, nothing to install.

.. seealso::

    For the conversational AI interface to the mouse brain atlas,
    see :doc:`FuseMap Agent <agent/index>`.

Integrate spatial atlases
--------------------------------------------------------------------------------

FuseMap trains one encoder per section against shared **cell** (:math:`Z_c`) and
**tissue** (:math:`Z_T`) embedding spaces, so sections with different gene panels,
technologies, and conditions become jointly analyzable. Sequencing-based bead/spot
datasets (Slide-seq, Stereo-seq, Visium HD) are deconvolved automatically when declared
as bead data. After integration, annotations **transfer** across datasets through the
shared embedding, and cell types / tissue regions can be **redefined de novo** from it.

.. nbgallery::

    Integrate imaging-based data across platforms <notebooks/1_spatial_integration_imaging>
    Integrate imaging- and sequencing-based data (with deconvolution) <notebooks/2_spatial_integration_cross_tech>
    Integrate across conditions (healthy × disease) <notebooks/7_cross_condition_integration>
    Beyond the brain: whole-embryo cross-resolution integration <notebooks/9_embryo_cross_resolution>

Impute transcriptome-wide expression
--------------------------------------------------------------------------------

The universal **gene embedding** ties every panel to a shared gene space,
letting FuseMap impute genes that were never measured in a section — here, imputing
genes unmeasured in the STARmap panel from the sequencing-based datasets integrated
in the tutorial above.

.. nbgallery::

    Spatially impute transcriptome-wide genes <notebooks/3_gene_spatial_imputation>

Map new data to a reference
--------------------------------------------------------------------------------

Project new sections onto a pretrained FuseMap model and transfer annotations —
either a model you trained yourself (like the ones from the integration tutorials)
or the molCCF mouse brain atlas. Mapping works best against large, diverse references
such as molCCF; for small custom references, consider integrating jointly instead
(see the note in Tutorial 4).

.. nbgallery::

    Map to a customized pretrained model <notebooks/4_map_new_dataset_customized>
    Map to the molCCF mouse brain atlas <notebooks/5_map_new_dataset_molCCF>

Downstream analysis
--------------------------------------------------------------------------------

.. nbgallery::

    Infer cell-cell communication <notebooks/6_cell_to_cell_interaction>
