.. _Tutorials:

Tutorials
================================================================================

Explanatory notebooks organized by FuseMap's core capabilities. Integration and mapping
examples cover data preparation, running FuseMap, and analysis. Tutorials 3.1, 3.3, and 3.4
reuse Tutorial 1.2's outputs; precomputed outputs are available for Tutorials 1.2, 1.3,
and 1.4. Tutorial 2.3 requires data available upon request. Colab links are provided, but the
runtime must support Python 3.9–3.11 and have enough memory for the selected dataset.
See :doc:`install` for local setup and :doc:`data` for download requirements.

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

    1.1 Integrate imaging-based data across platforms <notebooks/1_spatial_integration_imaging>
    1.2 Integrate imaging- and sequencing-based data (with deconvolution) <notebooks/2_spatial_integration_cross_tech>
    1.3 Integrate across conditions (healthy × disease) <notebooks/7_cross_condition_integration>
    1.4 Beyond the brain: whole-embryo cross-resolution integration <notebooks/9_embryo_cross_resolution>

Map new data to a reference
--------------------------------------------------------------------------------

Project new sections onto a pretrained FuseMap model and transfer annotations —
either a model you trained yourself (like the ones from the integration tutorials)
or the molCCF mouse brain atlas. Mapping works best against large, diverse references
such as molCCF; for small custom references, consider integrating jointly instead
(see the note in Tutorial 2.1).

.. nbgallery::

    2.1 Map to a customized pretrained model <notebooks/4_map_new_dataset_customized>
    2.2 Map to the molCCF mouse brain atlas <notebooks/5_map_new_dataset_molCCF>
    2.3 Cross-species: map marmoset onto the mouse molCCF <notebooks/12_cross_species_marmoset>

Downstream analysis
--------------------------------------------------------------------------------

Everything here runs on embeddings a FuseMap run has already produced. The universal
**gene embedding** ties every panel to a shared gene space — so FuseMap can impute genes
that were never measured in a section (below, genes absent from the STARmap panel are
imputed from the sequencing-based datasets integrated in **Tutorial 1.2**), and, because
the embedding groups genes by function, it can also guide the design of a compact
targeted panel. The **cell** and **tissue** embeddings drive spatial interaction analysis
and reveal structure finer than any single annotation.

.. nbgallery::

    3.1 Spatially impute transcriptome-wide genes <notebooks/3_gene_spatial_imputation>
    3.2 Infer cell-cell communication <notebooks/6_cell_to_cell_interaction>
    3.3 Discover finer subregions and microenvironment cell states <notebooks/11_subregion_discovery>
    3.4 Design a targeted gene panel from the gene embedding <notebooks/10_gene_panel_selection>
