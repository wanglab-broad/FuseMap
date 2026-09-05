.. _Tutorials:

Tutorials
================================================================================

Explanatory notebooks organized by FuseMap's core capabilities.
Example data for all tutorials:
`Google Drive <https://drive.google.com/drive/folders/1nMWUzIcmzd4BQztUenwPJdL9zh2gj2Dd?usp=sharing>`__.

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

Impute transcriptome-wide expression
--------------------------------------------------------------------------------

The universal **gene embedding** ties every panel to a shared gene space,
letting FuseMap impute genes that were never measured in a section.

.. nbgallery::

    Spatially impute transcriptome-wide genes <notebooks/3_gene_spatial_imputation>

Map new data to a reference
--------------------------------------------------------------------------------

Instead of retraining, project new sections onto a pretrained FuseMap model —
your own reference or the molCCF universal mouse brain atlas — and transfer
cell-type and tissue-region annotations.

.. nbgallery::

    Map to a customized pretrained model <notebooks/4_map_new_dataset_customized>
    Map to the molCCF mouse brain atlas <notebooks/5_map_new_dataset_molCCF>

Downstream analysis
--------------------------------------------------------------------------------

.. nbgallery::

    Infer cell-cell communication <notebooks/6_cell_to_cell_interaction>
