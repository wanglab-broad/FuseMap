.. _Data:

Example data
================================================================================

Public tutorial datasets are indexed in the
`Google Drive folder <https://drive.google.com/drive/folders/1nMWUzIcmzd4BQztUenwPJdL9zh2gj2Dd>`__.
Download instructions are provided in each tutorial; Tutorials 3, 10, and 11 reuse
Tutorial 2's data and outputs. **Tutorial 12 requires marmoset data obtained from
the corresponding authors upon reasonable request**, as stated in the manuscript.

Tutorial datasets
--------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 13 16 9 19 19

   * - Dataset
     - Technology
     - Tissue
     - Tutorial(s)
     - Publication
     - Download
   * - ``merfish.h5ad``
     - MERFISH
     - adult mouse brain
     - 1
     - `Zhang et al. 2023 <https://doi.org/10.1038/s41586-023-06808-9>`__
     - `merfish.h5ad <https://drive.google.com/uc?id=1PUvH3JpGEJ0CrP9DrpIzkH9tIi4VOZ6E>`__
   * - ``starmap.h5ad``
     - STARmap
     - adult mouse brain
     - 1, 2, 3, 4
     - `Shi et al. 2023 <https://doi.org/10.1038/s41586-023-06569-5>`__
     - `starmap.h5ad <https://drive.google.com/uc?id=1jPniIBchuYUPgBdyn16bjt11orsYiSKj>`__
   * - ``stereoseq_mousebrain.h5ad``
     - Stereo-seq
     - adult mouse brain
     - 2, 3
     - `Chen et al. 2022 <https://doi.org/10.1016/j.cell.2022.04.003>`__
     - `stereoseq_mousebrain.h5ad <https://drive.google.com/uc?id=17SZwS2qyOV4xNss9xb4UXYnAyz20Wery>`__
   * - ``slideseq_Puck34.h5ad``
     - Slide-seq V2
     - adult mouse brain
     - 5
     - `Langlieb et al. 2023 <https://doi.org/10.1038/s41586-023-06818-7>`__
     - `slideseq_Puck34.h5ad <https://drive.google.com/uc?id=1n4eokj-r3NzoNxLGbms8LEvU3TKLjtU9>`__
   * - ``slideseq_Puck60.h5ad``
     - Slide-seq V2
     - adult mouse brain
     - 2, 4
     - `Langlieb et al. 2023 <https://doi.org/10.1038/s41586-023-06818-7>`__
     - `slideseq_Puck60.h5ad <https://drive.google.com/uc?id=1wCjQSjRYxqf3gHZjtkYA2Xg9-QSoOwtq>`__
   * - ``13months-disease-replicate_1.h5ad``
     - STARmap PLUS
     - TauPS2APP AD model hippocampus (13-month)
     - 7
     - `Zeng et al. 2023 <https://doi.org/10.1038/s41593-022-01251-x>`__
     - `13months-disease-replicate_1.h5ad <https://drive.google.com/uc?id=1NUGPliw0ZCFxHwgVjjH3Pkm05LjdphaV>`__
   * - ``section1.h5ad``
     - STARmap
     - healthy hippocampus crop
     - 7
     - `Shi et al. 2023 <https://doi.org/10.1038/s41586-023-06569-5>`__
     - `section1.h5ad <https://drive.google.com/uc?id=1dBPYDHLp7S6XDpq-t1GcNNCbyYsd4uoO>`__
   * - ``E15.5_E2S1.MOSTA.h5ad``
     - Stereo-seq (MOSTA)
     - E15.5 whole embryo
     - 9
     - `Chen et al. 2022 <https://doi.org/10.1016/j.cell.2022.04.003>`__
     - `E15.5_E2S1.MOSTA.h5ad <https://drive.google.com/uc?id=1yDxtjvZjZs-gekai7dk9zNjfybW4Nu8L>`__
   * - ``Mouse_embryo_square_016um.h5ad``
     - Visium HD (16 µm)
     - whole mouse embryo
     - 9
     - `10x Genomics <https://www.10xgenomics.com/datasets>`__
     - `Mouse_embryo_square_016um.h5ad <https://drive.google.com/uc?id=1gNJqse6LA4fwfDp_qwd4_be-rsrlvzI1>`__
   * - ``marmoset_striatum.h5ad``
     - STARmap
     - marmoset (*Callithrix jacchus*) forebrain/striatum
     - 12
     - this study, Extended Data Fig. 9
     - Available from the corresponding authors upon reasonable request; place at
       ``tutorial12_data/marmoset_striatum.h5ad``.


Tutorials 3, 10, and 11 use Tutorial 2's output — its universal **gene embedding**
(Tutorial 10) and joint **cell/tissue embeddings** (Tutorial 11) — so run Tutorial 2
first, or fetch its precomputed outputs. Tutorial 6 uses precomputed cell-contact source
data (``tutorial6/`` in the Drive folder).

Precomputed outputs (skip training)
--------------------------------------------------------------------------------

Tutorials with long training ship the finished outputs, so the analysis sections run on any
machine without a GPU:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Tutorial
     - Contents
     - Location
   * - Tutorial 7 (cross-condition)
     - final embeddings
     - shipped in the tutorial's ``precomputed_output/`` subfolder (fetched by each notebook's download cell)
   * - Tutorial 2 (cross-technology + deconvolution)
     - embeddings + Stage-B results + trained model
     - shipped in the tutorial's ``precomputed_output/`` subfolder (fetched by each notebook's download cell)
   * - Tutorial 9 (embryo)
     - embeddings + Stage-B results + trained model
     - shipped in the tutorial's ``precomputed_output/`` subfolder (fetched by each notebook's download cell)


Pretrained models
--------------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Model
     - Description
     - Download
   * - **molCCF**
     - Pretrained reference for Tutorials 5 and 12 and the FuseMap Agent. The paper
       reports 18.6M cells/spots across the full seven-atlas corpus; individual
       downloadable reference assets may cover a subset.
     - `Google Drive folder <https://drive.google.com/drive/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS>`__
   * - molCCF atlas data (Agent)
     - Region/cell-type lookups and atlas assets used by the FuseMap Agent
     - `Google Drive <https://drive.google.com/file/d/15LIkQTridS_ATwDy6dejIdzbMm39sEv3/view?usp=sharing>`__

Data requirements for your own data
--------------------------------------------------------------------------------

See :doc:`getting_started` — one ``.h5ad`` per section, spatial coordinates in
``obs['x']/obs['y']`` (or ``obsm['spatial']``), raw counts recommended, gene symbols as
``var_names``.
