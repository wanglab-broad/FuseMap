Software and manuscript versions
================================================================================

This checkout is FuseMap **1.1.3**. It fixes reference-mapping validation after
the DDP refactor, completes Tutorials 10–12, and documents a separate environment
for Agent v2. See the repository's ``CHANGELOG.md`` for release and validation details.

Manuscript workflows
--------------------------------------------------------------------------------

The current manuscript is *An agentic AI spatial molecular foundation model of the
brain*. It describes the shared gene/cell/spatial representation, molCCF,
reference mapping, gene imputation, and the classic Supervisor + three-agent
interface. It identifies a frozen source and analysis archive at
`Zenodo <https://doi.org/10.5281/zenodo.22103552>`__. Use that archive and its
analysis-specific settings for reproducing the paper.

The current software also includes pairwise-gated MNN anchors (default weight
0.3), Stage-B post-processing, parallel data loading, and experimental DDP.
Disabling anchors removes that loss; it alone does not establish equivalence
with every historical analysis. Stage-B proportions are approximate, as discussed
in the manuscript; they are not calibrated cell counts.

Tutorial 10 illustrates module-balanced panel selection using **Leiden** gene
modules; the paper uses **WGCNA**. Tutorial 11 illustrates within-region and
within-cell-type clustering on a small example, with imputed markers as supporting
model evidence. Reproducing the 146-region atlas also requires the paper's
cross-section, cross-platform and anatomical validation. Tutorial 12 demonstrates
symbol-matched transfer to marmoset with data available upon request; its exploratory
clustering and label granularity differ from Extended Data Fig. 9.

Agent interfaces
--------------------------------------------------------------------------------

``app.py`` implements the paper's classic architecture (LangChain 0.3).
``app_v2.py`` is a later CodeAct interface (LangChain 1.x), using a separate
controller environment and a registered FuseMap kernel. Its behavior should be
validated separately from the paper's multi-agent experiments. See :doc:`agent/index`.
