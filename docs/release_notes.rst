Software and manuscript versions
================================================================================

This checkout is FuseMap **1.2.0**. It adds bead deconvolution to reference mapping:
query adaptation and mixture fitting reuse an unchanged pretrained reference.
The mixture solver is shared with integration Stage-B. Reference expression
signatures can be prepared once and reused without reloading reference expression
data. See Tutorial 2.1 and the API reference for the new arguments and readouts.

Tutorials 2.1 and 2.2 install the ``v1.2.0`` Git source tag to access this feature.
Other tutorials retain their existing PyPI installation. See ``CHANGELOG.md``
and ``RELEASE_VALIDATION.md`` for scope and validation.

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

Tutorial 3.4 illustrates module-balanced panel selection using **Leiden** gene
modules; the paper uses **WGCNA**. Tutorial 3.3 illustrates within-region and
within-cell-type clustering on a small example, with imputed markers as supporting
model evidence. Reproducing the 146-region atlas also requires the paper's
cross-section, cross-platform and anatomical validation. Tutorial 2.3 demonstrates
symbol-matched transfer to marmoset with data available upon request; its exploratory
clustering and label granularity differ from Extended Data Fig. 9.

Agent interfaces
--------------------------------------------------------------------------------

``app.py`` implements the paper's classic architecture (LangChain 0.3).
``app_v2.py`` is a later CodeAct interface (LangChain 1.x), using a separate
controller environment and a registered FuseMap kernel. Its behavior should be
validated separately from the paper's multi-agent experiments. See :doc:`agent/index`.
