# Changelog

## 1.2.0 — frozen-reference bead mapping, 2026-09-13

`map_to_reference` now accepts declared bead datasets and performs Stage-B after
query adaptation. The pretrained reference checkpoint and embeddings remain
unchanged. Cell and tissue embeddings are rebuilt from bead mixtures over fixed
reference archetypes; original mapped embeddings are preserved as `_nodeconv`.

- Add `bead_files`, `sig_ref`, `reference_data_folder_path`,
  `reference_signatures_path`, and `entropy_weight` to the mapping API. CLI mapping
  uses the same API and accepts the reference-data/signature options.
- Add `prepare_reference_signatures`: construct pooled expression signatures from
  explicitly selected single-cell reference sections, aligning observations by ID
  and genes by name. A saved artifact can be reused without reference raw files;
  its checkpoint fingerprint prevents using it with a different trained model.
- Share the Stage-B mixture solver between integration and mapping. Fit bead
  weights plus per-gene platform terms with frozen signatures. Mask genes outside
  the reference panel and bound GPU expression memory by chunking.
- Save mixtures in `stageB_pi.npz` and mapping embeddings' `obsm['stageB_pi']`,
  together with observation IDs, reconstruction errors and reference provenance.
- Update Tutorial 2.1 to run bead deconvolution and explain mixture readouts.
  Clear its earlier outputs, which predate this model path. Document the required
  signature artifact for molCCF bead deconvolution in Tutorial 2.2.
- Match the integration Stage-B objective, without claiming results identical to
  joint retraining or interpreting archetype weights as calibrated cell counts.

Validation is recorded in `RELEASE_VALIDATION.md`. The source tag is `v1.2.0`;
creating source/build artifacts does not itself imply a PyPI upload.

## 1.1.3 — local release preparation, 2026-09-11

Reference mapping could raise `NameError: dist_hooks is not defined` during
validation following the DDP refactor. `map_model` now defaults `dist_hooks` to
`None`, so ordinary mapping reaches scheduling and checkpoint selection.
The regression test executes that path with CPU optimizers and a small model,
including optional synchronization hooks.

- Complete Tutorials 10–12 and their navigation/data entries. Tutorial 10
  explicitly uses Leiden as a teaching adaptation of the paper's WGCNA panel
  workflow. Tutorial 11 distinguishes model-derived marker evidence from
  independent anatomical validation. Tutorial 12 documents data available upon
  request, replaces its placeholder download, and interprets the saved mixed
  domains and anatomically unexpected predictions.
- Keep classic LangChain 0.3 dependencies in the scientific environment. Agent
  v2 has a separate requirements file and setup instructions for its LangChain
  1.x controller and registered scientific kernel. Kernel startup now reports
  bootstrap errors; SQLite checkpoint directories are created on first use.
- Replace the obsolete requirements freeze (including a self-pin to FuseMap
  0.0.1) with installation of the checkout. Bound AnnData below 0.12 for Scanpy
  1.9.3 compatibility; provide `tutorials` and `agent` extras. Mark the Python 3.7
  environment YAML as historical.
- Update the manuscript title, retain the earlier preprint citation as such,
  document method/version differences, and use the installed Python API in
  quick starts. Add public API documentation and modern build metadata.
- Version the corrected artifacts as 1.1.3; keep 1.1.2 artifacts for comparison.
  Build products and Agent runtime state are excluded from version control.

### Scope

The current manuscript is *An agentic AI spatial molecular foundation model of
the brain*. Its frozen code and analysis archive is identified at
https://doi.org/10.5281/zenodo.22103552. Current software enhancements and teaching
adaptations are described in `docs/release_notes.rst`; they do not replace the
paper's analysis-specific configurations.

Notebook analysis outputs are retained from earlier runs. Modified setup cells
are cleared; release preparation does not claim fresh execution of all tutorials,
full-atlas retraining, live LLM-provider validation, or verification of remote
data downloads. This checkout and its local build are not a PyPI upload.

### Validation

Validation results are recorded in `RELEASE_VALIDATION.md` after the checks run.
