# FuseMap 1.2.0 validation

Date: 2026-09-13. CPU validation in Python 3.10 with the existing FuseMap scientific
environment. Tests verify implementation behavior; they do not establish biological
composition accuracy or equivalence to joint retraining.

| Check | Result |
|---|---|
| Source regression suite | 23 passed: existing mapping/Colab checks, numerical Stage-B equivalence, synthetic mixture recovery, reference/gene/observation alignment, multiple bead queries, cached signatures, immutable references, CLI, and integration Stage-B readouts |
| Installed wheel | Same 23 tests passed with imports confirmed under `/tmp/fusemap-wheel-120`, outside the source checkout |
| Package artifacts | Source distribution and wheel built successfully; `twine check` passed; all 26 packaged Python files match source byte for byte |
| Notebooks | All 12 notebook schemas validated; 128 code cells compile after IPython syntax transformation |
| Real-data mapping | 512 randomly sampled Slide-seq Puck60 beads, actual default query training, followed by Stage-B against the full saved MERFISH + STARmap reference (88,295 cells) |
| Real-data outputs | 40 archetypes; 1,471 reference-covered query genes; finite nonnegative weights summing to one; canonical cell embedding equals `pi @ archetype_centers`; original mapping embedding preserved |
| Reference preservation | All 20 files in the pretrained reference directory had identical SHA-256 hashes before and after the real-data run |
| Real-data reconstruction | Mean MSE 0.410203 on the covered, preprocessed expression genes; a reconstruction diagnostic, not a composition accuracy metric |
| Real-data runtime | Approximately 54 seconds with 4 CPU threads and `FUSEMAP_LOADER_WORKERS=0`; the latter avoids worker socket restrictions in the validation sandbox |
| Documentation | Strict offline Sphinx HTML build (`-W --keep-going`) passed without warnings |

The real-data report, hashes and reproduction script are saved locally under
`/ewsc/yhe/FuseMap-revision3/mapping_deconvolution_validation_20260913/`.
This is a real query subset validation, not a fresh execution of the full tutorial
or a GPU validation. The source tag is `v1.2.0`; Tutorials 2.1 and 2.2 install this
tag so the new API does not depend on a PyPI upload. PyPI publication is separate.

Reproduce source checks with:

```bash
python -m pytest tests -q
FUSEMAP_DOCS_OFFLINE=1 python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m build --no-isolation
python -m twine check dist/fusemap-1.2.0*
```

# FuseMap 1.1.3 release validation (historical)

Date: 2026-09-11. These checks apply to the local 1.1.3 working tree and builds.
They do not certify a GitHub, PyPI, Read the Docs, or Zenodo deployment.

| Check | Result |
|---|---|
| Mapping validation/checkpoint regression from source | 2 passed (ordinary mapping and optional synchronization hooks) |
| Same regression against an installed wheel, outside the source tree | 2 passed with Torch 2.0.1+cu117, Scanpy 1.9.3, AnnData 0.11.4, Python 3.10 |
| Installed import location/version | Confirmed installation under a temporary virtual environment, FuseMap 1.1.3 |
| Agent v2 dependency resolution | `pip install --dry-run -r requirements-agent-v2.txt` passed against the controller environment |
| Wheel plus tutorial/Agent extra dependencies | Online pip dry-run resolved successfully against the pinned tutorial environment |
| Agent v2 scientific kernel smoke check | Registered a temporary kernel; imported FuseMap 1.1.3, retained a variable across calls, saved a plot, and returned a Python error |
| Notebook validation | All 12 notebooks: valid schema and 115 code cells compile after IPython syntax transformation |
| Sphinx HTML documentation | Strict offline build (`-W --keep-going`) passed without warnings |
| Release archives | Source distribution and wheel built; `twine check` passed |
| Wheel/source consistency | All 24 package Python files match the working tree byte for byte |
| Patch whitespace | `git diff --check` passed |

## Reproduce the main checks

From the repository root with the appropriate dependencies installed:

```bash
python -m pytest tests/test_map_validation.py -q
FUSEMAP_DOCS_OFFLINE=1 python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m build
python -m twine check dist/fusemap-1.1.3*
```

Install the wheel in a separate environment and run a copy of the tests from
outside the repository to verify the packaged code rather than importing the
checkout. `dist/SHA256SUMS-1.1.3` records the final archive checksums.

## Environment and scope

The source regression and kernel smoke check used the existing scientific runtime
(Torch 2.6.0+cu124); the installed-wheel regression separately checked the declared
Torch 2.0.1 stack. Temporary test environments and kernel registration were created
under `/tmp`; existing Conda environments were not changed. The tutorial environment
had an incomplete optional Dask installation, repaired only in the temporary test
environment before the installed-wheel checks. Existing library deprecation warnings
remain; neither mapping regression run reported a test failure after setup.

The first online Sphinx build could not retrieve external intersphinx inventories
in the restricted network. Local notebook links were fixed, and the complete strict
build passed with only external inventories disabled via `FUSEMAP_DOCS_OFFLINE=1`.

The mapping tests execute CPU optimizers, validation, learning-rate scheduling and
checkpoint promotion with a small model, replacing graph preparation and loss
construction. They are regression checks, not a scientific integration benchmark.
The kernel smoke check makes no LLM-provider calls and does not load the full atlas.

Existing notebook analysis outputs were retained. Edited setup cells were cleared;
no fresh full-atlas training, end-to-end tutorial rerun, live LLM analysis, or external
reference/data download verification is claimed. Tutorial 12 follows the manuscript's
marmoset data availability-on-request statement. The manuscript and frozen local
validation snapshots were not modified.
