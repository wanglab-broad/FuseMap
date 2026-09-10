# integrate_multi_sample
> Integrate a folder of spatial h5ad sections (mixed platforms OK) into shared FuseMap embeddings with one call, then QC with iLISI, UMAP, and spatial clusters.

## When to use
- Multiple sections/samples — same or different technologies (STARmap, MERFISH, Slide-seq, Stereo-seq, Visium, ...) — to bring into one latent space.
- Also the fallback when mapping to a small custom reference failed the mixing check (integrate query + reference jointly).

## Steps
1. Check every file loads and has coordinates; report gene panels. `fusemap.read_input_folder` applies the exact coordinate rules training will use (`obs x/y` -> `col/row` -> `obsm['spatial']`) and raises an informative error on files needing manual fixes.
```python
sections = fusemap.read_input_folder(input_dir)
panels = []
for s in sections:
    print(s.obs["file_name"].iloc[0], s.shape)
    panels.append(set(g.upper() for g in s.var_names))
print("genes shared by all sections:", len(set.intersection(*panels)))
```
2. **Ask the user which files are bead/spot resolution.** Sequencing-based platforms (Slide-seq beads, Stereo-seq bins, Visium spots) measure MIXTURES of cells; imaging platforms are single-cell. Roles are USER-DECLARED, never auto-detected — do not guess from file names or gene counts. Also ask which single-cell file(s) serve as the deconvolution signature reference (`sig_ref`). If everything is single-cell imaging data, skip the bead arguments entirely.
3. Integrate — one call. With beads declared, Stage-B deconvolution runs AUTOMATICALLY after training and the canonical outputs contain the deconvolved bead rows.
```python
fusemap.integrate(
    input_dir, out_dir,
    keep_celltype="celltype_anno",          # existing obs label column to carry through ("" if none)
    keep_tissueregion="tissueregion_anno",
    bead_files="slideseq_Puck60,stereoseq", # ONLY as declared by the user; omit if all single-cell
    sig_ref="starmap",                      # required whenever bead_files is set
)
```
Re-running the same call resumes from the last checkpoint automatically.
4. Orient in the outputs:
- `ad_celltype_embedding.h5ad` / `ad_tissueregion_embedding.h5ad` / `ad_gene_embedding.h5ad` — canonical outputs; bead rows are ALREADY DECONVOLVED when beads were declared
- `ad_*_embedding_nodeconv.h5ad` — pre-deconvolution copies (diagnostics only)
- `stageB_pi.npz` — per-bead mixture matrix pi (see `bead_deconvolution_readout`)
5. QC 1 — batch mixing (iLISI in [0,1]: 0 = separated, 1 = perfectly mixed) + UMAP:
```python
ad_cell = sc.read_h5ad(f"{out_dir}/ad_celltype_embedding.h5ad")
if ad_cell.n_obs > 60000:
    sc.pp.subsample(ad_cell, n_obs=60000, random_state=0)
print("iLISI (cell embedding):", round(fm_metrics.ilisi(ad_cell, batch_key="file_name"), 3))
fm_pl.umap(ad_cell, by="file_name")
```
6. QC 2 — joint spatial clusters: cluster the tissue embedding once, plot back into each section's coordinates; matching anatomical structures should get matching cluster colors across sections and platforms.
```python
ad_tissue = sc.read_h5ad(f"{out_dir}/ad_tissueregion_embedding.h5ad")
if ad_tissue.n_obs > 60000:
    sc.pp.subsample(ad_tissue, n_obs=60000, random_state=0)
print("iLISI (tissue embedding):", round(fm_metrics.ilisi(ad_tissue, batch_key="file_name"), 3))
fm_pl.spatial_clusters(ad_tissue, resolution=0.5)
```
7. If labels were carried through (`keep_celltype`), annotate the unlabeled sections with `fusemap.transfer_labels(ad_cell, "<label_col>")` — adds `transfer_<label_col>` and `..._uncertainty` to `obs`, returns balanced held-out accuracy.

## Verification checks
```python
ads = {f: sc.read_h5ad(f"{out_dir}/ad_{f}_embedding.h5ad")   # raises if training did not finish
       for f in ["celltype", "tissueregion", "gene"]}
assert ads["celltype"].X.shape[1] == 64
assert {"file_name", "x", "y"} <= set(ads["celltype"].obs.columns)
assert ads["celltype"].obs["file_name"].nunique() == len(sections)
if declared_beads:  # bead_files was passed
    z = np.load(f"{out_dir}/stageB_pi.npz", allow_pickle=True)
    pi = z[z.files[0]]
    assert np.allclose(pi.sum(1), 1, atol=1e-3)          # rows are proper mixtures
    sc.read_h5ad(f"{out_dir}/ad_celltype_embedding_nodeconv.h5ad")  # pre-deconv copy kept
```

## Pitfalls
- Never guess bead roles — a wrong declaration silently "deconvolves" single-cell data, or leaves real beads posing as cells. Always ask the user.
- `bead_files` without `sig_ref` raises ValueError; both are comma-separated FILE-NAME SUBSTRINGS and each must match exactly one input file.
- When beads were declared, the canonical embeddings are already deconvolved — do not treat `*_nodeconv.h5ad` as the main result, and do not deconvolve again.
- `keep_celltype`/`keep_tissueregion` name obs columns in the INPUT files; sections lacking that column get `'nan'` labels (fine — `transfer_labels` fills them later).
- Subsample (~60k) before neighbors/UMAP/leiden on large runs; the full embedding on disk stays untouched.
- Bead sections separating from single-cell sections in the `*_nodeconv` files is EXPECTED — judge mixing on the canonical (deconvolved) files.
- Batch column: `file_name` is always present; outputs also carry `batch` (= sample0, sample1, ...) and `name` (= section0, ...).
