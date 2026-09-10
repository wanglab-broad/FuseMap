# map_and_annotate_new_data
> Map an unknown mouse-brain h5ad onto the pretrained molCCF atlas, transfer cell types + tissue regions with uncertainties, and summarize composition per region.

## When to use
- The user brings a new spatial mouse-brain dataset (`.h5ad`) with no annotations and wants cell types and brain regions.
- **Boundary rule — map vs integrate**: `fusemap.map_to_reference` projects the query into a FROZEN reference space. This works well against molCCF-scale references (13.8M cells, many technologies). Against a small custom reference (a few sections), quality depends on how well the reference covers the query's platform and regions — if the mapped query does not mix with the reference in UMAP, **integrate the query jointly with the reference sections instead** (`fusemap.integrate` on all files together; slower, but always the more accurate option).

## Steps
1. Inspect the input. FuseMap picks up coordinates from `obs['x']/obs['y']`, `obs['col']/obs['row']`, or `obsm['spatial']` — anything else must be assigned manually (ask the user which columns are spatial if unclear).
```python
ad = sc.read_h5ad(f"{input_dir}/{fname}")
print(ad, "\n", ad.obs.columns.tolist())
if "x" not in ad.obs.columns and "col" not in ad.obs.columns and "spatial" not in ad.obsm:
    ad.obs["x"] = ad.obs["Raw_Slideseq_X"]   # <- user-confirmed coordinate columns
    ad.obs["y"] = ad.obs["Raw_Slideseq_Y"]
    ad.write_h5ad(f"{input_dir}/{fname}")
```
2. Map to molCCF — one call. One output subdirectory per input file; the reference is not retrained.
```python
fusemap.map_to_reference(input_dir, out_dir, MOLCCF_DIR)
qry = sc.read_h5ad(f"{out_dir}/{fname}/ad_celltype_embedding.h5ad")   # 64-dim latents in .X
```
3. Transfer cell types from the atlas. Concatenate a labeled atlas subsample with the query in the shared latent space, then one `transfer_labels` call (query rows have no label -> 'nan' -> excluded from training, predicted like everything else). Use `main_STARmap` for main types, `sub_STARmap` for subtypes.
```python
atlas = load_atlas("cell")
idx = np.random.default_rng(0).choice(atlas.n_obs, 300_000, replace=False)
ref = atlas[idx]
combo = anndata.AnnData(
    X=np.vstack([np.asarray(ref.X), np.asarray(qry.X)]),
    obs=pd.concat([ref.obs[["main_STARmap", "tissue_main"]].assign(role="reference"),
                   qry.obs[[]].assign(role="query")]),
)
res = fusemap.transfer_labels(combo, label_key="main_STARmap")
print(f"balanced held-out accuracy (reference): {res['test_accuracy']:.3f}")
qry.obs["cell_type"] = combo.obs["transfer_main_STARmap"].values[ref.n_obs:]
qry.obs["cell_type_uncertainty"] = combo.obs["transfer_main_STARmap_uncertainty"].values[ref.n_obs:]
```
4. Tissue regions — same combo (atlas `tissue_main` labels live on the same cells/latents):
```python
res_t = fusemap.transfer_labels(combo, label_key="tissue_main")
qry.obs["tissue_region"] = combo.obs["transfer_tissue_main"].values[ref.n_obs:]
qry.obs["tissue_region_uncertainty"] = combo.obs["transfer_tissue_main_uncertainty"].values[ref.n_obs:]
```
5. Check mixing (this is the map-vs-integrate decision point) plus a label-free spatial QC:
```python
vis = combo[np.random.default_rng(1).choice(combo.n_obs, min(60_000, combo.n_obs), replace=False)].copy()
fm_pl.umap(vis, by="role")   # query should interleave with atlas cells
print("spatial coherence (cell types):", round(fm_metrics.spatial_coherence(qry, "cell_type"), 3))
```
6. Summary stats per region:
```python
summary = qry.obs.groupby("tissue_region").agg(
    n_cells=("cell_type", "size"),
    mean_ct_uncertainty=("cell_type_uncertainty", "mean"))
top3 = (qry.obs.groupby(["tissue_region", "cell_type"]).size().rename("n")
        .reset_index().sort_values("n", ascending=False).groupby("tissue_region").head(3))
print(summary.sort_values("n_cells", ascending=False)); print(top3)
```
7. Plot the transferred annotations in space:
```python
x, y = pd.to_numeric(qry.obs["x"]), pd.to_numeric(qry.obs["y"])
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, key in zip(axes, ["cell_type", "tissue_region"]):
    codes = pd.Categorical(qry.obs[key].astype(str)).codes
    ax.scatter(x, y, c=codes, cmap="tab20", s=2)
    ax.set_title(f"transferred {key}"); ax.set_aspect("equal"); ax.axis("off")
plt.show()
```

## Verification checks
```python
assert qry.X.shape[1] == 64
for c in ["cell_type", "cell_type_uncertainty", "tissue_region", "tissue_region_uncertainty"]:
    assert c in qry.obs.columns, c
assert res["test_accuracy"] > 0.7      # main-level labels; fine subtypes (sub_STARmap) can be ~0.5
assert res_t["test_accuracy"] > 0.7    # tissue regions typically 0.85+
assert qry.obs["cell_type_uncertainty"].between(0, 1).all()
assert qry.obs["cell_type"].nunique() > 1   # collapse to one label = failed transfer
```

## Pitfalls
- `test_accuracy` is measured on held-out REFERENCE cells — it says the classifier learned the atlas, not that the query annotations are right. Judge the query by uncertainty, UMAP mixing, and `fm_metrics.spatial_coherence`.
- Outputs land in `out_dir/<input_file_name>/ad_celltype_embedding.h5ad` (one subdirectory per file), not directly in `out_dir`.
- Do not train the transfer on all 10.4M atlas cells — a 200-500k subsample is plenty and far faster.
- High-uncertainty populations often mean cell types absent from the reference; for downstream stats filter `obs["cell_type_uncertainty"] < 0.5` rather than trusting every label.
- Small custom reference + query not mixing in UMAP -> stop and re-run as a joint integration (see `integrate_multi_sample`).
- Coordinates in nonstandard columns must be fixed and re-saved into the h5ad BEFORE calling `map_to_reference`, or it raises on that file.
