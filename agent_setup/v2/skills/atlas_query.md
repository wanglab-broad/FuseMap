# atlas_query
> Answer questions about the molCCF atlas itself: regions expressing a gene, cell-type composition of a region, section-wise and 3D visualization of 10.4M cells.

## When to use
- Questions about the ATLAS, not about user data: "which regions express Sst?", "what cell types make up the striatum?", "show me section 30", "where are activated microglia in 3D?".

## Steps
1. Vocabulary first — resolve the user's wording to atlas labels before touching the big file:
```python
regions = atlas_lookup("region")   # Region_ID + Description (+ color)
types = atlas_lookup("type")       # Symbol + Description with marker genes
print(regions[regions["Description"].str.contains("stria", case=False)])
print(types[types["Description"].str.contains("microglia", case=False)])
```
2. Load atlas cells (lazy, cached; 10.4M x 64 latents). Key obs columns: `main_STARmap`/`sub_STARmap` (cell type/subtype), `tissue_main`/`tissue_sub` (region), `ap_order` (anterior-posterior section index), `global_x/global_y/global_z` (3D CCF-space coordinates).
```python
atlas = load_atlas("cell")
print(atlas.shape, "\n", atlas.obs["tissue_main"].value_counts().head(10))
```
3. "Which regions express gene X?" — impute on a subsample (never all 10.4M at once), then average per region:
```python
sub = atlas[np.random.default_rng(0).choice(atlas.n_obs, 500_000, replace=False)]
expr = impute_gene_on(sub, "PLP1")["PLP1"]           # uses load_atlas('gene') internally
by_region = expr.groupby(sub.obs["tissue_main"].values).mean().sort_values(ascending=False)
print(by_region.head(10))
by_type = expr.groupby(sub.obs["main_STARmap"].values).mean().sort_values(ascending=False)
print(by_type.head(10))
```
4. "Cell-type composition of region Y?" — straight value_counts on the full obs (cheap, no X involved):
```python
m = atlas.obs["tissue_main"] == "STR"                # exact ID from atlas_lookup("region")
comp = atlas.obs.loc[m, "main_STARmap"].value_counts(normalize=True)
print(f"{int(m.sum()):,} cells in region"); print(comp.head(15))
comp.head(15)[::-1].plot.barh(figsize=(7, 6), title="STR composition"); plt.tight_layout(); plt.show()
```
5. Section-wise visualization — pick sections by `ap_order`, plot `global_x/global_y`:
```python
print(sorted(atlas.obs["ap_order"].unique())[:10], "...")   # available sections
sec = atlas[atlas.obs["ap_order"] == 30]
codes = pd.Categorical(sec.obs["tissue_main"].astype(str)).codes
plt.figure(figsize=(8, 7))
plt.scatter(sec.obs["global_x"], sec.obs["global_y"], c=codes, cmap="tab20", s=0.5)
plt.gca().set_aspect("equal"); plt.axis("off"); plt.title("section ap_order=30, tissue_main")
plt.show()
```
Overlay imputed expression the same way: `c=impute_gene_on(sec, "PLP1")["PLP1"], cmap="Purples"`.
6. 3D view — subsample hard, highlight the population of interest against a gray shadow of the brain:
```python
pick = atlas[np.random.default_rng(0).choice(atlas.n_obs, 300_000, replace=False)]
hit = pick.obs["sub_STARmap"].str.contains("Microglia", case=False).values
ax = plt.figure(figsize=(10, 8)).add_subplot(projection="3d")
ax.scatter(pick.obs["global_x"][~hit], pick.obs["global_y"][~hit], pick.obs["global_z"][~hit],
           s=0.1, c="lightgray", alpha=0.15)
ax.scatter(pick.obs["global_x"][hit], pick.obs["global_y"][hit], pick.obs["global_z"][hit],
           s=0.6, c="crimson")
ax.set_axis_off(); plt.show()
```
7. Multi-section panels (gene gradient along the AP axis): loop a few `ap_order` values, one subplot each, same recipe as step 5.

## Verification checks
```python
assert atlas.shape == (10405193, 64)
assert {"ap_order", "sub_STARmap", "main_STARmap", "tissue_main", "tissue_sub",
        "global_x", "global_y", "global_z"} <= set(atlas.obs.columns)
assert "PLP1" in load_atlas("gene").obs_names       # gene exists before imputing
assert int(m.sum()) > 0                             # region filter actually matched cells
```

## Pitfalls
- Region/type values are IDs (`STR`, `CTX_1`, `VLM_2`, ...) — match the user's plain-English request through `atlas_lookup` Descriptions first; an `==` filter on a guessed name silently returns 0 cells.
- Never `impute_gene_on` the full 10.4M cells for many genes at once — subsample (~500k) or go section-by-section.
- Gene names are UPPERCASE in the gene embedding; `impute_gene_on` uppercases for you, but membership checks must use uppercase too.
- `ap_order` is a section INDEX along the anterior-posterior axis, not a physical coordinate; use `global_z` for physical depth.
- Imputed atlas expression is relative (dot products) — rank regions/types, don't quote magnitudes.
- The kernel's matplotlib is non-interactive (Agg): figures are saved as artifact files, not shown inline — tell the user where they landed.
- `main_Allen`/`sub_Allen`/`cluster_Allen` columns offer the Allen taxonomy as an alternative to the STARmap vocabulary.
