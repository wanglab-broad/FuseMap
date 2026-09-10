# impute_unmeasured_genes
> Impute transcriptome-wide expression for any section via cell latents x gene latents; ALWAYS validate on a measured gene (Pearson r) before trusting unmeasured ones.

## When to use
- A section's panel lacks a gene of interest (e.g. a 1,022-gene STARmap panel), but a FuseMap run or molCCF ties it into a universal gene embedding (~26k genes).
- Works on any embedding AnnData with 64-dim latents in `.X`: integration outputs, mapped queries, atlas cells.

## Steps
1. Load cell latents and the MATCHING gene embedding (same model!):
```python
ad_cell = sc.read_h5ad(f"{out_dir}/ad_celltype_embedding.h5ad")
gene_embed = sc.read_h5ad(f"{out_dir}/ad_gene_embedding.h5ad")
print("cells:", ad_cell.shape, " genes in universal embedding:", gene_embed.shape[0])
```
For molCCF-mapped queries or atlas cells, omit `gene_embedding=` below — `impute_gene_on` defaults to the molCCF universal gene embedding (`load_atlas('gene')`).
2. Select the target section's rows (embedding rows preserve each input file's cell order):
```python
mask = ad_cell.obs["file_name"].str.contains("starmap").values
sub = ad_cell[mask]
```
3. Impute — one matrix product (E_hat = Z_c @ Z_g^T; the helper uppercases gene names and returns a DataFrame):
```python
genes = ["SLC17A7", "PLP1", "HPCA"]      # include >=1 gene MEASURED in this section for validation
missing = [g for g in genes if g.upper() not in gene_embed.obs_names]
assert not missing, f"not in gene embedding: {missing}"
imputed = impute_gene_on(sub, genes, gene_embedding=gene_embed)
print(imputed.describe().loc[["mean", "std"]])
```
4. Validate on the measured gene BEFORE anything else. Process the raw data the way FuseMap does (normalize, log1p, scale), then correlate:
```python
raw = sc.read_h5ad(f"{input_dir}/starmap.h5ad")
sc.pp.normalize_total(raw); sc.pp.log1p(raw)
sc.pp.scale(raw, zero_center=False, max_value=10)
name_map = {g.upper(): g for g in raw.var_names}
mv = raw[:, name_map["SLC17A7"]].X
measured = np.asarray(mv.todense()).ravel() if hasattr(mv, "todense") else np.asarray(mv).ravel()
r = np.corrcoef(measured, imputed["SLC17A7"])[0, 1]
print(f"SLC17A7 measured vs imputed: pearson r = {r:.2f}")
```
5. Side-by-side spatial plots — measured vs imputed, then the unmeasured genes:
```python
sx, sy = pd.to_numeric(sub.obs["x"]), pd.to_numeric(sub.obs["y"])
panels = [("SLC17A7 measured", measured)] + [(f"{g} imputed", imputed[g]) for g in genes]
fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.5))
for ax, (title, v) in zip(axes, panels):
    sca = ax.scatter(sx, sy, c=v, cmap="Purples", s=1)
    ax.set_title(title); ax.set_aspect("equal"); ax.axis("off"); fig.colorbar(sca, ax=ax)
plt.show()
```
6. For unmeasured genes there is no ground truth on this section — sanity-check the SPATIAL PATTERN against known anatomy (e.g. PLP1 = white-matter tracts, HPCA = hippocampal CA fields), or against another section whose panel does measure the gene.

## Verification checks
```python
assert imputed.shape == (int(mask.sum()), len(genes))
assert not imputed.isna().any().any()
assert measured.shape[0] == imputed.shape[0]     # raw <-> embedding row alignment
assert r > 0.3   # validation gate: below this, do NOT report unmeasured genes from this model/section
# reference point: the validated tutorial-2 model scores r = 0.835 on SLC17A7 (STARmap)
```

## Pitfalls
- Validation is per model AND per section: a good r on one section does not certify another. Always run step 4 on the section you report.
- Use the gene embedding from the SAME model as the cell latents — mixing molCCF gene latents with a custom run's cell latents (or vice versa) gives garbage.
- Gene names in the embedding are UPPERCASE (`gene_embed.obs_names`); check membership before indexing, otherwise KeyError.
- Imputed values are dot products — RELATIVE, not counts. Compare patterns and correlations; never read magnitudes as expression levels.
- Row alignment between the raw file and the embedding assumes the cell order was not modified — do not filter/reorder one side only.
- Pick a spatially structured measured gene for the validation gate: strong markers validate well, while housekeeping/unpatterned genes correlate poorly even when the model is fine.
