# bead_deconvolution_readout
> Interpret Stage-B deconvolution outputs: canonical vs nodeconv files, the per-bead mixture matrix pi (entropy = mixedness, argmax = dominant archetype), and semantic naming of archetypes.

## When to use
- A finished integration run declared `bead_files`/`sig_ref` and the user asks what the beads are made of, how mixed they are, or whether deconvolution worked.
- Stage-B was skipped at training time and must be run now: `fusemap.deconvolve_beads(out_dir, data_dir, bead_files="slideseq", sig_ref="starmap")` — then read out as below.

## Steps
1. Know the file map before touching anything:
- `ad_celltype_embedding.h5ad` / `ad_tissueregion_embedding.h5ad` — CANONICAL, bead rows already deconvolved (single-cell rows untouched)
- `ad_*_embedding_nodeconv.h5ad` — pre-deconvolution copies, diagnostics only
- `ad_*_embedding_stageB.h5ad` — identical to canonical (compatibility names)
- `stageB_pi.npz` — per-bead mixture over archetypes learned from the single-cell reference
2. Load pi; one key per declared bead file (`pi__<file-stem>`):
```python
z = np.load(f"{out_dir}/stageB_pi.npz", allow_pickle=True)
print(z.files)                                  # e.g. ['pi__slideseq_Puck60', 'pi__stereoseq_mousebrain']
pi = z["pi__slideseq_Puck60"]                   # (n_beads, n_archetypes), rows sum to 1
print("pi:", pi.shape)
```
3. The two readouts — mixedness and dominant identity:
```python
entropy = -(pi * np.log(pi + 1e-12)).sum(1)     # 0 = pure bead ... log(K) = uniform mixture
dominant = pi.argmax(1)                         # hard assignment to the strongest archetype
```
4. Align with the bead rows of the canonical embedding (pi rows follow that file's row order) and plot both readouts in space:
```python
ad_bead = sc.read_h5ad(f"{out_dir}/ad_celltype_embedding.h5ad")
bead_obs = ad_bead.obs[ad_bead.obs["file_name"].str.contains("slideseq")]
assert len(bead_obs) == pi.shape[0]
bx, by = pd.to_numeric(bead_obs["x"]), pd.to_numeric(bead_obs["y"])
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
axes[0].hist(entropy, bins=50); axes[0].set_title("per-bead mixing entropy")
axes[1].scatter(bx, by, c=entropy, cmap="viridis", s=2); axes[1].set_title("entropy in space")
axes[2].scatter(bx, by, c=dominant, cmap="tab20", s=2); axes[2].set_title("dominant archetype")
for ax in axes[1:]: ax.set_aspect("equal"); ax.axis("off")
plt.show()
```
Expect low entropy inside homogeneous structures (e.g. DG granule layer) and high entropy at boundaries/neuropil.
5. Semantic naming of archetypes via reference annotations: transfer the reference label (any label column present in the embedding obs, e.g. the sig_ref's annotations) onto the deconvolved beads, then take the majority label among beads each archetype dominates:
```python
res = fusemap.transfer_labels(ad_bead, "gt_cell_type_main")     # reference-labeled cells train it
lab = ad_bead.obs.loc[bead_obs.index, "transfer_gt_cell_type_main"].astype(str).values
names = (pd.DataFrame({"arch": dominant, "label": lab})
           .groupby("arch")["label"]
           .agg(lambda s: f"{s.value_counts().idxmax()} ({s.value_counts(normalize=True).iloc[0]:.0%})"))
print(names)   # archetype -> majority transferred cell type (+ purity)
```
6. Did deconvolution help? Compare batch mixing before vs after:
```python
for tag, path in [("before (nodeconv)", f"{out_dir}/ad_tissueregion_embedding_nodeconv.h5ad"),
                  ("after  (canonical)", f"{out_dir}/ad_tissueregion_embedding.h5ad")]:
    ad_t = sc.read_h5ad(path)
    if ad_t.n_obs > 60000: sc.pp.subsample(ad_t, n_obs=60000, random_state=0)
    print(tag, "iLISI:", round(fm_metrics.ilisi(ad_t, batch_key="file_name"), 3))
    fm_pl.umap(ad_t, by="file_name")
```
Before: bead rows sit apart from single-cell sections; after: they interleave (higher iLISI).

## Verification checks
```python
assert np.allclose(pi.sum(1), 1, atol=1e-3)                 # rows are proper mixtures
assert (pi >= 0).all()
assert len(bead_obs) == pi.shape[0]                          # bead-row alignment
assert entropy.min() >= 0 and entropy.max() <= np.log(pi.shape[1]) + 1e-6
sc.read_h5ad(f"{out_dir}/ad_celltype_embedding_nodeconv.h5ad")  # pre-deconv copy exists
```

## Pitfalls
- The canonical files are ALREADY deconvolved — analyses of "the beads" should use them, not `*_nodeconv.h5ad`; re-running `deconvolve_beads` on a deconvolved run is unnecessary.
- pi keys follow the declared file stems (`pi__<stem>`); list `z.files` instead of guessing.
- pi row order = that file's bead-row order in the embedding (original input order). Do not sort/filter `bead_obs` before aligning, or colors and names silently shuffle.
- Archetypes are data-driven prototypes (typically tens of them, e.g. 40), NOT 1:1 cell types — the dominant archetype is not a hard cell-type call; step 5's majority label plus its purity is the honest summary.
- Entropy scales with the number of archetypes (max = log K) — compare entropies within a run, never across runs with different K.
- Single-cell (sig_ref/imaging) rows are unchanged by Stage-B; only declared bead files were rebuilt.
- Step 5 needs a reference label column in the embedding obs (carried from the input files or via `keep_celltype`); if absent, transfer atlas labels through a combo with `load_atlas("cell")` first (see `map_and_annotate_new_data`).
