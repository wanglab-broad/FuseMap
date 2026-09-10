# subtype_shift_analysis
> Compare per-subtype composition between conditions/ages after mapping — region-stratified proportions with per-section bootstrap CIs (template: aging homeostatic vs activated microglia).

## When to use
- Two or more groups of sections (young vs old, healthy vs disease, treated vs control) have been annotated THROUGH THE SAME REFERENCE (each mapped to molCCF per `map_and_annotate_new_data` with `label_key="sub_STARmap"`, or one joint integration + transfer) — so subtype labels are comparable — and the user asks what shifted.
- The paper's aging analysis is the template: total microglia stay put while the homeostatic/activated balance diverges with age, region by region.

## Steps
1. Assemble one per-cell table. Group membership is USER-DECLARED (ask if unclear); filter high-uncertainty cells and report how much was dropped per section.
```python
section_groups = {"puck_young1.h5ad": "young", "puck_young2.h5ad": "young",
                  "puck_old1.h5ad": "old", "puck_old2.h5ad": "old"}   # USER-declared
frames = []
for fn, grp in section_groups.items():
    q = sc.read_h5ad(f"{out_dir}/{fn}/ad_celltype_embedding.h5ad")   # annotated per map_and_annotate_new_data
    q.obs = q.obs.rename(columns={"cell_type": "subtype", "cell_type_uncertainty": "subtype_uncertainty",
                                  "tissue_region": "region"})       # skill-1 column names -> this skill's
    keep = q.obs["subtype_uncertainty"] < 0.5
    print(fn, f"kept {keep.mean():.0%} of cells")
    frames.append(q.obs.loc[keep, ["subtype", "region"]].assign(section=fn, group=grp))
df = pd.concat(frames)
groups = ["young", "old"]
```
2. Per-section subtype proportions — the SECTION is the replication unit:
```python
counts = df.groupby(["group", "section", "subtype"]).size().unstack(fill_value=0)
props = counts.div(counts.sum(axis=1), axis=0)          # rows = (group, section)
```
3. Bootstrap CIs by resampling SECTIONS (not cells) within each group:
```python
def boot_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

rows = []
for st in props.columns:
    for g in groups:
        m, lo, hi = boot_ci(props.xs(g, level="group")[st].values)
        rows.append(dict(subtype=st, group=g, mean=m, lo=lo, hi=hi))
ci = pd.DataFrame(rows)
```
4. Rank the shifts and flag confident ones (CIs that do not overlap between groups):
```python
w = ci.pivot(index="subtype", columns="group", values=["mean", "lo", "hi"])
w["delta"] = w[("mean", groups[1])] - w[("mean", groups[0])]
w["separated"] = (w[("lo", groups[1])] > w[("hi", groups[0])]) | (w[("hi", groups[1])] < w[("lo", groups[0])])
print(w.sort_values("delta")[["delta", "separated"]])
```
5. The aging-microglia template — state balance WITHIN a parent class, region-stratified. Normalizing within microglia removes the compositional coupling to unrelated types:
```python
mg = df[df["subtype"].str.contains("microglia", case=False)].copy()
mg["state"] = np.where(mg["subtype"].str.contains("activ", case=False), "activated", "homeostatic")
frac = (mg.groupby(["region", "group", "section"])["state"]
          .apply(lambda s: (s == "activated").mean()).rename("frac_activated"))
```
6. Plot per region with per-section bootstrap error bars:
```python
regions = [r for r in frac.index.get_level_values("region").unique()
           if (mg["region"] == r).sum() > 200]           # enough microglia to estimate
fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(regions)), 5))
width = 0.35
for gi, g in enumerate(groups):
    means, err_lo, err_hi = [], [], []
    for reg in regions:
        m, lo, hi = boot_ci(frac.xs((reg, g), level=("region", "group")).values, seed=gi)
        means.append(m); err_lo.append(m - lo); err_hi.append(hi - m)
    ax.bar(np.arange(len(regions)) + (gi - 0.5) * width, means, width, label=g,
           yerr=[err_lo, err_hi], capsize=3)
ax.set_xticks(np.arange(len(regions))); ax.set_xticklabels(regions, rotation=60, ha="right")
ax.set_ylabel("activated microglia fraction"); ax.legend(); plt.tight_layout(); plt.show()
```
7. Repeat step 5-6 for any subtype family the user cares about (astrocyte states, oligodendrocyte maturation, ...) — same recipe, different `str.contains` filters. Check exact subtype spellings against `atlas_lookup("type")`.

## Verification checks
```python
assert np.allclose(props.sum(axis=1), 1)                       # proportions per section
assert df.groupby("group")["section"].nunique().min() >= 2, "need >=2 sections/group for section-level CIs"
assert set(df["group"].unique()) == set(groups)
assert mg["state"].nunique() == 2                              # both states present
```

## Pitfalls
- Proportions are compositional: one subtype expanding forces others down. For state questions, normalize within the parent class (step 5), not against all cells.
- Resample SECTIONS, not cells — per-cell bootstraps wildly underestimate variance because cells within a section are correlated.
- With 1 section per group, no honest CI exists — report point estimates and say so.
- Uncertainty filtering can bias composition if one group is systematically harder to map — always report the kept-fraction per section (step 1) and re-run without the filter as a robustness check.
- Only compare groups annotated through the same reference and pipeline; platform differences between groups masquerade as biology.
- Subtype string matching (`str.contains("activ")`) must be verified against the actual vocabulary (`atlas_lookup("type")` or `df["subtype"].unique()`) before trusting the split.
