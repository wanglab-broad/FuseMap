"""Plot full reference + mapped query UMAPs without modifying either input.

Example (run from a FuseMap checkout)::

    python examples/plot_mapping_umaps.py --reference ./output_tutorial1 \
        --query ./output_tutorial4/slideseq_Puck60.h5ad --output ./mapping_umaps

Each embedding uses one joint UMAP, shown with cell identities, region identities,
and reference/query colors. Reference annotations are retained where available;
missing reference annotations and query labels are predicted from labeled
reference observations only. Query labels never enter classifier training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc

import fusemap


MISSING = {"", "nan", "NA", "N/A", "None", "<NA>", "Unannotated"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(reference, query, level):
    parts = []
    for role, folder in (("reference", reference), ("query", query)):
        x = ad.read_h5ad(folder / f"ad_{level}_embedding.h5ad")
        obs = x.obs.copy()
        obs["observation_id"] = obs.index.astype(str)
        obs["role"] = role
        obs.index = [f"{role}|{file}|{name}" for file, name in
                     zip(obs["file_name"].astype(str), obs["observation_id"])]
        parts.append(ad.AnnData(np.asarray(x.X, dtype=np.float32), obs=obs))
    result = ad.concat(parts, join="outer")
    if not result.obs_names.is_unique:
        raise ValueError("Observation IDs must be unique within each section")
    return result


def add_identities(combined, label_key, output_key, epochs, seed):
    fusemap.seed_all(seed)
    labels = combined.obs[label_key].astype(str).str.strip()
    known = combined.obs["role"].eq("reference") & ~labels.isin(MISSING)
    # Use a separate column so query annotations can never become training data.
    combined.obs["reference_training_label"] = labels.where(known, np.nan)
    result = fusemap.transfer_labels(combined, label_key="reference_training_label",
                                     batch_size=1024, epochs=epochs, device="cpu")
    predicted = combined.obs["transfer_reference_training_label"].astype(str)
    combined.obs[output_key] = labels.where(known, predicted)
    combined.obs[output_key + "_source"] = np.where(known, "reference annotation", "inferred")
    combined.obs[output_key + "_uncertainty"] = combined.obs["transfer_reference_training_label_uncertainty"]
    combined.obs.drop(columns=["reference_training_label", "transfer_reference_training_label",
                               "transfer_reference_training_label_uncertainty"], inplace=True)
    return {"reference_labeled": int(known.sum()),
            "held_out_reference_balanced_accuracy": float(result["test_accuracy"]),
            "query_counts": combined.obs.loc[combined.obs["role"].eq("query"), output_key].value_counts().to_dict()}


def categorical_palette(values, colors):
    categories = sorted(pd.unique(values.astype(str)))
    if len(categories) > len(colors):
        raise ValueError("Provide enough distinct colors for all identity labels")
    return {name: colors[i] for i, name in enumerate(categories)}


def plot_panels(data, level_title, palettes, output, seed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.titlesize": 13})
    xy = data.obsm["X_umap"]
    n_ref = int(data.obs["role"].eq("reference").sum())
    n_query = data.n_obs - n_ref
    order = np.random.default_rng(seed).permutation(data.n_obs)
    # Identical point order across panels; avoid always drawing query on top.
    fig = plt.figure(figsize=(20, 10.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[5.5, 3.0], hspace=0.12, wspace=0.13)
    specifications = [("cell_identity", "Cell identities"),
                      ("region_identity", "Tissue-region identities"),
                      ("role", "Reference / query")]
    for panel, (key, title) in enumerate(specifications):
        ax = fig.add_subplot(grid[0, panel])
        colors = [palettes[key][value] for value in data.obs[key].astype(str).iloc[order]]
        ax.scatter(xy[order, 0], xy[order, 1], c=colors, s=0.7, alpha=0.65,
                   linewidths=0, rasterized=True)
        ax.set_title(title, loc="left", pad=13)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)
        legend_ax = fig.add_subplot(grid[1, panel])
        legend_ax.axis("off")
        handles = []
        for category, color in palettes[key].items():
            text = category
            if key == "role":
                text = f"{category.capitalize()} ({n_ref if category == 'reference' else n_query:,})"
            handles.append(Line2D([0], [0], marker="o", linestyle="none", color=color,
                                  markersize=5, label=textwrap.fill(text, width=31)))
        legend_ax.legend(handles=handles, loc="upper left", frameon=False,
                         ncol=2 if len(handles) > 12 else 1, fontsize=8.5,
                         handletextpad=0.5, columnspacing=1.2, borderaxespad=0)
    fig.suptitle(f"{level_title} | reference + mapped Slide-seq", x=0.06, ha="left", fontsize=17)
    fig.text(0.06, 0.935, f"{n_ref:,} reference cells · {n_query:,} query beads · all observations shown",
             color="#475569", fontsize=11)
    fig.text(0.06, 0.025,
             "Reference labels: observed where available; otherwise inferred. Query identities: inferred from reference labels.\n"
             "Joint UMAP of the 64-dimensional embeddings; coloring panels share coordinates. UMAP mixing is not a composition-accuracy test.",
             fontsize=9, color="#475569")
    fig.subplots_adjust(top=0.885, bottom=0.08, left=0.055, right=0.985)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(output.with_name(output.name + "_preview.png"), dpi=110, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-label", default="gt_cell_type_main")
    parser.add_argument("--region-label", default="gt_tissue_region_main")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-epochs", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [folder / f"ad_{level}_embedding.h5ad" for folder in (args.reference, args.query)
               for level in ("celltype", "tissueregion")]
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    settings = dict(input_sha256=hashes, cell_label=args.cell_label, region_label=args.region_label,
                    seed=args.seed, label_epochs=args.label_epochs, n_neighbors=50, min_dist=0.5,
                    metric="euclidean", use_rep="X", observations="all")
    report_path = args.output / "umap_report.json"
    cache_paths = [args.output / f"reference_query_{level}_umap.h5ad" for level in ("celltype", "tissueregion")]
    reuse = report_path.exists() and all(p.exists() for p in cache_paths)
    if reuse:
        report = json.loads(report_path.read_text())
        reuse = report["settings"] == settings
    if reuse:
        cell, tissue = [ad.read_h5ad(p) for p in cache_paths]
    else:
        cell = load_pair(args.reference, args.query, "celltype")
        tissue = load_pair(args.reference, args.query, "tissueregion")
        if set(cell.obs_names) != set(tissue.obs_names):
            raise ValueError("Cell and tissue embedding observation IDs differ")
        tissue = tissue[cell.obs_names].copy()
        print("Transfer cell identities from labeled reference only", flush=True)
        cell_scores = add_identities(cell, args.cell_label, "cell_identity", args.label_epochs, args.seed)
        print("Transfer tissue-region identities from labeled reference only", flush=True)
        region_scores = add_identities(tissue, args.region_label, "region_identity", args.label_epochs, args.seed)
        for key in ["cell_identity", "cell_identity_source", "cell_identity_uncertainty"]:
            tissue.obs[key] = cell.obs[key]
        for key in ["region_identity", "region_identity_source", "region_identity_uncertainty"]:
            cell.obs[key] = tissue.obs[key]
        for data, level, path in zip([cell, tissue], ["celltype", "tissueregion"], cache_paths):
            print(f"Joint {level} UMAP: {data.n_obs:,} observations, all 64 latent dimensions", flush=True)
            sc.pp.neighbors(data, n_neighbors=50, use_rep="X", metric="euclidean", random_state=args.seed)
            sc.tl.umap(data, min_dist=0.5, random_state=args.seed)
            assert np.isfinite(data.obsm["X_umap"]).all()
            # Retain only relevant observation fields in the exported plotting data.
            keep = ["observation_id", "file_name", "role", "cell_identity", "region_identity",
                    "cell_identity_source", "region_identity_source",
                    "cell_identity_uncertainty", "region_identity_uncertainty"]
            data.obs = data.obs[keep].copy()
            data.write_h5ad(path, compression="gzip")
        report = dict(settings=settings, counts=cell.obs["role"].value_counts().to_dict(),
                      cell_identity_classifier=cell_scores, region_identity_classifier=region_scores,
                      reference_labels="observed when present; otherwise predicted", query_labels="predicted",
                      note="Separate joint UMAP per embedding; same coordinates for every coloring of that embedding")
    palettes = {"cell_identity": categorical_palette(cell.obs["cell_identity"], sc.pl.palettes.default_28),
                "region_identity": categorical_palette(tissue.obs["region_identity"], sc.pl.palettes.default_28),
                "role": {"reference": "#2878A5", "query": "#E87524"}}
    (args.output / "palettes.json").write_text(json.dumps(palettes, indent=2))
    for data, level, title in [(cell, "celltype", "Cell-type embedding"),
                               (tissue, "tissueregion", "Tissue-region embedding")]:
        plot_panels(data, title, palettes, args.output / f"mapping_umap_{level}", args.seed)
    report_path.write_text(json.dumps(report, indent=2))
    assert hashes == {str(path.resolve()): sha256(path) for path in sources}, "Input embeddings changed"
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
