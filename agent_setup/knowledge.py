### ---knowledge.py--- ###
"""
Central domain knowledge module for the FuseMap multi-agent system.

Provides a single source of truth for atlas schema information,
with agent-specific formatters that produce tailored views for each agent.
"""

_schema_cache = None


def get_atlas_schema() -> dict:
    """Lazy-load and cache atlas schema from ad_cell/ad_gene.

    Imports brain_atlas_tools only on first call to avoid circular imports
    (brain_atlas_tools loads h5ad files at module level).

    Returns a dict with counts, column names, and vocabulary lists.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    from agent_setup.tools.brain_atlas_tools import ad_cell, ad_gene

    _schema_cache = {
        "n_cells": ad_cell.n_obs,
        "n_genes": len(ad_gene.obs.index),
        "obs_cols": list(ad_cell.obs.columns),
        "tissue_main_vals": sorted(ad_cell.obs["tissue_main"].unique().tolist()),
        "main_starmap_vals": sorted(
            ad_cell.obs["main_STARmap"].unique().tolist()
        ),
        "n_sub_starmap": ad_cell.obs["sub_STARmap"].nunique(),
        "gene_obs_cols": list(ad_gene.obs.columns),
        "gm_labels": sorted(ad_gene.obs["GM_label"].unique().tolist())
        if "GM_label" in ad_gene.obs.columns else [],
        "obsm_keys": list(ad_cell.obsm.keys()),
        "n_tissue_sub": ad_cell.obs["tissue_sub"].nunique(),
        "n_main_allen": ad_cell.obs["main_Allen"].nunique()
        if "main_Allen" in ad_cell.obs.columns
        else 33,
        "n_sub_allen": ad_cell.obs["sub_Allen"].nunique()
        if "sub_Allen" in ad_cell.obs.columns
        else 332,
    }
    return _schema_cache


def format_schema_full(schema: dict, output_dir: str) -> str:
    """Full schema for CodingAgent — Sections 1-4, capabilities 4A-4K."""
    s = schema
    return f"""
=== FUSEMAP SYSTEM OVERVIEW ===

FuseMap is a deep learning framework that learns a shared 64-dimensional latent space
for mouse brain spatial transcriptomics. Both cells and genes are embedded as 64-dim
vectors in this space. The key operations:
  - Imputed gene expression = cell_embedding @ gene_embedding.T (dot product)
  - Cell type annotation = NN classifier on cell embeddings
  - Tissue region annotation = NN classifier on spatial embeddings
  - Gene panel design = HVG selection balanced across gene co-expression modules
  - Cell-cell interactions = spatial neighbor contact counting with permutation tests
  - Cell deconvolution = assign cell types to multi-cell spots via embedding optimization

================================================================================
                        SECTION 1: DATA IN NAMESPACE
================================================================================

ad_cell — {s['n_cells']:,} cells × 64 CELL EMBEDDINGS (float32, dense)
  obs columns: {s['obs_cols']}
  obsm: {s['obsm_keys']}

  Cell type hierarchy:
    main_STARmap ({len(s['main_starmap_vals'])}): {s['main_starmap_vals']}
    sub_STARmap ({s['n_sub_starmap']} codes): e.g. 'GLU_1', 'MG_1', 'OLG_3', 'VIP_1', 'CHO_4'
  Allen Brain Atlas hierarchy:
    main_Allen ({s['n_main_allen']}) → sub_Allen ({s['n_sub_allen']}) → subsub_Allen (1175) → cluster_Allen (4072)
  Brain regions:
    tissue_main ({len(s['tissue_main_vals'])}): {s['tissue_main_vals']}
    tissue_sub ({s['n_tissue_sub']}): e.g. 'MB_P_MY_17', 'CTX_A_1', 'HPF_CA_1'

ad_gene — {s['n_genes']:,} genes × 64 GENE EMBEDDINGS (float32, dense)
  obs index: UPPERCASE gene symbols (e.g. 'SATB2', 'CUX2', 'SNAP25', 'VIP')
  obs columns: {s['gene_obs_cols']}
  Gene modules (GM_label): {len(s['gm_labels'])} modules — {s['gm_labels'][:10]}...

================================================================================
                        SECTION 2: CRITICAL USAGE RULES
================================================================================

1. ad_cell.X and ad_gene.X are 64-dim EMBEDDINGS, NOT raw counts or normalized expr.
2. IMPUTED EXPRESSION = dot product:
     imputed_expr = ad_cell.X @ ad_gene[gene_list].X.T   # shape: (n_cells, n_genes)
   This gives a float matrix of imputed expression per cell per gene.
3. Gene names MUST be UPPERCASE: 'VIP' not 'vip', 'Vip', or 'VIP '.
4. Always check gene existence before indexing:
     assert 'VIP' in ad_gene.obs.index, "Gene VIP not found"
5. Do NOT modify ad_cell or ad_gene in-place — always .copy() first.
6. Save ALL output files under OUTPUT_DIR:
     import os; path = os.path.join(OUTPUT_DIR, 'result.h5ad')
7. Use plt.savefig(path); plt.close() — never plt.show().

================================================================================
                        SECTION 3: PRECOMPUTED RESOURCES
================================================================================

Lookup tables (as CSV files read with pd.read_csv):
  agent_setup/atlas_data/type_lookup.csv      — sub_STARmap ↔ description
  agent_setup/atlas_data/region_lookup.csv    — tissue_main ↔ description
  agent_setup/atlas_data/unique_pairs_celltype.csv  — valid (cell_type, section) pairs
  agent_setup/atlas_data/unique_pairs_tissue.csv    — valid (region, section) pairs

Color maps (CSV):
  agent_setup/atlas_data/colors/type_starmap_sub.csv   — sub_STARmap → hex color
  agent_setup/atlas_data/colors/type_starmap_main.csv  — main_STARmap → hex color
  agent_setup/atlas_data/colors/region_starmap_sub.csv — tissue_sub → hex color
  agent_setup/atlas_data/colors/region_starmap_main.csv — tissue_main → hex color

FuseMap model outputs (under output/ after running FuseMapAgent):
  output/fusemap/integrated_*.h5ad — integrated query data aligned to molCCF
  output/fusemap/mapped_*.h5ad     — query data mapped to molCCF cell types
  output/data/2D_section_ids.csv   — 2D atlas sections with cell type/region overlap

================================================================================
            SECTION 4: ANALYSIS RECIPES (copy-paste ready Python)
================================================================================

4A. EXTRACT IMPUTED GENE EXPRESSION → h5ad
    gene_list = ['VIP', 'MBP', 'SATB2']
    for g in gene_list:
        assert g in ad_gene.obs.index, f"Gene {{g}} not found"
    imputed = ad_cell.X @ ad_gene[gene_list].X.T   # (n_cells, n_genes)
    import anndata as ad_lib, pandas as pd
    result = ad_lib.AnnData(
        X=imputed,
        obs=ad_cell.obs[['tissue_main','tissue_sub','main_STARmap','sub_STARmap',
                          'global_x','global_y','global_z','ap_order']].copy(),
        var=pd.DataFrame(index=gene_list),
    )
    result.write_h5ad(os.path.join(OUTPUT_DIR, 'gene_expression_imputed.h5ad'))

4B. MARKER GENES FOR A CELL TYPE
    target_type = 'GLU'
    all_types = ad_cell.obs['main_STARmap'].unique()
    target_mask = ad_cell.obs['main_STARmap'] == target_type
    scores = {{}}
    for gene in ad_gene.obs.index[:500]:  # limit for speed
        expr = (ad_cell.X @ ad_gene[[gene]].X.T).flatten()
        from scipy import stats
        t, p = stats.ttest_ind(expr[target_mask], expr[~target_mask])
        scores[gene] = {{'t': t, 'p': p, 'mean_target': expr[target_mask].mean()}}
    markers = pd.DataFrame(scores).T.sort_values('t', ascending=False)
    markers.to_csv(os.path.join(OUTPUT_DIR, 'marker_genes.csv'))

4C. GENE PANEL DESIGN (HVG + module balance)
    # Compute per-cell-type mean imputed expression for all genes
    # Then select top HVGs balanced across GM_label modules
    # See fusemap/preprocess.py for select_highly_variable_genes()

4D. MARKER GENES (simpler approach via mean expression)
    type_col = 'main_STARmap'
    cell_types = ad_cell.obs[type_col].unique()
    gene_sample = ad_gene.obs.index[:200].tolist()
    expr = ad_cell.X @ ad_gene[gene_sample].X.T   # (n_cells, 200)
    expr_df = pd.DataFrame(expr, index=ad_cell.obs.index, columns=gene_sample)
    expr_df[type_col] = ad_cell.obs[type_col].values
    mean_expr = expr_df.groupby(type_col).mean()
    markers = mean_expr.idxmax(axis=1)   # top gene per cell type
    markers.sort_values('score', ascending=False)

4E. GENE-GENE RELATIONSHIPS
    Cosine similarity in embedding space reveals functional relationships:
      from sklearn.metrics.pairwise import cosine_similarity
      emb = ad_gene[gene_list].X
      sim = pd.DataFrame(cosine_similarity(emb), index=gene_list, columns=gene_list)

    Gene UMAP: ad_gene.obsm['X_umap'] has pre-computed 2D UMAP of gene embeddings.
    Genes close in UMAP space have similar expression patterns.

4F. CELL-CELL SPATIAL INTERACTIONS
    Permutation-based contact analysis using spatial coordinates:
      from fusemap.permutation import generate_cell_type_contact_count_matrices
      # permutation_method: 'no_permutation', 'local_permutation', 'global_permutation'

4G. CELL DECONVOLUTION (spot → cell type proportions)
    For spot-level data (Slide-seq, Visium):
      from fusemap.deconvolution import (
          get_cell_spot_embedding, get_representative_embeddings,
          optimize_cell_spot_assignment, evaluate_topk_accuracy)

4H. SPATIAL VISUALIZATION
    Always use the atlas color maps for consistency:
      colors_df = pd.read_csv('agent_setup/atlas_data/colors/type_starmap_sub.csv')
      color_dict = dict(zip(colors_df['key'], colors_df['color']))
    Spatial scatter: plt.scatter(obs['x'], obs['y'], c=..., s=0.1)
    3D plots: use global_x, global_y, global_z coordinates.

4I. INTEGRATION QUALITY METRICS
    sklearn.metrics on embeddings + cell type/batch labels.

4J. GENE IMPUTATION VALIDATION
    Compare imputed vs measured expression (when original data is available):
      measured = original_adata[:, 'GENE'].X.toarray().flatten()
      imputed = cell_embeddings.X @ ad_gene['GENE'].X.T

4K. CROSS-SECTION / REGION ANALYSIS
    Filter by AP section: ad_cell[ad_cell.obs['ap_order'] == '200']
    Filter by region: ad_cell[ad_cell.obs['tissue_main'] == 'CTX_1']

OUTPUT_DIR: '{output_dir}' — save all outputs here
"""


def format_schema_compact(schema: dict) -> str:
    """Compact schema for AtlasAgent — vocabulary lists + rules."""
    s = schema
    return f"""
=== ATLAS DATA SCHEMA (CRITICAL — read before every tool call) ===

tissue_main codes ({len(s['tissue_main_vals'])} total) — required by plot_region_distribution, find_section_ids, query_gene_expression_in_region, compute_gene_correlation:
  {s['tissue_main_vals']}

main_STARmap labels ({len(s['main_starmap_vals'])} total) — human-readable main cell type names:
  {s['main_starmap_vals']}

sub_STARmap codes — {s['n_sub_starmap']} atlas codes (e.g. 'MG_1', 'ASC_1', 'GLU_2').
  These are the ONLY valid values for the cell_types parameter in find_section_ids.
  ALWAYS run match_brain_type first to convert natural names → sub_STARmap codes.

Gene names — {s['n_genes']} uppercase mouse gene symbols (e.g. 'SATB2', 'CUX2', 'SNAP25').
  Gene names passed to query_gene_expression_in_region and compute_gene_correlation must be uppercase.

RULE: If any tool returns 'Error:', read the error message, correct the parameter
(usually by running match_brain_region or match_brain_type first), and retry once.
================================================================
"""


def format_schema_summary(schema: dict) -> str:
    """Brief schema for FuseMapAgent — data scale + key vocabulary."""
    s = schema
    return f"""
=== ATLAS REFERENCE DATA (for context) ===
The molCCF reference atlas contains:
  - {s['n_cells']:,} cells with 64-dim embeddings
  - {s['n_genes']:,} genes with 64-dim embeddings
  - {len(s['main_starmap_vals'])} main cell types: {s['main_starmap_vals']}
  - {len(s['tissue_main_vals'])} main brain regions: {s['tissue_main_vals']}
  - {s['n_sub_starmap']} sub-level cell type codes (e.g. 'GLU_1', 'VIP_1', 'MG_1')

Use this to decide the analysis strategy:
  - If query data has all cell types of normal adult mouse brain → map_molCCF only
  - If query data has partial cell types (e.g. disease) → map_molCCF + fusemap_integrate
  - If query data has totally different cell types (e.g. embryo) → fusemap_integrate only
"""


def format_atlas_context_for_research(schema: dict) -> str:
    """Atlas grounding for ResearchAgent."""
    s = schema
    return f"""
=== ATLAS CONTEXT (ground your search in these) ===
The FuseMap mouse brain atlas contains {s['n_cells']:,} cells and {s['n_genes']:,} genes.

Cell types in the atlas ({len(s['main_starmap_vals'])} main types):
  {s['main_starmap_vals']}

Brain regions in the atlas ({len(s['tissue_main_vals'])} main regions):
  {s['tissue_main_vals']}

When searching literature, prioritize genes, cell types, and brain regions
that exist in this atlas so results can be directly compared to the reference data.
Use UPPERCASE gene symbols (e.g. 'TREM2', 'APP', 'APOE') for consistency.
"""


def format_agent_capabilities() -> str:
    """Capability summary for supervisor routing."""
    return """
=== AGENT CAPABILITIES (use for routing decisions) ===

AtlasAgent — queries the reference mouse brain atlas:
  - Match brain region names → tissue_main/tissue_sub codes
  - Match cell type names → main_STARmap/sub_STARmap codes
  - Plot 3D brain region distributions
  - Find 2D section IDs containing specific cell types + regions
  - Query gene expression statistics in specific regions
  - Compute gene-gene correlations within regions
  - Extract imputed gene expression as h5ad

CodingAgent — writes and executes arbitrary Python code:
  - Extract imputed gene expression (dot product: cell_emb @ gene_emb.T)
  - Design targeted gene panels (HVG + module balancing)
  - Discover marker genes for cell types
  - Custom visualizations (heatmaps, violin plots, spatial scatter)
  - Cell type / tissue region annotation transfer via NNTransfer models
  - Gene-gene relationships (cosine similarity, co-expression modules)
  - Cell-cell spatial interaction analysis (permutation tests)
  - Cell deconvolution (spot → cell type proportions)
  - Integration quality metrics (MAP, silhouette, batch entropy)
  - Cross-region / cross-section comparative analysis
  - Any custom analysis requiring Python code

FuseMapAgent — runs FuseMap model pipelines on user data:
  - map_molCCF: map new spatial data to reference atlas (transfer labels)
  - fusemap_integrate: integrate multiple spatial sections
  - finalize_mainlevel: finalize main-level cell type annotations
  - annotate_sublevel: annotate sub-level cell types
  - Requires user-provided .h5ad files

ResearchAgent — searches scientific literature:
  - Disease and condition research (Alzheimer's, aging, etc.)
  - Cell type / gene / region involvement in diseases
  - Literature references and citations
  - Background knowledge for analysis context
"""
