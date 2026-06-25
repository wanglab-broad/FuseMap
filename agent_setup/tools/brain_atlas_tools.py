### ---brain_atlas_tools.py--- ###
import os
import pandas as pd
import numpy as np
import re
import scanpy as sc
import plotly.express as px
import anndata as ad
import matplotlib.pyplot as plt
from typing import Annotated, List
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.base_language import BaseLanguageModel
# from agent_setup.config import llm
import pandas as pd
import itertools

# Fix numpy compatibility with older scanpy (NumPy 2.0+)
# Note: np.float_ was removed in NumPy 2.0, no compatibility shim needed

# ── Path-safety utilities ──────────────────────────────────────────────────────
# All Atlas tools that write output files must call _validate_output_dir()
# before creating any directory or writing any file.  This provides the
# same protection as the REPL's _safe_open(), but for tools that write
# via anndata / plotly / pandas (which bypass Python's open() hook).
#
# The allowed root is resolved once at module load time relative to the
# current working directory.  Running from the project root (as the
# Streamlit app does) makes this point to <project>/output.
_ALLOWED_OUTPUT_ROOT: str = os.path.abspath("output")


def _validate_output_dir(output_dir: str) -> str:
    """Resolve *output_dir* to an absolute path and assert it is inside
    ``_ALLOWED_OUTPUT_ROOT``.

    Returns the resolved absolute path so callers can use it directly.

    Raises:
        PermissionError: if the resolved path escapes the allowed root.
    """
    resolved = os.path.abspath(output_dir.strip("'").strip('"'))
    # Append os.sep to avoid matching '/output_extra' when root is '/output'
    allowed_prefix = _ALLOWED_OUTPUT_ROOT + os.sep
    if resolved != _ALLOWED_OUTPUT_ROOT and not resolved.startswith(allowed_prefix):
        raise PermissionError(
            f"Writing is only allowed under '{_ALLOWED_OUTPUT_ROOT}'. "
            f"Got: '{resolved}'. "
            f"Use a relative path like 'output/subdir' or just 'output'."
        )
    return resolved
# ─────────────────────────────────────────────────────────────────────

# Load atlas data (assume preloaded to avoid reloading per tool call)
ad_cell = sc.read_h5ad('agent_setup/atlas_data/ad_cell.h5ad')
ad_gene = sc.read_h5ad('agent_setup/atlas_data/ad_gene.h5ad')
ad_cell.obs['ap_order'] = ad_cell.obs['ap_order'].astype(float)


### ---------------- Tool 1: Match Brain Regions ---------------- ###
def create_region_matcher(llm):
    region_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a neuroscience expert who understands brain anatomy and nomenclature.
Your task is to match target brain regions with available regions in an atlas. Use ONLY the Region_ID values from the atlas.MATCHED_REGIONS cannot be numbers.

Format:
REGION: [region name]
MATCHED_REGIONS: [comma-separated Region_IDs]
REASONING: [explanation]
CONFIDENCE: [High/Medium/Low]"""),
        ("user", """Target regions to find: {target_regions}

Available atlas regions:
{available_regions}""")
    ])
    return region_prompt | llm

def parse_matches_region(text):
    sections = text.strip().split('\n\n')
    output = {}
    for section in sections:
        lines = section.split('\n')
        if len(lines) >= 2:
            region_name = lines[0].replace('REGION:', '').strip().lower()
            matched = lines[1].replace('MATCHED_REGIONS:', '').strip().split(',')
            output[region_name] = [m.strip() for m in matched]
    return output

def match_brain_regions(target_regions: List[str], 
                        llm, 
                        log: Annotated[callable, "Logger function"] = print) -> str:

    atlas_regions = pd.read_csv('agent_setup/atlas_data/region_lookup.csv')
    available_regions = atlas_regions.to_string(columns=['Region_ID', 'Description'],index=False)

    matcher = create_region_matcher(llm)
    response = matcher.invoke({
        "target_regions": "\n".join(f"- {r}" for r in target_regions),
        "available_regions": available_regions
    })
    return f"Matched brain regions in the spatial brain atlas: {parse_matches_region(response.content)}"



class match_brain_region_tool(BaseTool):
    name: str = "match_brain_region"
    description: str = "Given input description of target brain regions, output LLM-based match of target brain regions to known atlas regions in spatial brain atlas."
    llm: BaseLanguageModel = None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.llm = llm

    def _run(self, target_regions: List[str], log: Annotated[callable, "Logger function"] = print):
        return match_brain_regions(target_regions, self.llm, log)



### ---------------- Tool 2: Match Cell Types ---------------- ###
def create_type_matcher(llm):
    type_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a neuroscience expert who understands brain cell types.
Your task is to match target brain cell types with available types in the atlas. Use ONLY the Cell type ID values.

Format:
TYPE: [type name]
MATCHED_TYPES: [comma-separated Cell Type IDs]
REASONING: [explanation]
CONFIDENCE: [High/Medium/Low]"""),
        ("user", """Target types to find: {target_types}

Available atlas types:
{available_types}""")
    ])
    return type_prompt | llm

def parse_matches_type(text):
    sections = text.strip().split('\n\n')
    output = {}
    for section in sections:
        lines = section.split('\n')
        if len(lines) >= 2:
            type_name = lines[0].replace('TYPE:', '').strip()
            matched = lines[1].replace('MATCHED_TYPES:', '').strip().split(',')
            output[type_name] = [m.strip() for m in matched]
    return output


def match_brain_type(target_types: List[str], llm, log: Annotated[callable, "Logger function"] = print) -> str:
    
    atlas_types = pd.read_csv('agent_setup/atlas_data/type_lookup.csv')
    available_types = atlas_types.to_string(columns=['Symbol', 'Description'], index=False)
    matcher = create_type_matcher(llm)
    response = matcher.invoke({
        "target_types": "\n".join(f"- {r}" for r in target_types),
        "available_types": available_types
    })
    matched_types = parse_matches_type(response.content)
    log(f"🔍 Matched types: {matched_types}")
    return f"Matched cell types in the spatial brain atlas: {matched_types}"


class match_brain_type_tool(BaseTool):
    name: str = "match_brain_type"
    description: str = "Given input description of target brain cell types, output LLM-based match of target brain cell types to known atlas types in spatial brain atlas."
    llm: BaseLanguageModel= None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.llm = llm

    def _run(self, target_types: List[str], log: Annotated[callable, "Logger function"] = print):
        return match_brain_type(target_types, self.llm, log)



### ---------------- Tool 3: 3D Cell Distribution Plot ---------------- ###
@tool
def plot_region_distribution(region_id: List[str], gene_id: List[str] = None,
                             output_dir: str = "output",
                             log: Annotated[callable, "Logger function"] = print) -> str:
    """Given a list of atlas tissue_main regions, 3D scatter plot of cells in each region, colored by subregion.
    Optionally, input also may include a list of genes to plot. In this case, there will be separate plots for 3D gene expression.
    output_dir is the base directory to save all results (default: 'output').
    Figures will be saved in [output_dir]/figures/."""
    import gc

    output_dir = _validate_output_dir(output_dir)
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    output_string = ""

    # Validate region_ids against atlas vocabulary
    valid_tissue_main = set(ad_cell.obs['tissue_main'].unique())
    invalid_regions = set(region_id) - valid_tissue_main
    if invalid_regions:
        return (f"Error: {sorted(invalid_regions)} are not valid tissue_main codes. "
                f"Run match_brain_region first. "
                f"All valid values: {sorted(valid_tissue_main)}")

    # Limit to top 5 genes to prevent OOM and connection timeout
    if gene_id and len(gene_id) > 5:
        log(f"⚠️ Too many genes requested ({len(gene_id)}). Limiting to first 5 for performance.")
        gene_id = gene_id[:5]

    subset = ad_cell[ad_cell.obs['ap_order'] < 404]
    for region in region_id:
        log(f"📍 Plotting region: {region}")
        region_cells = subset[subset.obs['tissue_main'] == region]
        if region_cells.shape[0] == 0:
            log(f"⚠️ No cells found for region: {region}")
            continue

        # Simple downsampling if too many cells for 3D plot
        if region_cells.shape[0] > 50000:
            idx = np.random.choice(region_cells.shape[0], 50000, replace=False)
            region_cells_plot = region_cells[idx]
        else:
            region_cells_plot = region_cells

        fig = px.scatter_3d(
            region_cells_plot.obs,
            x='global_x', y='global_y', z='global_z',
            color='tissue_sub',
            title=f'3D Cell Distribution in {region}',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_traces(marker_size=1)
        out_path = os.path.join(figures_dir, f"Brain_region_{region}_subregion_3D.html")
        fig.write_html(out_path)
        del fig
        gc.collect()

        output_string += f"Saved 3D cell distribution plot for {region} in {out_path}.\n"
        log(f"✅ Saved 3D cell distribution plot for {region}")

        if gene_id:
            for gene in gene_id:
                gene = gene.upper()
                if gene in ad_gene.obs.index:
                    log(f"🎯 Plotting gene {gene} in region {region}")
                    ad_gene_target = ad_gene[gene]
                    gene_exp = region_cells.X @ ad_gene_target.X.T

                    # Create temporary visualization DF to avoid full AnnData overhead
                    vis_df = region_cells.obs[['global_x', 'global_y', 'global_z']].copy()
                    vis_df[gene] = gene_exp[:, 0]

                    # Downsample for gene plots too
                    if vis_df.shape[0] > 30000:
                        vis_df = vis_df.sample(30000)

                    fig = px.scatter_3d(
                        vis_df,
                        x='global_x', y='global_y', z='global_z',
                        color=gene,
                        title=f'3D Gene Expression of {gene} in {region}',
                        color_continuous_scale=px.colors.sequential.Reds,
                        range_color=[vis_df[gene].min(), vis_df[gene].max()]
                    )
                    fig.update_traces(marker_size=1)
                    out_path = os.path.join(figures_dir, f"Brain_region_{region}_gene_{gene}_3D.html")
                    fig.write_html(out_path)

                    del fig
                    gc.collect()

                    log(f"✅ Saved 3D gene expression plot for {gene}")
                    output_string += f"Saved 3D gene expression plot for {gene} in {out_path}.\n"

    gc.collect()
    return output_string



### ---------------- Tool 4: Find Relevant 2D Sections ---------------- ###
@tool
def find_section_ids(cell_types: List[str], brain_regions: List[str],
                     output_dir: str = "output",
                     log: Annotated[callable, "Logger function"] = print) -> List[float]:
    """Given a list of cell types and brain regions,
    the list cannot be empty,
    find all 2D sections with given cell types and brain regions.
    cell_types must be values from the sub_STARmap column (atlas codes, e.g. 'MG_1').
    brain_regions must be values from the tissue_main column (atlas codes, e.g. 'HPF_CA').
    Use match_brain_type and match_brain_region first to convert natural names to atlas codes.
    output_dir is the base directory to save results (default: 'output').
    Results will be saved in [output_dir]/data/."""
    output_dir = _validate_output_dir(output_dir)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if len(cell_types) == 0:
        log("⚠️ Empty cell types list provided.")
        return "No matching sections found. Please provide valid cell types list."

    if len(brain_regions) == 0:
        log("⚠️ Empty brain regions list provided.")
        return "No matching sections found. Please provide valid brain regions list."

    # Validate against known atlas vocabulary before querying
    valid_sub_STARmap = set(ad_cell.obs['sub_STARmap'].unique())
    invalid_types = set(cell_types) - valid_sub_STARmap
    if invalid_types:
        sample_valid = sorted(valid_sub_STARmap)[:10]
        return (f"Error: {sorted(invalid_types)} are not valid sub_STARmap codes. "
                f"Run match_brain_type first to convert natural-language names to atlas codes. "
                f"Example valid codes: {sample_valid} ...")

    valid_tissue_main = set(ad_cell.obs['tissue_main'].unique())
    invalid_regions = set(brain_regions) - valid_tissue_main
    if invalid_regions:
        return (f"Error: {sorted(invalid_regions)} are not valid tissue_main codes. "
                f"Run match_brain_region first. "
                f"All valid values: {sorted(valid_tissue_main)}")

    matched = ad_cell.obs[
        ad_cell.obs['sub_STARmap'].isin(cell_types) & 
        ad_cell.obs['tissue_main'].isin(brain_regions)
    ]['ap_order'].unique()

    ### choose 20% highest percentage sections
    ad_cell.obs['in_brain'] = ad_cell.obs['tissue_main'].isin(brain_regions)
    # filter to only the orders of interest, then group & compute percentage
    percent_df = (
        ad_cell.obs[ad_cell.obs['ap_order'].isin(matched)]
        .groupby('ap_order')['in_brain']
        .mean()                # fraction of True’s
        .mul(100)              # to percent
        .round(1)              # e.g. 50.0
        .astype(str)
        .add('%')              # append “%”
    )
    # convert to dict if you like
    result = percent_df.to_dict()
    # 1. Convert to a float Series
    s = (
        pd.Series(result)
        .str.rstrip('%')       # drop the “%”
        .str.strip()            # drop whitespace
        .astype(float)         # to numeric
    )
    # 2. Compute the 80th percentile threshold
    threshold = s.quantile(0.8)
    # 3. Select keys ≥ threshold
    top_keys = s[s >= threshold].index.tolist()
    # 4. If that yields fewer than 2 keys, fallback to the top‐2 overall
    if len(top_keys) < 2:
        top_keys = s.nlargest(2).index.tolist()
    matched = np.sort(top_keys)
    # print("Selected keys:", top_keys)


    log(f"📊 Found {len(matched)} matching sections: {matched}")



    # Precompute section-wise total counts
    section_counts = ad_cell.obs.groupby('ap_order').size()

    # Precompute counts per (section, cell_type, region)
    grouped = ad_cell.obs.groupby(['ap_order', 'sub_STARmap', 'tissue_main']).size()

    # Normalize to get proportion
    proportions = grouped / section_counts

    # Pivot to a table where rows are cell_type_region, columns are sections
    proportions = proportions.reset_index(name='value')

    # Create a combined key for cell_type and region
    proportions['CellType_BrainRegion'] = (
        'CellType_' + proportions['sub_STARmap'].astype(str) +
        '_BrainRegion_' + proportions['tissue_main'].astype(str)
    )

    # Pivot to final table
    table = proportions.pivot(index='CellType_BrainRegion', columns='ap_order', values='value').fillna(0)

    # Optional: reindex to ensure all expected pairs are present
    cell_types = ad_cell.obs['sub_STARmap'].unique()
    brain_regions = ad_cell.obs['tissue_main'].unique()
    pairs = ['CellType_' + ct + '_BrainRegion_' + br for ct, br in itertools.product(cell_types, brain_regions)]
    table = table.reindex(pairs).fillna(0)

    # Optional: reorder columns to match your 'matched' list
    table = table[matched]
    table.columns = [f'Section_{int(col)}' for col in table.columns]
    csv_path = os.path.join(data_dir, '2D_section_ids.csv')
    table.to_csv(csv_path)
    log(f"✅ Saved matched section IDs to `{csv_path}`\n\n")

    # Do NOT call any st.* functions here: this tool runs inside a background
    # thread where Streamlit has no session context. In bare mode st.* calls
    # succeed silently but corrupt the frontend delta queue → white screen.

    return f"Found {len(matched)} matching sections in the spatial brain atlas: {matched}.\n\nSaved percentage of cell types in brain regions table to {csv_path}"



### ---------------- Tool 5: Read and Plot One Section ---------------- ###
@tool
def read_one_section(section_id: str, target_gene: List[str] = None,
                     output_dir: str = "output",
                     log: Annotated[callable, "Logger function"] = print) -> str:
    """Given a section ID, extract the spatial transcriptomic data of the section and visualize gene expression and annotations for one AP section.
    The section_id either come from results of Tool find_section_ids or from the user input.
    Optionally, input also may include a list of target genes to plot. In this case, there will be separate plots for gene expression.
    output_dir is the base directory to save results (default: 'output').
    Results will be saved in [output_dir]/data/ and [output_dir]/figures/."""
    output_dir = _validate_output_dir(output_dir)
    data_dir = os.path.join(output_dir, 'data')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    output_string = ""
    sid = float(section_id)
    subset = ad_cell[ad_cell.obs['ap_order'] == sid]
    gene_exp = subset.X @ ad_gene.X.T
    ad_gene_exp = ad.AnnData(X=gene_exp)
    ad_gene_exp.obs = subset.obs
    ad_gene_exp.var.index = ad_gene.obs.index
    ad_gene_exp.write_h5ad(os.path.join(data_dir, f"adata_Section_{section_id}.h5ad"))

    # Layout scale
    y_span = ad_gene_exp.obs['global_y'].max() - ad_gene_exp.obs['global_y'].min()
    z_span = ad_gene_exp.obs['global_z'].max() - ad_gene_exp.obs['global_z'].min()
    x_y_scale = z_span / y_span

    # 1. Gene expression heatmap
    if target_gene:
        for gene in target_gene:
            gene = gene.upper()
            if gene in ad_gene_exp.var.index:
                plt.figure(figsize=(10, 10 / x_y_scale))
                plt.scatter(ad_gene_exp.obs['global_z'], ad_gene_exp.obs['global_y'],
                            c=ad_gene_exp[:, gene].X, s=3, cmap='Reds')
                plt.gca().invert_yaxis()
                plt.colorbar(label='Expression Level')
                out_path = os.path.join(figures_dir, f"Section_{section_id}_gene_{gene}.png")
                plt.savefig(out_path, dpi=300)
                output_string += f"Saved gene expression plot for {gene} in {out_path}.\n"
                plt.close()

    # 2. Cell types
    color_main = pd.read_csv('agent_setup/atlas_data/colors/type_starmap_main.csv', index_col=0).set_index('key')['color'].to_dict()
    color_sub = pd.read_csv('agent_setup/atlas_data/colors/type_starmap_sub.csv', index_col=0).set_index('key')['color'].to_dict()
    plt.figure(figsize=(10, 5 / x_y_scale))
    plt.scatter(560 * 2 - ad_gene_exp.obs['global_z'], ad_gene_exp.obs['global_y'],
                c=[color_main[i] for i in ad_gene_exp.obs['main_STARmap']], s=3)
    plt.scatter(ad_gene_exp.obs['global_z'], ad_gene_exp.obs['global_y'],
                c=[color_sub[i] for i in ad_gene_exp.obs['sub_STARmap']], s=3)
    plt.gca().invert_yaxis()
    out_path = os.path.join(figures_dir, f"Section_{section_id}_cell_type.png")
    plt.savefig(out_path, dpi=300)
    output_string += f"Saved cell types plot for {section_id} in {out_path}.\n"
    plt.close()

    # 3. Brain regions
    color_rmain = pd.read_csv('agent_setup/atlas_data/colors/region_starmap_main.csv', index_col=0).set_index('key')['color'].to_dict()
    color_rsub = pd.read_csv('agent_setup/atlas_data/colors/region_starmap_sub.csv', index_col=0).set_index('key')['color'].to_dict()
    plt.figure(figsize=(10, 5 / x_y_scale))
    plt.scatter(560 * 2 - ad_gene_exp.obs['global_z'], ad_gene_exp.obs['global_y'],
                c=[color_rmain[i] for i in ad_gene_exp.obs['tissue_main']], s=3)
    plt.scatter(ad_gene_exp.obs['global_z'], ad_gene_exp.obs['global_y'],
                c=[color_rsub[i] for i in ad_gene_exp.obs['tissue_sub']], s=3)
    plt.gca().invert_yaxis()
    out_path = os.path.join(figures_dir, f"Section_{section_id}_brain_region.png")
    plt.savefig(out_path, dpi=300)
    output_string += f"Saved brain regions plot for {section_id} in {out_path}.\n"
    plt.close()

    # Release the per-section AnnData now that all plots and the h5ad are
    # written.  This object can be hundreds of MB for large sections; without
    # an explicit del it would remain in memory until Python's cyclic GC runs
    # (which may be many invocations later in a long Streamlit session).
    import gc as _gc
    del ad_gene_exp, gene_exp
    _gc.collect()

    h5ad_path = os.path.join(data_dir, f"adata_Section_{section_id}.h5ad")
    return f"Saved Section {section_id} in the spatial brain atlas to {h5ad_path}. {output_string}"


### ---------------- Tool 6: Explain Cell Type ---------------- ###
@tool
def explain_cell_type(cell_type: list[str], log: Annotated[callable, "Logger function"] = print) -> str:
    """cell_type must be a list of cell type symbols in the spatial brain atlas. 
    This tool will explain the cell types, including the main cell type cluster, marker genes, description, etc.. 
    Note that the symbols in cell_type symbol must be the symbol in the spatial brain atlas, if not, use match_brain_type to find the correct symbols."""
    sub_cell_type = pd.read_csv('agent_setup/atlas_data/sub_cell_type_annotation.csv', index_col=0)
    sub_cell_type = sub_cell_type.set_index('Subcluster Symbol', inplace=False)
    output_string = ''
    for cell_type_j in cell_type:
        output_string += 'The detailed description for symbol ' + cell_type_j + ' is: '
        for i in sub_cell_type.columns:
            # print(i,sub_cell_type.loc[cell_type_j,i])
            try:
                output_string += i + ': ' + sub_cell_type.loc[cell_type_j,i] + '\n'
            except:
                output_string += i + ': ' + 'No description found' + '\n'
        output_string += '\n'
    return output_string


### ---------------- Tool 7: Atlas Code Interpreter ---------------- ###

# Patterns that would allow file I/O or shell access from the sandbox.
# These are blocked regardless of intent to prevent accidental data loss or
# unintended side-effects on the HPC filesystem.
_BLOCKED_PATTERNS = [
    (r'\bopen\s*\(',           "file I/O (open)"),
    (r'\bimport\s+os\b',       "os module"),
    (r'\bimport\s+subprocess\b', "subprocess module"),
    (r'\b__import__\s*\(',     "dynamic import"),
    (r'\.write_h5ad\s*\(',     "writing h5ad files"),
    (r'\.write\s*\(',          "file write method"),
    (r'\bos\.system\s*\(',     "shell execution"),
    (r'\bos\.popen\s*\(',      "shell pipe"),
]

@tool
def execute_atlas_query(code: str, log: Annotated[callable, "Logger function"] = print) -> str:
    """Execute arbitrary Python code against the pre-loaded atlas data objects.

    Use this tool for ad-hoc queries that the other tools cannot answer directly,
    such as custom aggregations, differential expression between groups, or
    multi-step pandas/scanpy operations.

    Pre-loaded variables available in the execution namespace:
      - ad_cell  : safe proxy — ad_cell.obs is a COPY (mutations are local and do not
                   affect the shared atlas). ad_cell.X and ad_cell.shape are the real data.
      - ad_gene  : AnnData, 26665 genes × 64 dims (gene embedding matrix).
                   ad_gene.obs.index contains uppercase gene symbols.
      - pd       : pandas
      - np       : numpy
      - sc       : scanpy
      - stats    : scipy.stats

    To return output, either print() or assign to the variable `result`.
    Output is truncated at 3000 characters.

    File I/O and shell access are blocked. Do not attempt to open files,
    import os/subprocess, or call write methods.

    Examples:
      "print(ad_cell.obs['tissue_main'].value_counts().to_string())"
      "result = ad_cell.obs.groupby('main_STARmap').size().sort_values(ascending=False)"
      "mask = ad_cell.obs['tissue_main'] == 'HPF_CA'; print(ad_cell[mask].shape)"
    """
    import io, re, types
    from contextlib import redirect_stdout
    from scipy import stats as _stats

    # Layer 1 — blocklist: reject dangerous patterns before any execution
    for pattern, reason in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return (f"Error: Blocked operation detected ({reason}). "
                    f"File I/O and shell access are not permitted in this sandbox.")

    # Layer 2 — safe proxy: expose ad_cell.obs as a pandas copy so that
    # any mutations (e.g. adding a column) stay local and never pollute the
    # shared in-memory atlas used by other tools.
    _safe_cell = types.SimpleNamespace(
        obs   = ad_cell.obs.copy(),   # copy — safe to mutate
        X     = ad_cell.X,            # sparse matrix — read-only in practice
        shape = ad_cell.shape,
        var   = ad_cell.var,
    )

    namespace = {
        "ad_cell": _safe_cell,
        "ad_gene": ad_gene,
        "pd": pd,
        "np": np,
        "sc": sc,
        "stats": _stats,
        "result": None,
    }

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(code, namespace)  # noqa: S102
        output = buf.getvalue()
        if not output and namespace["result"] is not None:
            output = str(namespace["result"])
        if not output:
            output = "Code executed successfully (no output). Use print() or assign to `result`."
        if len(output) > 3000:
            output = output[:3000] + "\n... [truncated at 3000 chars]"
        log(f"✅ execute_atlas_query completed ({len(output)} chars)")
        return output
    except Exception as e:
        import traceback
        return f"Error during execution:\n{e}\n\nTraceback:\n{traceback.format_exc()}"


### ---------------- Tool 7: Top Expressed Genes in Region/Cell-type ---------------- ###
@tool
def top_expressed_genes_in_region(tissue_main: str, sub_starmap: str = None, top_n: int = 10) -> str:
    """Return the top N most highly expressed genes in a given atlas region, optionally
    restricted to a specific cell type.

    Parameters
    ----------
    tissue_main : str
        Valid tissue_main atlas code (e.g. 'HPF_CA', 'CTX_1').
        Use match_brain_region to convert natural-language region names to codes.
    sub_starmap : str, optional
        Valid sub_STARmap cell-type code (e.g. 'INH_1', 'MGL_1', 'OLG_2').
        Use match_brain_type to convert natural-language cell-type names to codes.
        If omitted, all cells in the region are used.
    top_n : int, optional
        Number of top genes to return (default 10).

    Returns a ranked table of gene names and their association scores.
    """
    valid_regions = set(ad_cell.obs['tissue_main'].unique())
    if tissue_main not in valid_regions:
        return (f"Error: '{tissue_main}' is not a valid tissue_main code. "
                f"Valid values: {sorted(valid_regions)}")

    mask = ad_cell.obs['tissue_main'] == tissue_main
    region_label = tissue_main

    if sub_starmap:
        valid_types = set(ad_cell.obs['sub_STARmap'].unique())
        if sub_starmap not in valid_types:
            return (f"Error: '{sub_starmap}' is not a valid sub_STARmap code. "
                    f"Valid values: {sorted(valid_types)}")
        mask = mask & (ad_cell.obs['sub_STARmap'] == sub_starmap)
        region_label = f"{sub_starmap} in {tissue_main}"

    cells = ad_cell[mask]
    if cells.shape[0] == 0:
        return f"No cells found for {region_label}."

    # Centroid of cell embeddings → dot with gene embeddings = association score
    centroid = np.asarray(cells.X.mean(axis=0)).flatten()   # (64,)
    scores = np.asarray(ad_gene.X @ centroid).flatten()      # (n_genes,)

    top_idx = np.argsort(scores)[::-1][:top_n]
    top_genes = ad_gene.obs.index[top_idx]
    top_scores = scores[top_idx]

    lines = [f"Top {top_n} expressed genes in {region_label} ({cells.shape[0]} cells):"]
    for rank, (gene, score) in enumerate(zip(top_genes, top_scores), 1):
        lines.append(f"  {rank:2d}. {gene:<20s}  score = {score:.4f}")
    return "\n".join(lines)


### ---------------- Tool 8: Query Gene Expression in Region ---------------- ###
@tool
def query_gene_expression_in_region(gene: str, tissue_main: str) -> str:
    """Compute descriptive statistics of imputed expression for ONE specific gene
    across all cells in a given tissue_main region.

    Use top_expressed_genes_in_region instead if you want to discover which genes
    are most expressed (this tool requires you to already know the gene name).

    Parameters
    ----------
    gene : str
        Uppercase mouse gene symbol (e.g. 'GAD1', 'MBP').
    tissue_main : str
        Valid tissue_main atlas code (e.g. 'HPF_CA', 'CTX_1').

    Returns mean, std, min, max, median, and percent of cells expressing the gene.
    """
    gene = gene.upper()

    if gene not in ad_gene.obs.index:
        return (f"Error: Gene '{gene}' not found in the atlas "
                f"({len(ad_gene.obs.index)} mouse genes available). "
                f"Check spelling and use uppercase gene symbols.")

    valid_regions = set(ad_cell.obs['tissue_main'].unique())
    if tissue_main not in valid_regions:
        return (f"Error: '{tissue_main}' is not a valid tissue_main code. "
                f"Valid values: {sorted(valid_regions)}")

    region_cells = ad_cell[ad_cell.obs['tissue_main'] == tissue_main]
    gene_exp = np.asarray(region_cells.X @ ad_gene[gene].X.T)[:, 0].flatten()

    return (
        f"Gene {gene} expression in region {tissue_main}:\n"
        f"  Cells: {len(gene_exp)}\n"
        f"  Mean: {gene_exp.mean():.4f}\n"
        f"  Std:  {gene_exp.std():.4f}\n"
        f"  Min:  {gene_exp.min():.4f}\n"
        f"  Max:  {gene_exp.max():.4f}\n"
        f"  Median: {np.median(gene_exp):.4f}\n"
        f"  % cells expressing (>0): {(gene_exp > 0).mean() * 100:.1f}%"
    )


### ---------------- Tool 9: Compute Gene-Gene Correlation ---------------- ###
@tool
def compute_gene_correlation(gene1: str, gene2: str, tissue_main: str = None) -> str:
    """Compute Pearson correlation between two genes across atlas cells.

    Parameters
    ----------
    gene1, gene2 : str
        Uppercase mouse gene symbols (e.g. 'GAD1', 'GAD2').
    tissue_main : str, optional
        Valid tissue_main atlas code. If omitted, 50 000 cells are sampled from
        the full atlas for performance.

    Returns Pearson r, p-value, and number of cells analyzed.
    """
    from scipy import stats

    gene1, gene2 = gene1.upper(), gene2.upper()

    for g in [gene1, gene2]:
        if g not in ad_gene.obs.index:
            return (f"Error: Gene '{g}' not found in the atlas "
                    f"({len(ad_gene.obs.index)} mouse genes available). "
                    f"Check spelling and use uppercase gene symbols.")

    if tissue_main:
        valid_regions = set(ad_cell.obs['tissue_main'].unique())
        if tissue_main not in valid_regions:
            return (f"Error: '{tissue_main}' is not a valid tissue_main code. "
                    f"Valid values: {sorted(valid_regions)}")
        cells = ad_cell[ad_cell.obs['tissue_main'] == tissue_main]
        region_str = f" in region {tissue_main}"
    else:
        if ad_cell.shape[0] > 50000:
            idx = np.random.choice(ad_cell.shape[0], 50000, replace=False)
            cells = ad_cell[idx]
        else:
            cells = ad_cell
        region_str = f" (sampled {cells.shape[0]} cells from full atlas)"

    exp1 = np.asarray(cells.X @ ad_gene[gene1].X.T)[:, 0].flatten()
    exp2 = np.asarray(cells.X @ ad_gene[gene2].X.T)[:, 0].flatten()

    r, p = stats.pearsonr(exp1, exp2)
    return (
        f"Pearson correlation between {gene1} and {gene2}{region_str}:\n"
        f"  r = {r:.4f}\n"
        f"  p-value = {p:.2e}\n"
        f"  Cells analyzed: {len(exp1)}"
    )


### ---------------- Tool 10: Extract Gene Expression as h5ad ---------------- ###
@tool
def extract_gene_expression_h5ad(
    gene_list: List[str],
    output_dir: str = "output",
    cell_type_col: str = "main_STARmap",
    region_col: str = "tissue_main",
    log: Annotated[callable, "Logger function"] = print,
) -> str:
    """Extract imputed gene expression for a list of genes and save as an h5ad file.

    Computes imputed expression via dot product of cell embeddings and gene embeddings:
        imputed_expr = ad_cell.X @ ad_gene[gene_list].X.T   (n_cells, n_genes)

    The resulting AnnData contains:
      - X: imputed expression matrix (n_cells × n_genes)
      - obs: cell metadata (cell type, tissue region, spatial coordinates, section)
      - var: gene names as index

    Parameters
    ----------
    gene_list : List[str]
        List of uppercase mouse gene symbols (e.g. ['VIP', 'MBP', 'SATB2']).
        Genes not found in the atlas will be skipped with a warning.
    output_dir : str
        Base directory to save the output. File will be saved in [output_dir]/data/.
    cell_type_col : str
        Which cell type column to include in obs
        ('main_STARmap' or 'sub_STARmap', default: 'main_STARmap').
    region_col : str
        Which region column to include in obs
        ('tissue_main' or 'tissue_sub', default: 'tissue_main').

    Returns the path to the saved .h5ad file.
    """
    import pandas as _pd

    output_dir = _validate_output_dir(output_dir)
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Validate and filter gene list
    gene_list = [g.upper() for g in gene_list]
    valid_genes = [g for g in gene_list if g in ad_gene.obs.index]
    invalid_genes = [g for g in gene_list if g not in ad_gene.obs.index]

    if invalid_genes:
        log(f"⚠️ Genes not found in atlas (skipped): {invalid_genes}")

    if not valid_genes:
        return (f"Error: None of the requested genes were found in the atlas. "
                f"Invalid genes: {gene_list}. "
                f"Please use uppercase mouse gene symbols (e.g. 'VIP', 'MBP').")

    log(f"🧬 Extracting imputed expression for {len(valid_genes)} genes: {valid_genes}")

    # Compute imputed expression: cell_emb @ gene_emb.T
    gene_emb = ad_gene[valid_genes].X      # (n_genes, 64)
    imputed = ad_cell.X @ gene_emb.T      # (n_cells, n_genes)

    # Select obs columns to include
    obs_cols = ['ap_order', 'x', 'y']
    for col in [cell_type_col, region_col, 'sub_STARmap', 'tissue_sub',
                'global_x', 'global_y', 'global_z']:
        if col in ad_cell.obs.columns and col not in obs_cols:
            obs_cols.append(col)

    obs_df = ad_cell.obs[[c for c in obs_cols if c in ad_cell.obs.columns]].copy()

    # Build AnnData
    result_adata = ad.AnnData(
        X=imputed,
        obs=obs_df,
        var=_pd.DataFrame(index=valid_genes),
    )

    # Generate filename from gene list
    gene_tag = "_".join(valid_genes[:5])
    if len(valid_genes) > 5:
        gene_tag += f"_and_{len(valid_genes) - 5}_more"
    fname = f"gene_expression_{gene_tag}.h5ad"
    out_path = os.path.join(data_dir, fname)

    result_adata.write_h5ad(out_path)
    log(f"\u2705 Saved gene expression h5ad with {result_adata.shape[0]:,} cells \u00d7 {result_adata.shape[1]} genes to {out_path}")

    # Capture shape and path info before releasing the AnnData from memory.
    # result_adata + imputed can be several GB for a full-atlas extraction;
    # releasing them immediately keeps Streamlit session memory usage flat.
    _n_cells, _n_genes = result_adata.shape
    import gc as _gc
    del result_adata, imputed, gene_emb, obs_df
    _gc.collect()

    return (
        f"Extracted imputed expression for {len(valid_genes)} genes "
        f"({_n_cells:,} cells \u00d7 {_n_genes} genes).\n"
        f"Saved to: {out_path}\n"
        f"Genes included: {valid_genes}\n"
        + (f"Genes skipped (not in atlas): {invalid_genes}\n" if invalid_genes else "")
    )
