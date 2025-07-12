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

# Fix numpy compatibility with older scanpy
np.float_ = np.float64


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
def plot_region_distribution(region_id: List[str], gene_id: List[str] = None, log: Annotated[callable, "Logger function"] = print) -> str:
    """Given a list of atlas tissue_main regions, 3D scatter plot of cells in each region, colored by subregion.
    Optionally, input also may include a list of genes to plot. In this case, there will be separate plots for 3D gene expression."""

    output_string = ""

    subset = ad_cell[ad_cell.obs['ap_order'] < 404]
    for region in region_id:
        log(f"📍 Plotting region: {region}")
        region_cells = subset[subset.obs['tissue_main'] == region]
        if region_cells.shape[0] == 0:
            log(f"⚠️ No cells found for region: {region}")
            continue
        fig = px.scatter_3d(
            region_cells.obs,
            x='global_x', y='global_y', z='global_z',
            color='tissue_sub',
            title=f'3D Cell Distribution in {region}',
            color_discrete_sequence=px.colors.qualitative.Plotly  # or Dark24, Light24, etc.

        )
        fig.update_traces(marker_size=1)
        fig.write_html(f"output/figures/Brain_region_{region}_subregion_3D.html")

        output_string += f"Saved 3D cell distribution plot for {region} in output/figures/Brain_region_{region}_subregion_3D.html.\n"
        log(f"✅ Saved 3D cell distribution plot for {region}")


        if gene_id:
            for gene in gene_id:
                gene = gene.upper()
                if gene in ad_gene.obs.index:
                    log(f"🎯 Plotting gene {gene} in region {region}")
                    ad_gene_target = ad_gene[gene]
                    gene_exp = region_cells.X @ ad_gene_target.X.T
                    ad_gene_exp = ad.AnnData(X=gene_exp)
                    ad_gene_exp.obs = region_cells.obs
                    ad_gene_exp.var.index = [gene]
                    gene_vals = gene_exp[:, 0]
                    fig = px.scatter_3d(
                        ad_gene_exp.obs,
                        x='global_x', y='global_y', z='global_z',
                        color=gene_vals,
                        title=f'3D Gene Expression of {gene} in {region}',
                        color_continuous_scale=px.colors.sequential.Reds,  # or 'Plasma', 'Inferno', etc.
                        range_color=[gene_vals.min(), gene_vals.max()]        # optional: fixes the colorbar range
                    )
                    fig.update_traces(marker_size=1)
                    fig.write_html(f"output/figures/Brain_region_{region}_gene_{gene}_3D.html")

                    log(f"✅ Saved 3D gene expression plot for {gene}")
                    output_string += f"Saved 3D gene expression plot for {gene} in output/figures/Brain_region_{region}_gene_{gene}_3D.html.\n"
                        
    return output_string



### ---------------- Tool 4: Find Relevant 2D Sections ---------------- ###
@tool
def find_section_ids(cell_types: List[str], brain_regions: List[str], log: Annotated[callable, "Logger function"] = print) -> List[float]:
    """Given a list of cell types and brain regions, 
    the list cannot be empty, 
    find all 2D sections with given cell types and brain regions."""
    if  len(cell_types)==0: 
        log("⚠️ Empty cell types list provided.")
        return "No matching sections found. Please provide valid cell types list."

    if len(brain_regions)==0:
        log("⚠️ Empty brain regions list provided.")
        return "No matching sections found. Please provide valid brain regions list."

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
    table.to_csv(f"output/data/2D_section_ids.csv")
    log("✅ Saved matched section IDs to `output/data/2D_section_ids.csv`\n\n")

    try:
        import streamlit as st
        with st.expander("📄 Matched Section Table", expanded=True):
            st.dataframe(table)
    except:
        log("⚠️ Streamlit display failed (not in Streamlit context).")

    return f"Found {len(matched)} matching sections in the spatial brain atlas: {matched}.\n\nSaved percentage of cell types in brain regions table to output/data/2D_section_ids.csv"



### ---------------- Tool 5: Read and Plot One Section ---------------- ###
@tool
def read_one_section(section_id: str, target_gene: List[str] = None, log: Annotated[callable, "Logger function"] = print) -> str:
    """Given a section ID, extract the spatial transcriptomic data of the section and visualize gene expression and annotations for one AP section.
    The section_id either come from results of Tool find_section_ids or from the user input.
    Optionally, input also may include a list of target genes to plot. In this case, there will be separate plots for gene expression."""
    output_string = ""
    sid = float(section_id)
    subset = ad_cell[ad_cell.obs['ap_order'] == sid]
    gene_exp = subset.X @ ad_gene.X.T
    ad_gene_exp = ad.AnnData(X=gene_exp)
    ad_gene_exp.obs = subset.obs
    ad_gene_exp.var.index = ad_gene.obs.index
    ad_gene_exp.write_h5ad(f"output/data/adata_Section_{section_id}.h5ad")

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
                plt.savefig(f"output/figures/Section_{section_id}_gene_{gene}.png",dpi=300)
                output_string += f"Saved gene expression plot for {gene} in output/figures/Section_{section_id}_gene_{gene}.png.\n"
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
    plt.savefig(f"output/figures/Section_{section_id}_cell_type.png",dpi=300)
    output_string += f"Saved cell types plot for {section_id} in output/figures/Section_{section_id}_cell_type.png.\n"
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
    plt.savefig(f"output/figures/Section_{section_id}_brain_region.png",dpi=300)
    output_string += f"Saved brain regions plot for {section_id} in output/figures/Section_{section_id}_brain_region.png.\n"
    plt.close()

    return f"Saved Section {section_id} in the spatial brain atlas to output/data/adata_Section_{section_id}.h5ad. {output_string}"


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
