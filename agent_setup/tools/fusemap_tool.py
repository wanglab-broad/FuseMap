### ---fusemap_tools.py--- ###
import os
import pandas as pd
import numpy as np
import re
import scanpy as sc
import plotly.express as px
import anndata as ad
import matplotlib.pyplot as plt
from typing import Annotated, List
import pandas as pd
import numpy as np
import scanpy as sc
import ast
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import scanpy as sc
from easydict import EasyDict as edict
from fusemap import seed_all, ModelType, setup_logging
import logging
import pandas as pd
from time import time
import os
import pandas as pd
import numpy as np
import os
import scanpy as sc
from easydict import EasyDict as edict
from fusemap import seed_all, spatial_map, spatial_integrate,transfer_celltype
import copy
seed_all(0)
start_time = time()
import logging
logger = logging.getLogger()  # Or get the specific logger used in the function
logger.setLevel(logging.WARNING)
import scanpy.external as sce
from langchain.tools import BaseTool
from langchain.base_language import BaseLanguageModel

# Fix numpy compatibility with older scanpy
np.float_ = np.float64



### ---------------- Tool 1: map to molCCF ---------------- ###
@tool
def map_molCCF(path: str, section_IDs: List[str], 
               transfer_main_level: bool,
               transfer_sub_level: bool,
               log=print) -> str:
    """Given a path to a directory that has query spatial transcriptomics, 
    the matched section IDs, 
    transfer_main_level, a bool value where true indicates transfer main level cell types and false indicates not to transfer,
    transfer_sub_level, a bool value where true indicates transfer sub level cell types and false indicates not to transfer,
    the tool will map each sample to the molCCF. output will be saved in output/fusemap/[input directory name]/molCCF_mapping/."""
    
    import os

    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "map_molCCF", f"Starting molCCF mapping for path: {path}, sections: {section_IDs}")
    except ImportError:
        pass

    print(os.getcwd())
    last_folder = os.path.basename(os.path.normpath(path))  
    section_IDs = [str(i) for i in section_IDs]


    pretrain_model_path = "./molCCF"
    output_save_dir = f"./output/fusemap/{last_folder}/molCCF_mapping/"
    ### if output dir already exists, continue
    # if os.path.exists(output_save_dir):
    #     return f"New query data mapped to molCCF and results saved in {output_save_dir}."

    args = edict(dict(pretrain_model_path=pretrain_model_path,
                    output_save_dir=output_save_dir,
                    use_llm_gene_embedding="false",
                    keep_celltype='',
                    keep_tissueregion='',
                    ))

    data_dir_list = os.listdir(path)
    data_dir_list=[path+'/'+i for i in data_dir_list]

    X_input = []
    for ind, data_dir in enumerate(data_dir_list):
        print(f"Loading {data_dir}")
        data = sc.read_h5ad(data_dir)
        # Handle spatial coordinates
        if "x" not in data.obs.columns:
            if "col" in data.obs.columns and "row" in data.obs.columns:
                data.obs["x"] = data.obs["col"]
                data.obs["y"] = data.obs["row"]
            elif "spatial" in data.obsm.keys():
                data.obs["x"] = data.obsm["spatial"][:,0]
                data.obs["y"] = data.obsm["spatial"][:,1]
            elif 'Raw_Slideseq_X' in data.obs.columns:
                data.obs['x'] = data.obs['Raw_Slideseq_X']
                data.obs['y'] = data.obs['Raw_Slideseq_Y']
            else:
                raise ValueError(f"Please provide spatial coordinates in the obs['x'] and obs['y'] columns for {data_dir}")

        # Add metadata
        data.obs['name'] = f'section{ind}'
        data.obs['file_name'] = os.path.basename(data_dir)
        print(f"Loaded {data.shape[0]} cells with {data.shape[1]} genes from {data.obs['file_name'].iloc[0]}")
        X_input.append(data)

    # Set parameters for integration
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)
    print(f"Loaded {len(X_input)} datasets")

    ### start to map
    for i in range(len(X_input)):
        args_i=copy.copy(args)
        args_i.output_save_dir = os.path.join(args.output_save_dir, X_input[i].obs['file_name'].unique()[0])
        spatial_map([X_input[i]], args_i, [kneighbor[i]], [input_identity[i]])
    
    if transfer_main_level:
        ### read reference single-cell embeddings
        cell_path = './molCCF/ad_embed_single_cell.h5ad'
        ad_cell = sc.read_h5ad(cell_path)

        ### select certain samples to map
        ad_cell_subset = ad_cell[ad_cell.obs['ap_order'].isin(section_IDs)]

        ### transfer main-level cell types
        main_level_key = 'transfer_gt_cell_type_main_STARmap'
        ad_cell_subset = ad_cell_subset[ad_cell_subset.obs[main_level_key]!='nan']
        
        ### random select 10k cells
        if ad_cell_subset.shape[0] > 50000:
            ad_cell_subset = ad_cell_subset[np.random.choice(ad_cell_subset.shape[0], 50000, replace=False)]

        data_dir_list = os.listdir(path)
        for name in data_dir_list:  
            ad_embed_control = sc.read_h5ad(f'{output_save_dir}/{name}/ad_celltype_embedding.h5ad')
            ad_embed_control=transfer_celltype(ad_cell_subset, main_level_key, ad_embed_control)
            main_celltype = ad_embed_control.obs['predicted_celltype'].value_counts()[ad_embed_control.obs['predicted_celltype'].value_counts()>50].index
            if transfer_sub_level:
                sub_level_key = 'transfer_gt_cell_type_sub_STARmap'
                ad_embed_control.obs['predicted_celltype_sub'] = ad_embed_control.obs['predicted_celltype'].copy()
                for i in main_celltype:
                    ad_embed_control.obs['predicted_celltype_sub'] = ad_embed_control.obs['predicted_celltype_sub'].astype('str')
                    ad_embed_control_sub = ad_embed_control[ad_embed_control.obs['predicted_celltype']==i]
                    ad_cell_subset_sub = ad_cell_subset[ad_cell_subset.obs[main_level_key]==i]
                    ad_embed_control_sub = transfer_celltype(ad_cell_subset_sub, sub_level_key, ad_embed_control_sub)
                    ad_embed_control.obs.loc[ad_embed_control_sub.obs.index, 'predicted_celltype_sub'] = ad_embed_control_sub.obs['predicted_celltype']
            ad_embed_control.write_h5ad(f'{output_save_dir}/{name}/ad_celltype_embedding.h5ad')
        

        ### transfer spatial regions
        cell_path = './molCCF/ad_embed_spatial.h5ad'
        ad_cell = sc.read_h5ad(cell_path)
        ad_cell.obs['ap_order']=ad_cell.obs['ap_order'].astype('str')
       
        ### select certain samples to map
        ad_cell_subset = ad_cell[ad_cell.obs['ap_order'].isin(section_IDs)]

        ### transfer main-level cell types
        main_level_key = 'transfer_gt_tissue_region_main_STARmap'
        ad_cell_subset = ad_cell_subset[ad_cell_subset.obs[main_level_key]!='nan']

        ### random select 10k cells
        if ad_cell_subset.shape[0] > 50000:
            ad_cell_subset = ad_cell_subset[np.random.choice(ad_cell_subset.shape[0], 50000, replace=False)]

        data_dir_list = os.listdir(path)
        for name in data_dir_list:  
            ad_embed_control = sc.read_h5ad(f'{output_save_dir}/{name}/ad_tissueregion_embedding.h5ad')
            ad_embed_control=transfer_celltype(ad_cell_subset, main_level_key, ad_embed_control, assign_key = 'predicted_tissueregion')
            main_celltype = ad_embed_control.obs['predicted_tissueregion'].value_counts()[ad_embed_control.obs['predicted_tissueregion'].value_counts()>50].index

            if transfer_sub_level:
                sub_level_key = 'transfer_gt_tissue_region_sub_STARmap'
                ad_embed_control.obs['predicted_tissueregion_sub'] = ad_embed_control.obs['predicted_tissueregion'].copy()
                for i in main_celltype:
                    ad_embed_control.obs['predicted_tissueregion_sub'] = ad_embed_control.obs['predicted_tissueregion_sub'].astype('str')
                    ad_embed_control_sub = ad_embed_control[ad_embed_control.obs['predicted_tissueregion']==i]
                    ad_cell_subset_sub = ad_cell_subset[ad_cell_subset.obs[main_level_key]==i]

                    ad_embed_control_sub = transfer_celltype(ad_cell_subset_sub, sub_level_key, ad_embed_control_sub,  assign_key = 'predicted_tissueregion_sub')
                    ad_embed_control.obs.loc[ad_embed_control_sub.obs.index, 'predicted_tissueregion_sub'] = ad_embed_control_sub.obs['predicted_tissueregion_sub']

            ad_embed_control.write_h5ad(f'{output_save_dir}/{name}/ad_tissueregion_embedding.h5ad')
        
    result = f"New query data mapped to molCCF and results saved in {output_save_dir}."
    
    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "map_molCCF", result)
    except ImportError:
        pass
    
    return result



### ---------------- Tool 2: spatially integrate new datasets ---------------- ###
@tool
def fusemap_integrate(path: str, description, log=print) -> str:
    """Given a path to a directory that has query spatial transcriptomics, 
    description of the query dataset, the tool will use FuseMap to spatially integrate datasets. 
    Output will be saved in output/fusemap/[input directory name]/spatial_integrate/."""

    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "fusemap_integrate", f"Starting integration for path: {path}, description: {description}")
    except ImportError:
        pass

    data_dir_list = [path+'/'+f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    last_folder = os.path.basename(os.path.normpath(path))
    output_dir = f'output/fusemap/{last_folder}/integrate/'
    ### make dir if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # ### if output dir already exists, continue
    # if os.path.exists(output_dir):
    #     return f"New query data are integrated by FuseMap and results are saved in {output_dir}."

    args = edict(dict(output_save_dir=output_dir,
                    keep_celltype="",
                    keep_tissueregion="",
                    use_llm_gene_embedding="false",
                    pretrain_model_path=""))

    setup_logging(args.output_save_dir)

    arg_dict = vars(args)
    dict_pd = {}
    for keys in arg_dict.keys():
        dict_pd[keys] = [arg_dict[keys]]
    pd.DataFrame(dict_pd).to_csv(os.path.join(args.output_save_dir, "config.csv"), index=False)
    logging.info("\n\n\033[95mArguments:\033[0m \n%s\n\n", vars(args))
    logging.info("\n\n\033[95mArguments:\033[0m \n%s\n\n", vars(ModelType))

    X_input = []
    for ind, data_dir in enumerate(data_dir_list):
        print(f"Loading {data_dir}")
        data = sc.read_h5ad(data_dir)
        data.obs['input_data_path']=data_dir
        # Handle spatial coordinates
        if "x" not in data.obs.columns:
            if "col" in data.obs.columns and "row" in data.obs.columns:
                data.obs["x"] = data.obs["col"]
                data.obs["y"] = data.obs["row"]
            elif "spatial" in data.obsm.keys():
                data.obs["x"] = data.obsm["spatial"][:,0]
                data.obs["y"] = data.obsm["spatial"][:,1]
            else:
                raise ValueError(f"Please provide spatial coordinates in the obs['x'] and obs['y'] columns for {data_dir}")

        # Add metadata
        data.obs['name'] = f'section{ind}'
        data.obs['file_name'] = os.path.basename(data_dir)
        print(f"Loaded {data.shape[0]} cells with {data.shape[1]} genes from {data.obs['file_name'].iloc[0]}")
        X_input.append(data)

    # Set parameters for integration
    kneighbor = ["delaunay"] * len(X_input)
    input_identity = ["ST"] * len(X_input)
    print(f"Loaded {len(X_input)} datasets")

    spatial_integrate(X_input, args, kneighbor, input_identity)
    print(f"Time elapsed: {(time() - start_time) / 60:.2f} s")

    result = f"New query data are integrated by FuseMap and results are saved in {output_dir}."
    
    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "fusemap_integrate", result)
    except ImportError:
        pass
    
    return result



### ---------------- Tool 3: finalize main-level cell types ---------------- ###
def get_marker_gene_across_tissue(ad_embed, data_dir_list, key='leiden', num=5):
    dict_marker = {}
    for i in ad_embed.obs[key].unique():
        dict_marker[i] = []

    for path_i in data_dir_list:
        adata1 = sc.read_h5ad(path_i)

        intersect_index = np.intersect1d(adata1.obs.index, ad_embed.obs.index)
        adata1 = adata1[intersect_index, :]
        adata1.obs['cluster'] = ad_embed.obs.loc[intersect_index, key]

        sc.pp.normalize_total(adata1)
        sc.pp.log1p(adata1)

        while True:
            try:
                sc.tl.rank_genes_groups(adata1, 'cluster', method='t-test')
                break
            except ValueError as e:
                # Only handle the specific error about groups with one sample
                msg = str(e)
                if "since they only contain one sample" in msg:
                    # Extract group names using regex
                    bad_groups = re.findall(r'groups (.*?) since', msg)
                    bad_groups_list = [g.strip() for g in bad_groups[0].split(',')]
                adata1 = adata1[~adata1.obs['cluster'].isin(bad_groups_list)]

        
        temp = pd.DataFrame(adata1.uns['rank_genes_groups']['names']).head(num)
        temp_score = pd.DataFrame(adata1.uns['rank_genes_groups']['scores']).head(num)

        for i in range(temp.shape[1]):
            curr_col = temp.iloc[:, i].to_list()
            curr_col_score = temp_score.iloc[:, i].to_list()
            list_true = [x > 0 for x in curr_col_score]
            curr_col = list(np.array(curr_col)[list_true])
            dict_marker[temp.columns[i]].append(curr_col)
            
        for left_other in ad_embed.obs[key].unique():
            if left_other not in temp.columns:
                dict_marker[left_other].append([])

    return dict_marker

def create_maintype_matcher(llm):
    region_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a biological expert.
        Your task is to decide the cell type in {tissue_description} based on gene markers.
        Output must be one from the cell type candidates and only the cell type name.
        """),
        
        ("user", """Between these cell type candidaes {candidates}, based on differently expressed genes {genes_across_sample}, what is the cell type?
        
        """)
    ])
    return region_prompt | llm

def create_mainregion_matcher(llm):
    region_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a biological expert.
        Your task is to decide the tissue region in {tissue_description} based on gene markers.
        Output must be one from the tissue region candidates and only the tissue region name.
        """),
        
        ("user", """Between these tissue region candidaes {candidates}, based on differently expressed genes {genes_across_sample}, what is the tissue region?
        """)
    ])
    return region_prompt | llm

def create_mainregion_matcher_no_mapped(llm):
    region_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a biological expert.
        Your task is to decide the tissue region in {tissue_description} based on gene markers.
        Output must be one from the tissue region candidates and only the tissue region name.
        """),
        
        ("user", """Based on differently expressed genes {genes_across_sample}, what is the tissue region?
        """)
    ])
    return region_prompt | llm


def finalize_mainlevel(data_path,
                        map_path,
                        integrate_path,
                        tissue_description,
                        key_select,
                        llm,
                        log=print) -> str:
    
    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "finalize_mainlevel", f"Starting main-level finalization for {tissue_description}, key: {key_select}")
    except ImportError:
        pass
    
    adata_output_save_path = './output/data/annotated_user_data.h5ad'
    ### if folder not exist, create it
    os.makedirs(os.path.dirname(adata_output_save_path), exist_ok=True)

    if not os.path.exists(adata_output_save_path):
        ### Case 1: normal mouse brain, use all transferred results
        if map_path is not None and integrate_path is None:
            data_dir_list = os.listdir(map_path)
            adata_integrate=[]
            for name in data_dir_list:  
                print(map_path+'/'+name)
                adata_map = sc.read_h5ad(map_path+'/'+name+ f'/ad_{key_select}_embedding.h5ad')
                adata_map.obs['file_name'] = name
                adata_map.obs[f'main_level_{key_select}']=list(adata_map.obs[f'predicted_{key_select}'])        
                adata_map.obs[f'sub_level_{key_select}']=list(adata_map.obs[f'predicted_{key_select}_sub'])      
                adata_map.obs = adata_map.obs.drop(columns=[f'predicted_{key_select}', f'predicted_{key_select}_sub'])
                adata_integrate.append(adata_map)
            adata_integrate=sc.concat(adata_integrate)
            # adata_integrate.write_h5ad(adata_output_save_path)

        ### Case 2:  mouse brain with disease cell states, use integrated results
        if map_path is not None and integrate_path is not None:
            ### clustering on FuseMap embeddings
            adata_integrate = sc.read_h5ad(integrate_path+f'/ad_{key_select}_embedding.h5ad')
            sc.tl.pca(adata_integrate,n_comps=30)
            sce.pp.harmony_integrate(adata_integrate, 'batch')
            sc.pp.neighbors(adata_integrate,n_neighbors=50,use_rep='X_pca_harmony')
            sc.tl.umap(adata_integrate)
            sc.tl.leiden(adata_integrate,resolution=0.5)

            ### transfer molCCF mapped cell types
            data_dir_list = os.listdir(map_path)
            for name in data_dir_list:  
                print(map_path+'/'+name)
                adata_map = sc.read_h5ad(map_path+'/'+name+ f'/ad_{key_select}_embedding.h5ad')
                adata_integrate.obs.loc[adata_map.obs.index,f'map_{key_select}']=list(adata_map.obs[f'predicted_{key_select}'])        

            ### get marker genes in each clusters
            data_dir_list = [data_path+'/'+f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

            dict_marker = get_marker_gene_across_tissue(adata_integrate, data_dir_list,
                                                        key='leiden', num=5)

            pd_marker = pd.DataFrame(dict_marker)
            pd_marker.index = data_dir_list
            pd_marker.to_csv(f'./output/data/main_level_{key_select}.csv')


            ### compute mapped cell types per cluster
            mapping_dict = {}
            for i in adata_integrate.obs['leiden'].unique():
                mapping_dict[i]=[]
                for j in adata_integrate.obs['batch'].unique():
                    adata_integrate_sub = adata_integrate[adata_integrate.obs['batch']==j]
                    sub_df = adata_integrate_sub.obs[adata_integrate.obs['leiden'] == i]
                    most_frequent = sub_df[f'map_{key_select}'].mode()[0]
                    # print(i,j,most_frequent)
                    mapping_dict[i].append(most_frequent)

            ### use LLM to decide main cell type for each cluster
            if key_select=='celltype':
                matcher=create_maintype_matcher(llm)
            elif key_select=='tissueregion':
                matcher=create_mainregion_matcher(llm)
            dict_mappp={}
            for i in dict_marker.keys():
                response = matcher.invoke({
                    "tissue_description": tissue_description,
                    "candidates": mapping_dict[i],
                    "genes_across_sample": dict_marker[i]
                })
                # print(response.content)
                dict_mappp[i]=response.content
            
            ### keep the main cell type for each cluster
            adata_integrate.obs[f'main_level_{key_select}'] = adata_integrate.obs['leiden'].map(dict_mappp)
            try:
                adata_integrate.obs[f'sub_level_{key_select}'] = 'unassigned'
            except:
                pass
            # Remove 'leiden' and 'map_cell_type' columns from adata_integrate.obs
            adata_integrate.obs = adata_integrate.obs.drop(columns=['leiden', f'map_{key_select}'])


        ### Case 3: new sample with totally different cell types from normal mouse brain
        if map_path is None and integrate_path is not None:
            ### clustering on FuseMap embeddings
            adata_integrate = sc.read_h5ad(integrate_path+f'/ad_{key_select}_embedding.h5ad')
            sc.tl.pca(adata_integrate,n_comps=30)
            sce.pp.harmony_integrate(adata_integrate, 'batch')
            sc.pp.neighbors(adata_integrate,n_neighbors=50,use_rep='X_pca_harmony')
            sc.tl.umap(adata_integrate)
            sc.tl.leiden(adata_integrate,resolution=0.5)


            ### get marker genes in each clusters
            data_dir_list = [data_path+'/'+f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
            dict_marker = get_marker_gene_across_tissue(adata_integrate, 
                                                        data_dir_list,
                                                        key='leiden',
                                                        num=5)


            pd_marker = pd.DataFrame(dict_marker)
            pd_marker.index = data_dir_list
            pd_marker.to_csv(f'./output/data/main_level_{key_select}.csv')


            matcher=create_mainregion_matcher_no_mapped(llm)
            dict_mappp={}
            for i in dict_marker.keys():
                response = matcher.invoke({
                    "tissue_description": tissue_description,
                    "genes_across_sample": dict_marker[i]
                })
                # print(response.content)
                dict_mappp[i]=response.content

            ### keep the main cell type for each cluster
            adata_integrate.obs[f'main_level_{key_select}'] = adata_integrate.obs['leiden'].map(dict_mappp)
            try:
                adata_integrate.obs[f'sub_level_{key_select}'] = 'unassigned'
            except:
                pass


        adata_integrate.write_h5ad(adata_output_save_path)

    else:
        adata_integrate=sc.read_h5ad(adata_output_save_path)

    ### plot main-level cell types
    cell_types = adata_integrate.obs[f'main_level_{key_select}'].unique()   
    # Create a dictionary mapping cell types to random colors
    colors = {}
    for cell_type in cell_types:
        colors[cell_type] = '#%06x' % random.randint(0, 0xFFFFFF)
    try:
        fig,ax = plt.subplots(figsize=(10,10))
        ax=sc.pl.umap(adata_integrate,size=5, 
                    legend_loc='on data',color=f'main_level_{key_select}',
                    legend_fontsize=12,
                    palette = colors,
                    ax=ax,show=False)
        ax.axis('off')
        ax.set_title('')
        fig.savefig(f'./output/figures/user_data_main_level_{key_select}.png',dpi = 300,transparent=True)
    except:
        pass

    for i in adata_integrate.obs['file_name'].unique():
        adata_plot = adata_integrate[adata_integrate.obs['file_name']==i]
        plt.figure()
        plt.scatter(adata_plot.obs['y'].astype('float'),adata_plot.obs['x'].astype('float'),s=0.5,
                    c=[colors[i] for i in  adata_plot.obs[f'main_level_{key_select}']])
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig(f'./output/figures/user_data_spatial_{i}_main_level_{key_select}.png',dpi=300,transparent=True)

    result = f"Finalized adata saved in with main level cell types saved in column key 'main_level_{key_select}'. " \
    f"Plotted figures saved in './output/figures/user_data_main_level_{key_select}.png'."
    
    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "finalize_mainlevel", result)
    except ImportError:
        pass
    
    return result


class finalize_mainlevel_tool(BaseTool):
    name: str = "finalize_mainlevel"
    description: str = """Given a data_path to a directory that has query spatial transcriptomics, 
    a map_path where new query data mapped to molCCF and results saved,
    a integrate_path where new query data are integrated by FuseMap and results are saved,
    the description of the query dataset, 
    key_select is the key to select the cell type analysis or tissue region analysis, should be 'celltype' or 'tissueregion',
    the tool will gather information and finalize the main level annotation.
    if the tool of mapping to molCCF is not run, the map_path value should be None.
    if the tool of integrating is not run, the integrate_path value should be None.
    Output will be saved in output/data/ and output/figures."""
    llm: BaseLanguageModel= None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.llm = llm

    def _run(self, data_path: str, 
             map_path: str, 
             integrate_path: str, 
             tissue_description: str, 
             key_select: str, 
             log=print):
        return finalize_mainlevel(data_path, map_path, integrate_path, tissue_description, key_select, self.llm, log)
    



### ---------------- Tool 4: annotate sublevel cell types ---------------- ###


# @tool
def annotate_sublevel(data_path: str, 
                     tissue_type: str, 
                     map_path, 
                     integrate_path,
                     key_select,
                     llm, 
                     log):

    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "annotate_sublevel", f"Starting sub-level annotation for {tissue_type}, key: {key_select}")
    except ImportError:
        pass

    def create_subtype_matcher(llm):
        region_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biological expert who understands cell types.
            Your task is to annotate subtype of {tissue_type} under a main level cell type {main_level_cell_type} based on gene markers across tissue samples.
            """),
            
            ("user", """What are the subtypes of {tissue_type} under this main level cell type {main_level_cell_type} with gene markers {cluster_marker}?
            
            """)
        ])
        return region_prompt | llm

    def create_subregion_matcher(llm):
        region_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biological expert who understands tissue regions.
            Your task is to annotate sublevel tissue regions of {tissue_type} under a main level tissue region {main_level_cell_type} based on gene markers across tissue samples.
            """),
            
            ("user", """What are the sublevels of {tissue_type} under this main level tissue region {main_level_cell_type} with gene markers {cluster_marker}?
            
            """)
        ])
        return region_prompt | llm

    def convert_to_dict(llm):
        region_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            """Given a scientific input text describing different sublevel annotations of clusters, your task is to:
    1. Extract the annotation for each cluster as a concise sentence.
    2. Return a Python dictionary mapping the cluster index (as a string) to its annotation.

    Format your answer as a valid Python dict. Only return the dict, no extra text.
    """),
            ("user", """input text is: {input_text}""")
        ])
        return region_prompt | llm

    def extract_python_dict_from_response(text):
        # This regex extracts anything between ```python ... ```
        match = re.search(r"```python\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            # fallback: extract anything between triple backticks (no language)
            match = re.search(r"```[\w]*\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            dict_str = match.group(1)
        else:
            # If not found, try to find the first {...} block
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                dict_str = match.group(1)
            else:
                raise ValueError("No dictionary found in response text")
        # Now safely evaluate
        return ast.literal_eval(dict_str)
    
    
    adata_output_save_path = './output/data/annotated_user_data.h5ad'
    adata_integrate=sc.read_h5ad(adata_output_save_path)

    ### Case 1: normal mouse brain, use all transferred results
    if map_path is not None and integrate_path is None:
        all_main_level_celltype=adata_integrate.obs[f'main_level_{key_select}'].unique()
        ### find matched focus main type from all_main_level_celltype
        for focus_main_level_type in all_main_level_celltype:
            adata_integrate_subset = adata_integrate[adata_integrate.obs[f'main_level_{key_select}']==focus_main_level_type]
            
            ### plot subtype
            cell_types = adata_integrate_subset.obs[f'sub_level_{key_select}'].unique()   
            # Create a dictionary mapping cell types to random colors
            colors = {}
            for cell_type in cell_types:
                colors[cell_type] = '#%06x' % random.randint(0, 0xFFFFFF)

            for i in adata_integrate.obs['file_name'].unique():
                adata_plot = adata_integrate[adata_integrate.obs['file_name']==i]
                adata_plot_sub = adata_integrate_subset[adata_integrate_subset.obs['file_name']==i]
                plt.figure()
                plt.scatter(adata_plot.obs['y'].astype('float'),adata_plot.obs['x'].astype('float'),s=2,
                            c='lightgrey')
                plt.scatter(adata_plot_sub.obs['y'].astype('float'),adata_plot_sub.obs['x'].astype('float'),s=5,
                            c=[colors[i] for i in  adata_plot_sub.obs[f'sub_level_{key_select}']])
                plt.gca().invert_yaxis()
                plt.axis('off')
                plt.savefig(f'./output/figures/user_data_sub_{focus_main_level_type}_spatial_{i}.png',dpi=300,transparent=True)


    ### Case 2: mouse brain with disease cell states, use integrated results
    ### and Case 3: new sample with totally different cell types from normal mouse brain
    if integrate_path is not None:
        all_main_level_celltype=adata_integrate.obs[f'main_level_{key_select}'].unique()
        ### find matched focus main type from all_main_level_celltype
        for focus_main_level_type in all_main_level_celltype:
            adata_integrate_subset = adata_integrate[adata_integrate.obs[f'main_level_{key_select}']==focus_main_level_type]
            if sum(adata_integrate_subset.obs[f'sub_level_{key_select}']=='unassigned')==0:
                continue
            sc.tl.pca(adata_integrate_subset,n_comps=30)
            sce.pp.harmony_integrate(adata_integrate_subset, 'batch')
            sc.pp.neighbors(adata_integrate_subset,n_neighbors=50,use_rep='X_pca_harmony')
            sc.tl.umap(adata_integrate_subset)
            sc.tl.leiden(adata_integrate_subset, resolution=0.3)

            data_dir_list = [data_path+'/'+f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
            dict_marker = get_marker_gene_across_tissue(adata_integrate_subset, data_dir_list,num=10)

            if key_select=='celltype':
                matcher=create_subtype_matcher(llm)
            elif key_select=='tissueregion':
                matcher=create_subregion_matcher(llm)
            response = matcher.invoke({
                "cluster_marker": dict_marker,
                "main_level_cell_type": focus_main_level_type,
                "tissue_type": tissue_type
            })


            matcher = convert_to_dict(llm)
            response = matcher.invoke({
                "input_text": response.content
            })
            dict_map = extract_python_dict_from_response(response.content)


            adata_integrate_subset.obs[f'sub_level_{key_select}'] = adata_integrate_subset.obs['leiden'].map(dict_map)
            dict_marker = get_marker_gene_across_tissue(adata_integrate_subset, 
                                                        data_dir_list,
                                                        key = f'sub_level_{key_select}', 
                                                        num=10)
            
            ### save adata
            adata_integrate.obs[f'sub_level_{key_select}']=adata_integrate.obs[f'sub_level_{key_select}'].astype(str)
            adata_integrate.obs.loc[adata_integrate.obs[f'main_level_{key_select}']==focus_main_level_type,f'sub_level_{key_select}']=list(adata_integrate_subset.obs[f'sub_level_{key_select}'])
            adata_integrate.write_h5ad(f'./output/data/annotated_user_data.h5ad')

            pd_marker = pd.DataFrame(dict_marker)
            pd_marker.index = data_dir_list
            pd_marker.to_csv(f'./output/data/subtype_{focus_main_level_type}.csv')


            ### plot subtype
            cell_types = adata_integrate_subset.obs[f'sub_level_{key_select}'].unique()   
            # Create a dictionary mapping cell types to random colors
            colors = {}
            for cell_type in cell_types:
                colors[cell_type] = '#%06x' % random.randint(0, 0xFFFFFF)

            fig,ax = plt.subplots(figsize=(10,10))
            ax=sc.pl.umap(adata_integrate_subset,size=5, 
                        legend_loc='on data',color=f'sub_level_{key_select}',
                        legend_fontsize=12,
                    s=50,
                    palette = colors,
                    ax=ax,show=False)
            ax.axis('off')
            ax.set_title('')
            fig.savefig(f'./output/figures/user_data_sub_{focus_main_level_type}.png',dpi = 300,transparent=True)

            for i in adata_integrate.obs['file_name'].unique():
                adata_plot = adata_integrate[adata_integrate.obs['file_name']==i]
                adata_plot_sub = adata_integrate_subset[adata_integrate_subset.obs['file_name']==i]
                plt.figure()
                plt.scatter(adata_plot.obs['y'].astype('float'),adata_plot.obs['x'].astype('float'),s=2,
                            c='lightgrey')
                plt.scatter(adata_plot_sub.obs['y'].astype('float'),adata_plot_sub.obs['x'].astype('float'),s=5,
                            c=[colors[i] for i in  adata_plot_sub.obs[f'sub_level_{key_select}']])
                plt.gca().invert_yaxis()
                plt.axis('off')
                plt.savefig(f'./output/figures/user_data_sub_{focus_main_level_type}_spatial_{i}.png',dpi=300,transparent=True)


    result = f"Annotated subtype {focus_main_level_type}. adata saved in {adata_output_save_path} of key 'sub_level_{key_select}'. \
        gene markers saved in f'./output/data/subtype_{focus_main_level_type}.csv'.\
    Figures saved in './output/figures/'."
    
    # Add to memory
    try:
        from fusemap_agent import add_to_memory
        add_to_memory("FuseMapTool", "annotate_sublevel", result)
    except ImportError:
        pass
    
    return result


class annotate_sublevel_tool(BaseTool):
    name: str = "annotate_sublevel"
    description: str =       """
    Input is a data_path to a directory that has query spatial transcriptomics, 
    tissue_type is the description of the query dataset,  
    a map_path where new query data mapped to molCCF and results saved,
    a integrate_path where new query data are integrated by FuseMap and results are saved,
    key_select is the key to select the cell type analysis or tissue region analysis, should be 'celltype' or 'tissueregion',
    llm is the language model instance to use for annotations,
    this tool will annotate the sublevel annotation of each main level annotation.
    if the tool of mapping to molCCF is not run, the map_path value should be None.
    if the tool of integrating is not run, the integrate_path value should be None.
    Output is will be saved in output/data/ and output/figures.
    """

    llm: BaseLanguageModel= None

    def __init__(self, llm: BaseLanguageModel):
        super().__init__()
        self.llm = llm

    def _run(self, data_path: str, 
             tissue_type: str, 
             map_path: str, 
             integrate_path: str, 
             key_select: str, 
             log=print):
        return annotate_sublevel(data_path, tissue_type, map_path, integrate_path, key_select, self.llm, log)
    
    