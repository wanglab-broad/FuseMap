### ---supervisor_tool.py--- ###
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

# Fix numpy compatibility with older scanpy
np.float_ = np.float64

@tool
def supervisor_tool(llm, action_history: str, log: Annotated[callable, "Logger function"] = print) -> str:
    """This tool will supervise the analysis."""
    ### list files under ./output/figures and plot
    import os
    import plotly.express as px
    import plotly.graph_objects as go

    # List all files in the output/figures directory and plot, there are files in .html format and .png format
    figures_dir = "output/figures"  
    files = os.listdir(figures_dir)
    for file in files:
        if file.endswith(".html"):
            log(f"📊 Plotting {file}")
            fig = go.Figure(px.data.iris())
            fig.write_html(os.path.join(figures_dir, file))
        elif file.endswith(".png"):
            log(f"📊 Plotting {file}")
            fig = go.Figure(px.data.iris())
            fig.write_html(os.path.join(figures_dir, file))
        else:
            log(f"📊 Plotting {file}")
            fig = go.Figure(px.data.iris())
            fig.write_html(os.path.join(figures_dir, file))

    return "The analysis is complete. Please refer to the output directory for the results."



