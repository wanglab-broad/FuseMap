# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FuseMap is a deep learning framework for spatial transcriptomics that integrates spatial gene expression data with universal gene, cell, and tissue embeddings. It creates a unified mouse brain atlas (molCCF) by bridging single-cell/single-spot data with spatial contexts.

## Repository Structure

- **fusemap/** - Core ML package: model architecture, training, loss functions, preprocessing
- **agent_setup/** - Multi-agent system: LangChain/LangGraph-based orchestration with specialized agents
- **app.py** - Streamlit web UI entry point
- **main.py** - CLI entry point for model training
- **scripts/** - Test and utility scripts for the agent system
- **docs/** - Documentation (Sphinx + markdown guides)

## Build and Installation

```bash
# Install FuseMap (from repo root)
pip install -e .

# Or install dependencies manually
pip install -r requirements.txt
```

## Running the Application

### CLI Mode
```bash
# Integration mode (multi-sample integration)
python main.py --input_data_folder_path /path/to/h5ad/files --output_save_dir /output/dir --mode integrate

# Map mode (reference mapping)
python main.py --input_data_folder_path /path/to/h5ad/files --output_save_dir /output/dir --mode map
```

### Web UI (Multi-Agent System)
```bash
streamlit run app.py
```
Requires OpenAI API key and Tavily API key for web search functionality.

## Architecture

### Core FuseMap Pipeline (`fusemap/`)
- `spatial_integrate.py` - Main integration pipeline for multi-sample data
- `spatial_map.py` - Reference mapping pipeline for new data
- `model.py` - Neural network architectures (VAE, Discriminator, GNN components)
- `train_model.py` - Training orchestration with adversarial learning
- `loss.py` - Loss functions for reconstruction, adversarial, and alignment objectives
- `dataset.py` - DGL-based graph dataset construction
- `preprocess.py` - Data normalization and spatial graph construction
- `config.py` - Model hyperparameters (`ModelType` enum) and CLI argument parsing

### Multi-Agent System (`agent_setup/`)
Uses LangChain/LangGraph for orchestration with four specialized agents:

- **AtlasAgent** (`agents/atlas_agent.py`) - Queries the 3D mouse brain atlas, matches brain regions/cell types
- **ResearchAgent** (`agents/research_agent.py`) - Literature search using Tavily API
- **CodingAgent** (`agents/coding_agent.py`) - Python code execution for custom analysis
- **FuseMapAgent** (`agents/fusemap_agent.py`) - Executes FuseMap analysis workflows

Agent tools are defined in `tools/` subdirectory. LLM configuration is in `agent_setup/config.py`.

### Data Format
- Input: H5AD files (AnnData format)
- Spatial coordinates expected in `obs['x']` and `obs['y']` columns (or `obsm['spatial']`)
- Each section gets a unique name in `obs['name']`

### Key Model Parameters (`fusemap/config.py`)
- `pca_dim=50`, `hidden_dim=512`, `latent_dim=64`
- `n_epochs=16`, `batch_size=64`, `learning_rate=0.001`
- Optimizer: RMSprop

## Key Dependencies
- PyTorch 2.0.1, DGL 1.1.1 (Graph Neural Networks)
- scanpy 1.9.3 (single-cell analysis)
- LangChain 0.3.25, LangGraph 0.4.7 (LLM orchestration)
- Streamlit 1.45.1 (web UI)
