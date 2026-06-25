# FuseMap

**FuseMap** is a deep-learning framework for spatial transcriptomics that integrates spatial gene expression data with universal gene, cell, and tissue embeddings. It creates a unified mouse brain atlas (molCCF) by bridging single-cell/single-spot data with spatial contexts.

On top of the core ML pipeline, FuseMap ships a **multi-agent AI interface** — a natural-language assistant powered by LangGraph and LangChain that lets you query the 3D mouse brain atlas, search the literature, run custom Python analysis, and execute the full FuseMap pipeline, all by chat.

---

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Usage — CLI](#usage--cli)
- [Usage — Web UI (Agent)](#usage--web-ui-agent)
- [Data Format](#data-format)
- [HPC / Resource Requirements](#hpc--resource-requirements)
- [Documentation](#documentation)

---

## Architecture

```
FuseMap-revision/
├── fusemap/                    # Core ML package (PyTorch + DGL)
│   ├── model.py                # VAE + Discriminator + GNN architectures
│   ├── train_model.py          # Adversarial training orchestration
│   ├── spatial_integrate.py    # Multi-sample integration pipeline
│   ├── spatial_map.py          # Reference mapping pipeline
│   ├── loss.py                 # Reconstruction / adversarial / alignment losses
│   ├── dataset.py              # DGL graph dataset construction
│   ├── preprocess.py           # Data normalisation + spatial graph building
│   └── config.py               # ModelType enum + CLI argument parsing
│
├── agent_setup/                # Multi-agent system (LangChain + LangGraph)
│   ├── supervisor_graph.py     # ReAct Supervisor agent + routing
│   ├── agent_utils.py          # AgentWrapper + create_model_agnostic_agent
│   ├── progress_utils.py       # Thread-safe progress queue for Streamlit
│   ├── config.py               # LLM factory (create_llm, SUPPORTED_MODELS)
│   ├── knowledge.py            # Atlas schema lazy-loader + cache
│   ├── agents/
│   │   ├── atlas_agent.py      # 3D mouse brain atlas queries (10 tools)
│   │   ├── research_agent.py   # Literature search via Tavily API
│   │   ├── coding_agent.py     # StatefulPythonREPL code execution
│   │   └── fusemap_agent.py    # FuseMap analysis pipeline
│   └── tools/
│       ├── brain_atlas_tools.py
│       ├── coding_tools.py
│       └── fusemap_tool.py
│
├── app.py                      # Streamlit web UI entry point
├── main.py                     # CLI entry point for model training
├── scripts/                    # Test and utility scripts
└── docs/                       # Documentation
```

---

## Installation

### Requirements

| Dependency | Version |
| ---------- | ------- |
| Python     | ≥ 3.10 |
| PyTorch    | 2.0.1   |
| DGL        | 1.1.1   |
| scanpy     | 1.9.3   |
| LangChain  | 0.3.25  |
| LangGraph  | 0.4.7   |
| Streamlit  | 1.45.1  |

### Setup with uv (recommended)

```bash
git clone https://github.com/your-org/FuseMap-revision.git
cd FuseMap-revision
uv sync
```

### Setup with pip

```bash
pip install -e .
```

---

## Usage — CLI

Use `main.py` to run the FuseMap pipeline from the command line.

### Integration mode (multi-sample)

Learns a joint embedding across all samples without a reference atlas.

```bash
python main.py \
  --input_data_folder_path /path/to/h5ad/files \
  --output_save_dir /output/dir \
  --mode integrate
```

### Map mode (reference mapping)

Maps new data onto the pre-trained molCCF reference atlas.

```bash
python main.py \
  --input_data_folder_path /path/to/h5ad/files \
  --output_save_dir /output/dir \
  --mode map
```

### Key model parameters (`fusemap/config.py`)

| Parameter         | Default | Description              |
| ----------------- | ------- | ------------------------ |
| `pca_dim`       | 50      | PCA dimension before GNN |
| `hidden_dim`    | 512     | GNN hidden dimension     |
| `latent_dim`    | 64      | Latent embedding size    |
| `n_epochs`      | 16      | Training epochs          |
| `batch_size`    | 64      | Batch size               |
| `learning_rate` | 0.001   | RMSprop learning rate    |

---

## Usage — Web UI (Agent)

FuseMap includes a multi-agent AI interface built on Streamlit. The **Supervisor Agent** orchestrates four specialized sub-agents:

| Agent                   | Role                                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **AtlasAgent**    | Queries the 3D mouse brain atlas — cell types, brain regions, gene expression, section IDs |
| **ResearchAgent** | Literature search via Tavily API                                                            |
| **CodingAgent**   | Runs arbitrary Python in a stateful REPL with atlas data pre-loaded                         |
| **FuseMapAgent**  | Executes the full FuseMap spatial analysis pipeline                                         |

### Required data files

| File                     | Location                                              | Size    |
| ------------------------ | ----------------------------------------------------- | ------- |
| Pretrained model weights | `molCCF/trained_model/FuseMap_final_model_final.pt` | ~4.8 GB |
| Transfer weights         | `molCCF/transfer/`                                  | ~200 KB |
| Atlas cell data          | `agent_setup/atlas_data/ad_cell.h5ad`               | ~15 GB  |
| Atlas gene data          | `agent_setup/atlas_data/ad_gene.h5ad`               | ~15 MB  |

### Required API keys

| Key                                  | Purpose                                               |
| ------------------------------------ | ----------------------------------------------------- |
| OpenAI**or** Anthropic API key | LLM backend                                           |
| Tavily API key                       | Literature search (free tier available at tavily.com) |

### Supported models

- `gpt-4o` — OpenAI GPT-4o
- `claude-sonnet-4` — Anthropic Claude Sonnet 4

### Starting the web UI

**Local machine:**

```bash
streamlit run app.py
# → open http://localhost:8501
```

**Remote server (e.g., HPC):**

```bash
# On the server
streamlit run app.py --server.port 8501 --server.headless true

# On your local machine — SSH port forwarding
ssh -L 8501:localhost:8501 <user>@<hpc-login-node>
# → open http://localhost:8501
```

### Example queries

**Atlas query:**

```
Which brain sections have the highest concentration of microglia (MGL_1)
in the cortex? Show a distribution plot.
```

**Literature + atlas pipeline:**

```
I want to study Alzheimer's disease in the mouse brain.
First, tell me which cell types and genes are most affected in AD.
Then find those cell types and their spatial distribution in the mouse brain atlas.
```

**Full analysis (FuseMap):**

```
I have spatial transcriptomics data at:
/path/to/example_data/application_data/disease
Help me annotate the cell types and save results to ./output/
```

**Custom Python analysis:**

```
Find the top 20 marker genes for microglia using a t-test approach.
Save the result as a CSV file.
```

See [`docs/demos/agent_test_prompts.md`](docs/demos/agent_test_prompts.md) for a complete set of 16 demo prompts that cover all routing scenarios.

---

## Data Format

- **Input:** H5AD files (AnnData format)
- Spatial coordinates in `obs['x']` / `obs['y']` or `obsm['spatial']`
- Each section must have a unique identifier in `obs['name']`

### Output files (FuseMap analysis)

```
output/
├── data/
│   ├── annotated_user_data.h5ad       # Final annotated AnnData
│   ├── main_level_celltype.csv        # Marker genes per leiden cluster
│   └── subtype_{CellType}.csv         # Marker genes per subtype cluster
├── figures/
│   ├── user_data_main_level_celltype.png
│   ├── user_data_spatial_*.png
│   └── user_data_sub_{CellType}*.png
└── fusemap/
    ├── disease/molCCF_mapping/        # molCCF mapping embeddings
    └── disease/integrate/             # FuseMap integrated embeddings
```

---

## HPC / Resource Requirements

The full agent workflow (including `map_molCCF`) requires significant memory:

| Resource | Minimum | Recommended |
| -------- | ------- | ----------- |
| RAM      | 64 GB   | 128 GB      |
| GPU      | 1×     | 1×         |
| CPUs     | 8       | 16          |

**SLURM example:**

```bash
srun --partition=gpu --gres=gpu:1 --mem=128G --cpus-per-task=16 --pty bash
```

**Reduce thread contention before starting the UI:**

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

See [`docs/HPC_RESOURCES.md`](docs/HPC_RESOURCES.md) for details on the OOM root cause and data file sizes.

---

## Documentation

| File                                                                                    | Description                                                       |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`docs/agent_usage.md`](docs/agent_usage.md)                                             | Full guide to launching and using the agent UI                    |
| [`docs/demos/agent_test_prompts.md`](docs/demos/agent_test_prompts.md)                   | 16 demo prompts covering all agent routing scenarios              |
| [`docs/src/agent_system_implementation.md`](docs/src/agent_system_implementation.md)     | Architecture deep-dive: routing, sub-agents, error propagation    |
| [`docs/src/security_permission_hardening.md`](docs/src/security_permission_hardening.md) | REPL sandbox: blocklist, AST path scan, write guards              |
| [`docs/src/resource_release_management.md`](docs/src/resource_release_management.md)     | Memory management: REPL cleanup, AnnData release, windowed memory |
| [`docs/src/streaming_agent_trace.md`](docs/src/streaming_agent_trace.md)                 | Real-time agent trace timeline in the Streamlit UI                |
| [`docs/HPC_RESOURCES.md`](docs/HPC_RESOURCES.md)                                         | HPC resource requirements and SLURM configuration                 |
