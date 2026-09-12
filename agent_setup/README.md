# FuseMap-Agent: Multi-Agent AI Interface Tutorial

> **Note**: For the core FuseMap package tutorial, see: https://fusemap.readthedocs.io/en/latest/

## Part A: Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/wanglab-broad/FuseMap.git
cd FuseMap
```

### Step 2: Download required data

| Data | Link | Location |
|------|------|----------|
| Pretrained model weights | [Google Drive](https://drive.google.com/drive/u/2/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS) | `FuseMap/molCCF/` |
| Atlas molCCF data | [Google Drive](https://drive.google.com/file/d/15LIkQTridS_ATwDy6dejIdzbMm39sEv3/view?usp=sharing) | `FuseMap/agent_setup/atlas_data/` |
| Example datasets (optional) | [Google Drive](https://drive.google.com/drive/folders/1ZRIbHTd9TAjmtr3V6WLkvrY4iLF5SH_U?usp=drive_link) | Your choice |

### Step 3: Set up environment and run
```bash
conda create -n fusemap python=3.10.16
conda activate fusemap
python -m pip install ".[agent]"
streamlit run app.py
```

### Step 4: Open in browser
Navigate to `http://localhost:xxxx` (port shown in terminal)

---

## Part B: Using the Interface

### Required API keys:
1. **OpenAI API key**: Required for the language model
2. **Tavily API key**: Free, get one at https://www.tavily.com/
3. **Base URL** (optional): Leave blank for default

### Example query:
```
How do the cell state changes in mouse hippocampus with Alzheimer's disease? 
I have measured spatial transcriptomics datasets of mouse brain with two 
Alzheimer's disease model at path '/path/to/data'. Save results at 'path/to/output'. 
Help me analyze the cell types.
```

### Outputs:
- All results saved to `path/to/output/`
- Final annotated data: `path/to/output/data/annotated_user_data.h5ad`
- To view HTML figures on remote SSH: `python3 -m http.server 8000`

---

## Run in Colab

The classic Agent notebook can be opened in Colab with a Python 3.9–3.11 runtime.
Atlas files must be downloaded first and require sufficient RAM; integration and
mapping require a suitable compute environment:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wanglab-broad/FuseMap/blob/main/docs/notebooks/agent_colab.ipynb)

The notebook clones this repository, installs dependencies, prompts for your OpenAI API key (and an optional Tavily key for the literature ResearchAgent), then starts an interactive chat loop with the supervisor agent.

## Agent v2

`app.py` is the paper's Supervisor + three-agent architecture (LangChain 0.3).
`app_v2.py` is a later single CodeAct agent (LangChain 1.x) that drives a persistent
FuseMap Python kernel. Use two environments; installing LangChain 1.x alongside
the classic pinned dependencies produces conflicts.

From the repository root, register the scientific kernel:

```bash
conda create -n fusemap python=3.10
conda activate fusemap
python -m pip install . ipykernel
python -m ipykernel install --user --name fusemap-kernel --display-name "FuseMap"
```

Then install and launch the controller in its own environment:

```bash
conda create -n fusemap-agent-v2 python=3.11
conda activate fusemap-agent-v2
python -m pip install -r requirements-agent-v2.txt
jupyter kernelspec list
streamlit run app_v2.py
```

Do not install `requirements.txt` or `fusemap` into the v2 controller environment.
Both environments must be accessible on the same machine under the same user.
The kernel name defaults to `fusemap-kernel`; set `FUSEMAP_KERNEL_NAME` to select
another registered scientific kernel. No developer-specific Python path is needed.
Download the pretrained weights to `molCCF/` and the atlas files `ad_cell.h5ad`
and `ad_gene.h5ad` to `agent_setup/atlas_data/` for atlas workflows. Reference data
are not bundled in the wheel. Large atlas files may exceed Colab memory.

The v2 controller supports OpenAI, Anthropic and Google provider adapters.
Provider credentials are entered in the interface. Tavily is optional for
literature search. Launching a kernel and making an LLM call are separate checks;
successful installation alone does not validate an end-to-end scientific analysis.
