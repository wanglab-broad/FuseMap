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
pip install fusemap
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
