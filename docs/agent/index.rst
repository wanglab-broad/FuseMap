.. _Agent:

FuseMap Agent
================================================================================

FuseMap Agent is the agentic AI system of the FuseMap paper — a conversational
interface to the spatial brain foundation model and the molCCF mouse brain atlas.
Ask questions in natural language; a **supervisor agent** decomposes your request
into subtasks, delegates them to three specialized agents communicating through a
shared memory (zero-shot ReAct paradigm), and aggregates their results:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Agent
     - What it does
   * - **SupervisorAgent**
     - Decomposes the user query into subtasks, delegates to the agents below,
       and maintains coherent context across steps.
   * - **AtlasAgent**
     - Queries the 3D mouse brain atlas (molCCF): matches brain regions and
       cell types, retrieves marker genes, performs visual section matching.
   * - **ResearchAgent**
     - Searches the literature (via Tavily) for diseases, conditions, and
       related studies to ground the analysis.
   * - **FuseMapAgent**
     - Executes FuseMap workflows on your data: integration, mapping,
       annotation transfer.

Two interfaces
--------------------------------------------------------------------------------

The repository ships two chat interfaces to the same foundation-model arsenal:

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * -
     - **Classic** (``app.py``) — the paper architecture
     - **v2** (``app_v2.py``) — modernized, recommended
   * - Architecture
     - Supervisor + research / atlas / FuseMap agents (zero-shot ReAct,
       shared memory)
     - Single CodeAct agent: writes Python against a **persistent kernel**
       pre-loaded with the FuseMap API, molCCF atlas, and imputation helpers
   * - Model
     - Chosen in the sidebar (GPT-4o-era)
     - **Auto-selects the newest flagship** for your API key (GPT-6 / latest
       Claude / Gemini; custom gateways via Base URL)
   * - Long sessions
     - Conversation buffer
     - Checkpointed threads + automatic context compaction; artifacts
       (figures/tables) saved to files and rendered inline
   * - Workflows
     - Fixed tools
     - Six validated skill playbooks (map+annotate, integrate, impute,
       subtype shifts, atlas query, bead deconvolution readout) loaded on
       demand; heavy GPU jobs queue with a wait notice
   * - Run
     - ``streamlit run app.py``
     - ``streamlit run app_v2.py`` (needs ``pip install "langchain>=1.0"
       langgraph langgraph-checkpoint-sqlite jupyter-client ipykernel``
       alongside fusemap)

Try it in Colab (zero install)
--------------------------------------------------------------------------------

The fastest way to try the agent — no GPU, nothing to install:
open the `Colab notebook <https://colab.research.google.com/github/wanglab-broad/FuseMap/blob/main/docs/notebooks/agent_colab.ipynb>`__,
paste your OpenAI API key (Tavily key optional, enables literature search), and chat.

.. toctree::
   :hidden:

   ../notebooks/agent_colab

Setup
--------------------------------------------------------------------------------

1. Clone the repository and install:

   .. code-block:: bash

       git clone https://github.com/wanglab-broad/FuseMap.git
       cd FuseMap
       conda create -n fusemap python=3.10.16
       conda activate fusemap
       pip install fusemap

2. Download the required data:

   .. list-table::
      :header-rows: 1
      :widths: 34 40 26

      * - Data
        - Link
        - Location
      * - Pretrained model weights
        - `Google Drive <https://drive.google.com/drive/u/2/folders/1auybpmekWuW_G-7YPloJr-B96qiT1nFS>`__
        - ``FuseMap/molCCF/``
      * - Atlas molCCF data
        - `Google Drive <https://drive.google.com/file/d/15LIkQTridS_ATwDy6dejIdzbMm39sEv3/view?usp=sharing>`__
        - ``FuseMap/agent_setup/atlas_data/``
      * - Example datasets (optional)
        - `Google Drive <https://drive.google.com/drive/folders/1ZRIbHTd9TAjmtr3V6WLkvrY4iLF5SH_U?usp=drive_link>`__
        - your choice
3. Launch the web interface:

   .. code-block:: bash

       streamlit run app.py

   Then open the ``localhost`` URL shown in the terminal.

API keys
--------------------------------------------------------------------------------

Enter these in the sidebar of the web interface:

- **OpenAI API key** (required) — powers the language model.
- **Tavily API key** (required for literature search) — free at
  `tavily.com <https://www.tavily.com/>`__.
- **Base URL** (optional) — custom OpenAI-compatible endpoint; leave blank
  for default.

Example prompts
--------------------------------------------------------------------------------

.. code-block:: text

    Which brain region is enriched for Pvalb+ interneurons?

.. code-block:: text

    I have a new MERFISH dataset of an Alzheimer's disease mouse model.
    Map it to the molCCF atlas and transfer cell type annotations.

.. code-block:: text

    Find recent literature on dentate gyrus vulnerability in aging,
    then check which molCCF regions express the reported marker genes.

The agent shows its intermediate reasoning steps (tool calls, atlas queries,
FuseMap runs) in the chat so you can verify each action.

.. seealso::

    Full step-by-step instructions with screenshots:
    `agent_setup/README.md <https://github.com/wanglab-broad/FuseMap/blob/main/agent_setup/README.md>`__
