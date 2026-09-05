.. _Agent:

FuseMap Agent
================================================================================

FuseMap Agent is a conversational AI interface to FuseMap and the molCCF mouse
brain atlas. Ask questions in natural language — the system orchestrates three
specialized agents to answer them:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Agent
     - What it does
   * - **AtlasAgent**
     - Queries the 3D mouse brain atlas (molCCF): matches brain regions and
       cell types, retrieves marker genes, performs visual section matching.
   * - **ResearchAgent**
     - Searches the literature (via Tavily) for diseases, conditions, and
       related studies to ground the analysis.
   * - **FuseMapAgent**
     - Executes FuseMap workflows on your data: integration, mapping,
       annotation transfer.

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
