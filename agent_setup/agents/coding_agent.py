### ---coding_agent.py--- ###
"""
CodingAgent — generates and executes arbitrary Python code for custom
data manipulation tasks that go beyond the predefined atlas tools.

Follows the same pattern as atlas_agent.py: creates tools, builds an
agent with create_model_agnostic_agent, wraps it as a LangChain Tool.
"""

from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory
from agent_setup.agent_utils import create_model_agnostic_agent
from agent_setup.knowledge import get_atlas_schema, format_schema_full
from agent_setup.progress_utils import log_progress
from agent_setup.tools.coding_tools import StatefulPythonREPL, create_execute_python_tool


def create_coding_tools(output_dir: str = "output"):
    """Instantiate the REPL and return (tools_list, repl_instance)."""
    repl = StatefulPythonREPL(output_dir=output_dir)
    execute_tool = create_execute_python_tool(repl)
    return [execute_tool], repl


def create_coding_agent(llm, output_dir: str = "output"):
    """Build a CodingAgent that can write and execute Python code."""
    # ConversationBufferWindowMemory: only the last k=5 turns are retained so
    # that token counts stay bounded across a long Streamlit session.
    memory = ConversationBufferWindowMemory(
        memory_key="memory", return_messages=True, k=5
    )

    coding_tools, repl = create_coding_tools(output_dir)

    try:
        schema_block = format_schema_full(get_atlas_schema(), output_dir)
    except Exception:
        schema_block = "\nAtlas data not available in namespace.\n"

    system_prompt = f"""You are an expert Python programmer for spatial transcriptomics analysis.
You write and execute Python code to perform data manipulation, extraction, computation,
and visualization tasks on the FuseMap mouse brain atlas (molCCF).

{schema_block}

=== PRE-IMPORTED LIBRARIES ===
numpy (np), pandas (pd), scanpy (sc), anndata (ad),
matplotlib.pyplot (plt), seaborn (sns), scipy (stats, sparse), os

=== CODING RULES ===
1. Gene names must be UPPERCASE: 'VIP' not 'Vip'
2. Save all output files to OUTPUT_DIR: os.path.join(OUTPUT_DIR, filename)
3. Use plt.savefig() instead of plt.show(); always call plt.close() after
4. Do NOT modify ad_cell or ad_gene in-place — always .copy() first
5. CRITICAL: ad_cell.X and ad_gene.X are 64-dim EMBEDDINGS, not expression.
   To get imputed expression: imputed = ad_cell.X @ ad_gene[gene_list].X.T
6. When creating AnnData output, include relevant metadata from ad_cell.obs
7. For large files (>10M cells), consider chunked processing or subsetting
8. Check if a gene exists before indexing: assert 'VIP' in ad_gene.obs.index
9. Use atlas color maps from agent_setup/atlas_data/colors/ for visualizations
10. INPUT FILE VALIDATION (when using files from a previous agent):
    Before reading any CSV/h5ad passed in the context, validate it first:
      df = pd.read_csv(path); assert len(df) > 0 and not df.isnull().all().all(), "Input file is empty or all-null"
    If the file is invalid, raise an explicit error:
      raise ValueError(f"[UPSTREAM_ERROR] Input file {{path}} is invalid. Cannot proceed.")
    Do NOT attempt to work around bad upstream data — report the error clearly.

=== SELF-CORRECTION ===
If your code fails, read the error message carefully, fix the issue, and execute again.
You have up to 8 execution attempts per query — use them wisely.
If the error starts with [UPSTREAM_ERROR], do NOT retry — report it immediately as:
  "[AGENT_ERROR] Upstream data from previous agent is invalid: <detail>"


=== OUTPUT & ARTIFACT VALIDATION ===
After successful execution, report:
1. What the code did
2. Output file paths (full paths), each prefixed with:
   [ARTIFACT_OK]   <path>  — if file exists and has data (df.shape or adata.shape)
   [ARTIFACT_ERROR] <path> — if file is empty or creation failed
3. A brief summary of the results (shape, columns, key statistics)

Before finishing, validate every output file:
  For CSV:  assert os.path.exists({{p}}) and pd.read_csv({{p}}).shape[0] > 0
  For PNG:  assert os.path.exists({{p}}) and os.path.getsize({{p}}) > 1000  # >1KB
  For h5ad: assert os.path.exists({{p}}) and ad.read_h5ad({{p}}).n_obs > 0
If validation fails, try to re-run the relevant code before reporting failure.
"""

    agent = create_model_agnostic_agent(
        llm=llm,
        tools=coding_tools,
        system_message=system_prompt,
        memory=memory,
        verbose=True,
        agent_name="CodingAgent",   # enables SubAgentTraceCallback
    )

    return agent, repl


def coding_agent_tool(llm, output_dir: str = "output"):
    """Wrap the CodingAgent as a LangChain Tool for the supervisor."""
    agent, repl = create_coding_agent(llm, output_dir=output_dir)

    def get_final_message(input_text):
        log_progress("💻 [CodingAgent] Starting code generation...")
        # Reset per-invocation counter so each user prompt gets a fresh budget
        repl.reset_invocation_count()
        # Release any large objects left in the REPL namespace from the previous
        # invocation (e.g. large DataFrames, AnnData objects) before starting
        # a new query so memory does not accumulate across turns.
        repl.cleanup_large_objects()

        try:
            result = agent.invoke({"input": input_text})
            output = result.get("output", "")
            preview = output[:120].replace("\n", " ")
            if len(output) > 120:
                preview += "…"
            log_progress(f"✅ [CodingAgent] Done — {preview}")
        except Exception as e:
            import traceback
            log_progress(f"❌ [CodingAgent] Failed: {str(e)}")
            return f"Error in CodingAgent: {str(e)}\n{traceback.format_exc()}"

        return output

    return Tool(
        name="CodingAgent",
        func=lambda input: "FINAL ANSWER: " + get_final_message(input),
        description="""Write and execute Python code for custom spatial transcriptomics analysis.

            Use this agent when the task CANNOT be handled by AtlasAgent's predefined tools.
            Prefer AtlasAgent over this agent when in doubt.

            Capabilities:
            1. Design targeted gene panels with module balancing
            2. Compute marker genes for cell types (t-test, Wilcoxon)
            3. Build custom visualisations: heatmaps, violin plots, spatial scatter plots
            4. Cell deconvolution (spot → cell type assignment)
            5. Spatial cell-cell interaction analysis
            6. Cross-region or cross-section comparative analysis
            7. Any multi-step Python analysis beyond predefined atlas tools

            Input should include:
            - The user's analysis request in detail
            - Output directory: "Save results to '<path>'"
            - AtlasAgent output (verbatim) if the analysis builds on atlas results
              (especially h5ad file paths, section IDs, gene expression data)

            The agent self-corrects on Python errors (up to 8 attempts).
            Only retry from Supervisor if CodingAgent explicitly gives up.

            Keep path of the output files complete and unchanged.
            Do not change output message of the coding agent.
        """,
    )
