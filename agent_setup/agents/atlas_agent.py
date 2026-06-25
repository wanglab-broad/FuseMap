### ---atlas_agent.py--- ###

from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory
import agent_setup.tools.brain_atlas_tools as brain_atlas_tools
import datetime
from agent_setup.tools.brain_atlas_tools import (
    match_brain_region_tool, match_brain_type_tool,
    top_expressed_genes_in_region,
    query_gene_expression_in_region, compute_gene_correlation,
    execute_atlas_query,
    extract_gene_expression_h5ad,
)
from agent_setup.agent_utils import create_model_agnostic_agent, extract_tool_descriptions
from agent_setup.knowledge import get_atlas_schema, format_schema_compact
from agent_setup.progress_utils import log_progress

# Global memory variable to record all outputs from each agent and tool
atlas_agent_memory = []

def add_to_atlas_memory(agent_name: str, tool_name: str, output: str):
    """Add an entry to the atlas agent memory"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "agent": agent_name,
        "tool": tool_name,
        "output": output
    }
    atlas_agent_memory.append(entry)
    return entry

def get_atlas_memory_summary() -> str:
    """Get a summary of all recorded atlas agent outputs"""
    if not atlas_agent_memory:
        return "No atlas agent outputs recorded yet."
    
    summary = "=== ATLAS AGENT MEMORY SUMMARY ===\n\n"
    for entry in atlas_agent_memory:
        summary += f"[{entry['timestamp']}] {entry['agent']} - {entry['tool']}:\n"
        summary += f"{entry['output'][:200]}...\n\n"
    return summary

def clear_atlas_memory():
    """Clear the atlas agent memory"""
    global atlas_agent_memory
    atlas_agent_memory = []



def create_atlas_tools(llm):
    # List of available tools
    atlas_tools = [
        match_brain_region_tool(llm),
        match_brain_type_tool(llm),
        brain_atlas_tools.plot_region_distribution,
        brain_atlas_tools.find_section_ids,
        # brain_atlas_tools.read_one_section,
        brain_atlas_tools.explain_cell_type,
        top_expressed_genes_in_region,
        query_gene_expression_in_region,
        compute_gene_correlation,
        execute_atlas_query,
        extract_gene_expression_h5ad,
    ]
    return atlas_tools


def create_atlas_agent(llm):
    # Setup memory
    # ConversationBufferWindowMemory: retain only the last k=5 turns to keep
    # token counts bounded during long Streamlit sessions.  AtlasAgent tasks
    # are typically short lookup chains, so 5 turns is more than sufficient.
    memory = ConversationBufferWindowMemory(
        memory_key="memory", return_messages=True, k=5
    )

    atlas_tools = create_atlas_tools(llm)

    tool_descriptions = extract_tool_descriptions(atlas_tools)

    # Build atlas schema block from centralized knowledge module (lazy-loaded and cached)
    atlas_schema_block = format_schema_compact(get_atlas_schema())

    # Define the system prompt
    atlas_agent_system_prompt = f"""You are an expert AI assistant for searching and doing basic analysis of spatial transcriptomics data in the spatial mouse brain atlas.
{atlas_schema_block}

    You have access to a set of tools that allow you to analyze mouse brain spatial transcriptomics data.
    {tool_descriptions}

    === OUTPUT DIRECTORY RULES (CRITICAL) ===
    All tools accept an output_dir parameter that controls where results are saved.
    STEP 1 - Extract the output directory from the input message:
      - Look for phrases like "save results at", "output path", "save to", "save at"
      - If found, use that path as output_dir for ALL tool calls
      - If not found, use the default "output"

    STEP 2 - Pass output_dir consistently to EVERY tool call:
      - plot_region_distribution(..., output_dir=output_dir)
      - find_section_ids(..., output_dir=output_dir)
      - read_one_section(..., output_dir=output_dir)

    EXAMPLE:
      Input contains: "Save results at '/path/to/output/my_run/'"
      → output_dir = '/path/to/output/my_run/'
      → Pass this to ALL tool calls throughout the entire workflow.
    ========================================================

    MANDATORY WORKFLOW — execute ALL four steps every time, even when called as part of a larger pipeline:
    1. Identify gene names, cell types, and brain regions mentioned in the input.
    2. Match cell types and brain regions using `match_brain_region` and `match_brain_type` to get atlas codes.
    3. ALWAYS call `plot_region_distribution` with the matched region codes to generate 3D HTML visualisations.
       Do NOT skip this step even if your primary task is to supply section IDs to a downstream agent.
    4. Call `find_section_ids` with the matched codes to find relevant 2D tissue sections.
    If any step fails, retry once before moving on.
    You MUST complete all four steps. Only skip a step if the user explicitly asks you not to perform it.

    === EXTRACTING CELL TYPES AND REGIONS FROM CONTEXT ===
    When called as part of a pipeline, your input will include ResearchAgent output.
    You MUST extract cell types and brain regions from that output BEFORE considering
    whether to ask the user for clarification.

    Priority order for identifying cell types and brain regions:
    1. Explicit values in the user's own message (highest priority)
    2. Cell types / regions mentioned in the ResearchAgent output (e.g. "microglia",
       "astrocytes", "hippocampus CA1") — USE THESE directly as input to `match_brain_type`
       and `match_brain_region`. Do NOT ask the user if ResearchAgent already provided them.
    3. Only ask the user for clarification if NEITHER the user message NOR any prior
       agent output contains any cell type or brain region information.

    When the user specifically requires something like visualizing a 2D section, you can perform the corresponding tool.
    Before you finish, if there is cell type symbols from the spatial brain atlas in the input or output, you should explain the symbols with detailed information.
    You should also Match cell types, and match brain regions and then match section ids in the mouse brain atlas for the other species based on the description of the query dataset.
    Keep path of the output files complete and unchanged.
    Keep all output information unchanged when you return the final answer.


    === ARTIFACT VALIDATION (MANDATORY before FINAL ANSWER) ===
    After every tool call that saves a file, you MUST validate the artifact:

    For CSV files:
      import os, pandas as pd
      df = pd.read_csv('<path>')
      print(f"Rows: {{len(df)}}, Cols: {{list(df.columns[:5])}}, Nulls: {{df.isnull().sum().sum()}}")
      If rows == 0 OR all values are null/NaN → the tool FAILED.
        • Do NOT report this file as a success.
        • Report [ARTIFACT_ERROR]: <path> is empty or all-null. Re-run the tool.

    For h5ad files:
      import os, anndata as ad
      adata = ad.read_h5ad('<path>')
      print(f"Shape: {{adata.shape}}, obs keys: {{list(adata.obs.columns[:5])}}")
      If adata.n_obs == 0 → [ARTIFACT_ERROR]: h5ad is empty.

    Only after successful validation, include the file path in FINAL ANSWER.
    If validation fails and retries also fail, report [ARTIFACT_ERROR] explicitly
    so the downstream agent knows NOT to use this file.
    ===============================================================

    Always return a 'FINAL ANSWER' with a complete summary of all actions taken.
    Prefix successful file artifacts with [ARTIFACT_OK] and failed ones with [ARTIFACT_ERROR].
    """


    # Use model-agnostic agent creation (supports both OpenAI and Anthropic)
    atlas_agent = create_model_agnostic_agent(
        llm=llm,
        tools=atlas_tools,
        system_message=atlas_agent_system_prompt,
        memory=memory,
        verbose=True,
        agent_name="AtlasAgent",   # enables SubAgentTraceCallback
    )

    return atlas_agent


def _validate_artifacts_in_output(output_text: str) -> list[str]:
    """
    Layer 2 artifact guardrail.

    Scans the agent's text output for file paths ending in .csv or .h5ad,
    then validates each one:
      - file must exist on disk
      - CSV files must have at least one row and not be entirely null
      - h5ad files must have at least one observation

    Returns a list of human-readable error strings (empty list = all OK).
    """
    import re
    import os

    errors = []
    # Match absolute paths and relative paths ending in .csv or .h5ad
    path_pattern = re.compile(r"([/\w\-\.]+\.(?:csv|h5ad))")
    paths = path_pattern.findall(output_text)

    for path in set(paths):   # deduplicate
        if not os.path.exists(path):
            errors.append(f"{path}: file does not exist")
            continue
        if path.endswith(".csv"):
            try:
                import pandas as pd
                df = pd.read_csv(path)
                if len(df) == 0:
                    errors.append(f"{path}: CSV has 0 rows (empty)")
                elif df.isnull().all().all():
                    errors.append(f"{path}: CSV has {len(df)} rows but ALL values are null")
            except Exception as e:
                errors.append(f"{path}: CSV could not be read — {e}")
        elif path.endswith(".h5ad"):
            try:
                import anndata as ad
                adata = ad.read_h5ad(path)
                if adata.n_obs == 0:
                    errors.append(f"{path}: h5ad has 0 observations (empty)")
            except Exception as e:
                errors.append(f"{path}: h5ad could not be read — {e}")

    return errors


def atlas_agent_tool(llm):
    atlas_agent = create_atlas_agent(llm)

    def get_final_message_from_stream(input_text):
        # Avoid any Streamlit UI calls here: this function runs inside a
        # ThreadPoolExecutor background thread where there is no Streamlit
        # session context. Calling st.* from threads corrupts the delta queue
        # and causes the frontend to show a white/blank page.
        log_progress(f"🧠 [AtlasAgent] Starting analysis...")
        add_to_atlas_memory("AtlasAgent", "Input", input_text)

        try:
            agent_result = atlas_agent.invoke({"input": input_text})
            result = agent_result.get("output", "")
            add_to_atlas_memory("AtlasAgent", "FinalResult", result)

            # ── Layer 2: Automatic artifact validation ────────────────────────
            # Scan the output text for file paths. For each CSV/h5ad file
            # found, verify it exists and is non-empty. Prepend a
            # [UPSTREAM_ERROR] block if a broken artifact is detected so that
            # the Supervisor and downstream agents can detect it mechanically
            # without relying on the LLM to self-report failures.
            validation_errors = _validate_artifacts_in_output(result)
            if validation_errors:
                error_block = "[UPSTREAM_ERROR] AtlasAgent produced invalid artifacts:\n"
                error_block += "\n".join(f"  - {e}" for e in validation_errors)
                error_block += "\nThe downstream agent should NOT use these files.\n\n"
                result = error_block + result
                log_progress(f"⚠️ [AtlasAgent] Artifact validation failed: {validation_errors[0]}")
            # ─────────────────────────────────────────────────────────────────

            preview = result[:120].replace("\n", " ")
            if len(result) > 120:
                preview += "…"
            log_progress(f"✅ [AtlasAgent] Done — {preview}")
        except Exception as e:
            import traceback
            error_msg = f"Error in AtlasAgent: {str(e)}\n{traceback.format_exc()}"
            log_progress(f"❌ [AtlasAgent] Failed: {str(e)}")
            add_to_atlas_memory("AtlasAgent", "Error", error_msg)
            return "FINAL ANSWER: Error occurred."

        return result
    
    return Tool(
        name="AtlasAgent",
        func=lambda input: "FINAL ANSWER: " + get_final_message_from_stream(input),
        description="""Query, visualise, and analyse the mouse spatial brain atlas (molCCF).

            Capabilities:
            1. Match brain region names to atlas abbreviations (e.g. 'hippocampus' → 'HPF')
            2. Match cell type names to atlas symbols (e.g. 'microglia' → 'MGL')
            3. Plot 3D spatial distribution of brain regions, cell types, or genes
            4. Find 2D tissue section IDs for given cell types and brain regions
            5. Query gene expression statistics and gene-gene correlations in atlas regions
            6. Extract imputed gene expression across atlas cells, save as h5ad file
            7. Explain cell type symbols with detailed biological information

            Input should include:
            - The user's question or analysis request
            - Output directory: "Save results to '<path>'"
            - Research context (verbatim ResearchAgent output, if available)

            Prefer this agent over CodingAgent for any task the predefined tools can handle.
            If this agent returns 'no matching sections found', the Supervisor should retry
            with broader region or cell-type terms.

            Do not change output message of this agent.
        """
    )