### ---fusemap_agent.py--- ###
from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory
import agent_setup.tools.fusemap_tool as fusemap_tool
import inspect
from agent_setup.tools.fusemap_tool import finalize_mainlevel_tool, annotate_sublevel_tool
from agent_setup.agent_utils import create_model_agnostic_agent
from agent_setup.progress_utils import log_progress


def extract_tool_descriptions(tools):
    """Extract descriptions from tool functions automatically"""
    descriptions = []
    
    for i, tool in enumerate(tools, 1):
        # Determine if it's a function-based or class-based tool
        if hasattr(tool, "func"):
            # Function-based tool (e.g. LangChain's Tool)
            func = tool.func
            name = getattr(tool, "name", func.__name__)
            doc = func.__doc__ or "No description available"
        else:
            # Class-based tool (custom tools)
            func = getattr(tool, "_run", None)
            name = getattr(tool, "name", tool.__class__.__name__)
            doc = getattr(tool, "description", None) or (func.__doc__ if func else "No description available")

        # Extract parameters from the function signature
        if func:
            sig = inspect.signature(func)
            params = [p for p in sig.parameters if p != "log"]
        else:
            params = []

        param_str = ", ".join(params)
        description = f"{i}. {name}({param_str}): {doc.strip()}"
        descriptions.append(description)
    
    return "\n".join(descriptions)


# def extract_tool_descriptions(tools):
#     """Extract descriptions from tool functions automatically"""
#     descriptions = []
    
#     for i, tool_func in enumerate(tools, 1):
#         # Get the function signature
#         sig = inspect.signature(tool_func.func)
#         params = list(sig.parameters.keys())
        
#         # Remove the 'log' parameter as it's internal
#         if 'log' in params:
#             params.remove('log')
        
#         # Get the docstring
#         doc = tool_func.func.__doc__ or "No description available"
        
#         # Format the description
#         param_str = ", ".join(params)
#         description = f"{i}. {tool_func.name}({param_str}): {doc.strip()}"
#         descriptions.append(description)
    
#     return "\n".join(descriptions)


def create_fusemap_tools(llm):
    """Create fusemap tools with the provided LLM"""
    # Create tools that accept the LLM parameter
    tools = [
        fusemap_tool.map_molCCF,
        fusemap_tool.fusemap_integrate,
        # Create wrapper functions for tools that need LLM
        finalize_mainlevel_tool(llm),
        annotate_sublevel_tool(llm),
    ]
    return tools



def create_fusemap_agent(llm):
    """Create and return a fusemap agent with the provided LLM"""
    # ConversationBufferWindowMemory: keep only the last k=5 turns so that
    # a long pipeline session does not exhaust the context window.  The
    # Supervisor passes the full task description in each invocation, so
    # recent history (not the entire session) is sufficient for continuity.
    memory = ConversationBufferWindowMemory(
        memory_key="memory", return_messages=True, k=5
    )

    # Create tools with LLM
    tools = create_fusemap_tools(llm)

    # Automatically extract tool descriptions
    tool_descriptions = extract_tool_descriptions(tools)

    # Define the system prompt
    fusemap_agent_system_prompt = f"""You are an expert AI assistant for using FuseMap model, a deep-learning-based framework for spatial transcriptomics that bridges single-cell or single-spot gene expression with spatial contexts and consolidates various gene panels across spatial transcriptomics atlases.
                                as well as integrating query spatial transcriptomics data with the reference brain atlas and analyze.

    You have access to the following tools:
    {tool_descriptions}

    === PATH HANDLING RULES (CRITICAL - READ CAREFULLY) ===
    The `path` argument MUST be a DIRECTORY that DIRECTLY CONTAINS the .h5ad
    file(s) — the tools call os.listdir(path) and read every *.h5ad at the top
    level. It must NOT be the path to a .h5ad file itself, and must NOT be a
    parent directory whose .h5ad files live in sub-folders.

    Resolve the path the user ACTUALLY gave. DO NOT invent, append, or rename
    folders the user did not mention:
      1. User gives a path to a specific .h5ad FILE
         (e.g. '/ewsc/.../retreat_ad/ad_starmap.h5ad'):
         → pass its PARENT DIRECTORY ('/ewsc/.../retreat_ad/').
      2. User gives a DIRECTORY that already contains .h5ad files directly:
         → use it AS-IS.
      3. ONLY for the bundled tutorial fixture whose path ends in 'example_data'
         (that folder has NO .h5ad files directly inside — it only has an
         'application_data/' sub-folder) map the dataset type to a subdirectory:
            Alzheimer's / disease model → '<that_path>/application_data/disease/'
            embryo / development        → '<that_path>/application_data/embryo/'
            human brain / cross-species → '<that_path>/application_data/species/'

    The disease/embryo/species keywords only decide WHICH analysis to run; they
    do NOT change the path unless rule 3 (the 'example_data' tutorial fixture)
    applies. NEVER append 'application_data/...' onto a user path that already
    points to, or into, a real data folder.

    IMPORTANT: The user must provide a REAL path that exists on the filesystem.
    If the resolved path does not exist, do NOT guess another path — report the
    exact path you tried and ask the user to confirm the correct location.

    EXAMPLES:
      User: "analyze my Alzheimer's data at '/data/retreat_ad/ad_starmap.h5ad'"
      → path = '/data/retreat_ad/'        (parent dir of the .h5ad file — rule 1)

      User: "my data is at '/data/my_experiment/'" and it directly contains .h5ad files
      → path = '/data/my_experiment/'     (rule 2)

      User: "my data is at 'example_data'" and mentions Alzheimer's disease
      → path = 'example_data/application_data/disease/'   (rule 3 — tutorial fixture only)

    === OUTPUT DIRECTORY RULES ===
    All tools accept an output_dir parameter that controls where results are saved.
    STEP 1 - Extract the output directory from the user's message:
      - Look for phrases like "save results at", "output path", "save to", "save at"
      - If found, use that path as output_dir for ALL tool calls
      - If not found, use the default "output"

    STEP 2 - Pass output_dir consistently to EVERY tool call:
      - map_molCCF(..., output_dir=output_dir)
      - fusemap_integrate(..., output_dir=output_dir)
      - finalize_mainlevel(..., output_dir=output_dir)
      - annotate_sublevel(..., output_dir=output_dir)

    EXAMPLE:
      User says: "Save results at 'output/my_run'"
      → output_dir = 'output/my_run'
      → Pass this to ALL tool calls throughout the entire workflow.
    ========================================================

    === DATA PATH RULES (data_path parameter) ===
    The data_path parameter in finalize_mainlevel and annotate_sublevel must point to
    the directory that DIRECTLY CONTAINS .h5ad files (the original input data directory),
    NOT a parent directory and NOT the output directory.

    CORRECT:   data_path = '/data/retreat_ad/'   ← the directory that directly contains .h5ad files
    INCORRECT: data_path = '/data/'              ← parent directory (.h5ad files are in sub-folders)
    INCORRECT: data_path = output_dir            ← output directory

    Use the SAME resolved data path as passed to map_molCCF and fusemap_integrate.
    ========================================================

    === map_molCCF ERROR HANDLING ===
    If map_molCCF returns an error message starting with "Error:", it means mapping failed.
    In that case, pass map_path=None to finalize_mainlevel and annotate_sublevel,
    and proceed using only the integrate results (integrate_path).
    ========================================================

    When a user asks to analyze spatial transcriptomics data, you should decide which tools to use.
    If the new sample has all cell types included in normal adult mouse brain,
                                like a part of the mouse brain or aging mouse brain,
                                only use molCCF mapping to transfer both main-level and sub-level cell types.

    If the new sample has partial cell types included in normal adult mouse brain,
                                like diseased mouse brain,
                                first use molCCF mapping to transfer only main-level,
                                then based on the results, apply spatial_integrate and annotate_new_sample.

    If the new sample has totally different cell types from normal adult mouse brain,
                                like another organ liver, mouse embryo with multiple organs,
                                apply spatial_integrate and annotate_new_sample.

    After this, finalize main-level cell types.
    Then annotate sublevel cell types.

    Finally, you can call plotting functions to show cell type composition changes across conditions.
    Always plot the annotation results after the analysis.

    If the user specifically request to run a tool, you should just call the tool with correct input.

    The final answer should be in the following format:
    'FINAL ANSWER'
    first take action 1, get result 1, detailed output is ...
    then take action 2, get result 2, detailed output is ...
    then take action 3, get result 3, detailed output is ...
    """

    # Use model-agnostic agent creation (supports both OpenAI and Anthropic)
    fusemap_agent = create_model_agnostic_agent(
        llm=llm,
        tools=tools,
        system_message=fusemap_agent_system_prompt,
        memory=memory,
        verbose=True,
        agent_name="FuseMapAgent",   # enables SubAgentTraceCallback
    )

    return fusemap_agent


def fusemap_agent_tool(llm):
    """Create a fusemap agent tool with the provided LLM"""
    agent = create_fusemap_agent(llm)
    
    def get_final_message_from_stream(input_text):
        # Tools run in LangGraph's ThreadPoolExecutor — no st.* calls allowed here.
        # Use log_progress() which writes to a thread-safe Queue read by the main thread.
        log_progress(f"🧬 [FuseMapAgent] Starting analysis...")

        try:
            agent_result = agent.invoke({"input": input_text})
            result = agent_result.get("output", "")
            preview = result[:120].replace("\n", " ")
            if len(result) > 120:
                preview += "…"
            log_progress(f"✅ [FuseMapAgent] Done — {preview}")
            return result

        except Exception as e:
            import traceback
            error_msg = f"Error in FuseMapAgent: {str(e)}\n{traceback.format_exc()}"
            log_progress(f"❌ [FuseMapAgent] Failed: {str(e)}")
            return f"FINAL ANSWER: An error occurred during FuseMap analysis: {str(e)}"

    return Tool(
        name="FuseMapAgent",
        func=lambda input: get_final_message_from_stream(input),
        description="""Run the FuseMap spatial transcriptomics integration pipeline on user-provided data.

        Use this agent ONLY when the user provides their OWN data files (h5ad, CSV, etc.).
        Do NOT use for reference atlas queries — use AtlasAgent for those.

        Capabilities:
        1. Map user's spatial data to the mouse brain atlas (molCCF) via map_molCCF
        2. Integrate user's data with the atlas via fusemap_integrate
        3. Finalise main-level cell type annotations via finalize_mainlevel
        4. Annotate sub-level cell types via annotate_sublevel
        5. Plot cell type composition changes across conditions

        Input MUST include:
        - The user's data file path (full absolute path to the directory containing .h5ad files)
        - Output directory: "Save results to '<path>'"
        - AtlasAgent output (verbatim) including section IDs, if available — this is critical
          for accurate spatial mapping
        - Description of the dataset (disease model, embryo, normal brain, etc.)

        The agent automatically selects the correct pipeline strategy based on the dataset type:
        - Normal / partial mouse brain → molCCF mapping only
        - Disease model → molCCF mapping + spatial integration + annotation
        - Non-brain organ or embryo → spatial integration + annotation only

        Do not change output message of this agent.
        """
    )
