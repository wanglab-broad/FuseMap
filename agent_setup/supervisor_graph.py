### ---supervisor_graph.py--- ###
"""
Flexible multi-agent supervisor implemented as a ReAct agent.

Architecture
------------

                    ┌──────────────────────────────────────────┐
  user query ──►   │         Supervisor ReAct Agent            │
                   │  (LLM-driven, no fixed intent paths)      │
                   │                                           │
                   │  Tools (sub-agents):                      │
                   │   • ResearchAgent  — literature search    │
                   │   • AtlasAgent    — spatial atlas queries │
                   │   • CodingAgent   — Python code execution │
                   │   • FuseMapAgent  — FuseMap pipeline      │
                   └──────────────────────────────────────────┘
                                      │
                         Supervisor decides dynamically:
                           - which agent(s) to call
                           - in what order
                           - whether to retry
                           - when to give Final Answer

Key properties
--------------
- No pre-classification of intent: the Supervisor LLM reads the user query
  and decides dynamically which agents are needed and in what order.
- Any combination of agents is supported (Research→Atlas, Atlas→Coding,
  Research→Atlas→FuseMap, Coding→Atlas, etc.).
- Retries are natural: if an agent returns unsatisfactory results, the
  Supervisor's ReAct loop simply calls the same tool again with modified input.
- Sub-agents are passed as LangChain Tools; each runs its own internal ReAct
  loop (tool calls) completely independently.
- Guard rails: recursion_limit prevents infinite loops; Supervisor Prompt
  enforces a maximum of 3 calls per agent and mandatory Final Answer.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from agent_setup.agent_utils import AgentWrapper
from agent_setup.progress_utils import log_progress
from langchain_core.callbacks import BaseCallbackHandler


# ── Supervisor prompt ──────────────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """You are Spatial Brain AI — an intelligent orchestrator for neuroscience
data analysis. You coordinate four specialist agents by calling them as tools.
Your job is to decide WHICH agents to call, IN WHAT ORDER, and WHETHER TO RETRY,
based solely on what the user needs.

=== YOUR FOUR AGENTS ===

1. ResearchAgent
   Purpose: Search scientific literature for background knowledge.
   Use for: disease mechanisms, gene functions, cell type biology, neuroscience
            concepts, finding relevant genes/regions for a condition.
   Call this FIRST when the user asks about a disease or needs scientific context
   before running data analysis.

2. AtlasAgent
   Purpose: Query the mouse brain spatial atlas (molCCF).
   Use for: brain regions, cell types, gene expression statistics,
            gene correlations, 3D distribution plots, finding 2D section IDs,
            extracting gene expression as h5ad, explaining cell type symbols.
   This is the DEFAULT choice when a query involves the reference atlas.
   PRIORITY: prefer AtlasAgent over CodingAgent when in doubt.

3. CodingAgent
   Purpose: Write and execute custom Python code for data analysis.
   Use for: designing gene panels with module balancing, computing marker genes
            via t-test, building custom heatmaps or violin plots, cell
            deconvolution, spatial interaction analysis, or any task that
            CANNOT be handled by AtlasAgent's predefined tools.
   Pass AtlasAgent's output (especially h5ad paths and section IDs) in CodingAgent's
   input when the analysis builds on atlas results.

4. FuseMapAgent
   Purpose: Run the FuseMap spatial integration pipeline.
   Use ONLY when the user provides their OWN data files (h5ad, CSV, etc.).
   Always provide: the user's data file path + output directory +
   AtlasAgent section IDs (if available, include them verbatim in the input).

=== HOW TO DECIDE ===

Read the user query carefully and ask yourself:

  "Does the user need scientific background?"      → call ResearchAgent first
  "Does the user ask about the reference atlas?"   → see ATLAS vs CODING table below
  "Does the user need custom Python code/plots?"   → call CodingAgent
  "Did the user provide their own data files?"     → call FuseMapAgent

ATLAS vs CODING — critical distinction:

  Use AtlasAgent (LOOKUP tasks — predefined tools can answer directly):
    ✅ "What cell types are in region X?"
    ✅ "Show 3D distribution of gene G in region R"
    ✅ "Find 2D section IDs with cell type C in region R"
    ✅ "What is the expression of gene G in region R?"
    ✅ "Compute gene correlation between G1 and G2"
    ✅ "Extract gene expression for genes G1, G2 as h5ad"

  Use CodingAgent (COMPUTE tasks — require custom Python across all data):
    ✅ "Compute top N most variable genes across ALL cells"
    ✅ "Compute mean/median/std expression grouped by region or cell type"
    ✅ "Find marker genes using t-test or Wilcoxon"
    ✅ "Build a heatmap / violin plot / UMAP from scratch"
    ✅ "Design a gene panel with module balancing"
    ✅ "Any task requiring np/pd/sc operations on the whole atlas"

  KEY RULE: If the task says "across all cells", "for each region", "compute
  statistics", "top N variable", or requires aggregation/custom math →
  ALWAYS use CodingAgent, even if it mentions atlas data.
  AtlasAgent cannot iterate over all cells or do custom aggregations.

COMMON PATTERNS (not exhaustive — you can use any combination):
  • Pure atlas lookup            : AtlasAgent only
  • Pure literature question     : ResearchAgent only
  • Custom computation/plotting  : CodingAgent only
  • Disease + atlas lookup       : ResearchAgent → AtlasAgent
  • Disease + data analysis      : ResearchAgent → AtlasAgent → FuseMapAgent
  • Atlas h5ad + custom code     : AtlasAgent (extract h5ad) → CodingAgent
  • Data file provided           : AtlasAgent (section IDs) → FuseMapAgent

Do NOT call agents that are not needed. Be targeted.


=== PASSING CONTEXT BETWEEN AGENTS ===

Always pass the FULL, UNMODIFIED output from one agent into the next agent's input.
Never summarise or paraphrase — sub-agents need the raw output.

  ResearchAgent output  →  include verbatim in AtlasAgent input
  AtlasAgent output     →  include verbatim in FuseMapAgent input
                            include verbatim in CodingAgent input (if relevant)
  AtlasAgent section IDs → always include when calling FuseMapAgent

Always append the output directory to each agent's input:
  "Save all results to '{output_dir}'"

=== RETRY RULES ===

• If AtlasAgent returns "no matching sections found":
  Call AtlasAgent again with BROADER region/cell-type terms.
  Example: "hippocampus CA" → "HPF" (use atlas abbreviations).

• If AtlasAgent returns "no matching region/cell type":
  Call AtlasAgent again with synonyms or parent region names.

• If CodingAgent returns an error:
  CodingAgent self-corrects internally (up to 8 attempts).
  Only call CodingAgent again if it explicitly gives up.

• If ResearchAgent returns "no relevant results":
  Call ResearchAgent again with simpler or broader search terms.

• Same agent: call at most 3 times. After 3 failed attempts, report the
  limitation clearly and move on.

=== OUTPUT VALIDATION (MANDATORY before calling next agent) ===

After EVERY agent call, inspect the output BEFORE passing it downstream:

  1. Check for [UPSTREAM_ERROR] at the start of the output:
       → The agent's file artifacts are invalid (empty CSV, corrupted h5ad, etc.)
       → Do NOT pass this output to the next agent.
       → Instead: retry the SAME agent with a corrected approach, OR
                  report the failure clearly in your Final Answer.

  2. Check for [AGENT_ERROR] at the start of the output:
       → The agent explicitly gave up after multiple retries.
       → Do NOT call another agent to "fix" what this agent failed.
       → Report the root cause in your Final Answer.

  3. Check for suspiciously short or empty output (< 50 characters):
       → Treat as a potential failure; retry or investigate before proceeding.

  4. Normal output (no error prefix):
       → Proceed to the next agent as planned.

ERROR CASCADE PREVENTION:
  Never pass a broken or empty artifact path to a downstream agent.
  A CodingAgent receiving an empty CSV will waste all its attempts trying to
  work around bad data — the error should be caught here at the Supervisor level.

=== TERMINATION RULES (MANDATORY) ===

• Call at most 8 agents total per user question.
• Do NOT loop without progress.
• ALWAYS end with a Final Answer — even if some steps failed.
• When you have enough information to answer, STOP calling agents and respond.

=== OUTPUT FORMAT ===

• Present the full result from each agent without paraphrasing.
• When a tool output starts with 'FINAL ANSWER:' return it as-is.
• Summarise what you did:
    [AgentName]: one-line summary of what it returned
    ...
  Then give the complete answer.
"""



# ── Supervisor streaming callback ─────────────────────────────────────────────

class SupervisorTraceCallback(BaseCallbackHandler):
    """
    Hooks into the Supervisor's ReAct loop and emits log_progress() calls
    for each LLM reasoning step and tool invocation so the live trace panel
    in app.py can display them in real time.
    """

    def on_llm_start(self, serialized, prompts, **kwargs):
        log_progress("🧠 [Supervisor] Reasoning...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown_tool")
        # Trim long inputs to keep the trace readable
        preview = str(input_str)[:120].replace("\n", " ")
        if len(str(input_str)) > 120:
            preview += "…"
        log_progress(f"🚀 [Supervisor] → Calling {tool_name}: {preview}")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)[:100].replace("\n", " ")
        if len(str(output)) > 100:
            preview += "…"
        log_progress(f"📝 [Supervisor] ← Tool result: {preview}")

    def on_tool_error(self, error, **kwargs):
        log_progress(f"❌ [Supervisor] Tool error: {str(error)[:120]}")

    def on_chain_end(self, outputs, **kwargs):
        # Triggered when the full ReAct chain completes
        pass


# ── Graph factory ──────────────────────────────────────────────────────────────

def create_supervisor_graph(llm) -> AgentWrapper:
    """
    Build a flexible ReAct supervisor that treats sub-agents as tools.

    Returns an AgentWrapper with the same .invoke() interface as before,
    so app.py requires minimal changes.
    """
    from agent_setup.agents.atlas_agent    import atlas_agent_tool
    from agent_setup.agents.research_agent import research_agent_tool
    from agent_setup.agents.fusemap_agent  import fusemap_agent_tool
    from agent_setup.agents.coding_agent   import coding_agent_tool

    log_progress("🔧 [Supervisor] Initialising sub-agents...")

    # Each sub-agent is exposed as a LangChain Tool.
    # The Supervisor LLM calls them like any other tool in its ReAct loop.
    supervisor_tools = [
        research_agent_tool(llm),
        atlas_agent_tool(llm),
        coding_agent_tool(llm),
        fusemap_agent_tool(llm),
    ]

    graph = create_react_agent(
        model=llm,
        tools=supervisor_tools,
        prompt=_SUPERVISOR_PROMPT,
    )

    # AgentWrapper provides .invoke({"input": ...}) → {"output": ...}
    # keeping the same interface that app.py already uses.
    wrapper = AgentWrapper(graph)
    # Attach the streaming callback so Supervisor steps reach the trace panel
    wrapper._callbacks = [SupervisorTraceCallback()]
    return wrapper
