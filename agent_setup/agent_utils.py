### ---agent_utils.py--- ###
"""
Shared utilities for creating model-agnostic agents.

This module uses LangGraph's `create_react_agent` which supports both
OpenAI and Anthropic models via their native tool-calling protocols.
"""

import os
import inspect
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler

# Module-level checkpointer.
# Prefer SqliteSaver so conversation history survives Streamlit server restarts.
# Falls back to in-process MemorySaver if the sqlite package is not installed.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", ".agent_memory.db")
try:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    _checkpointer = SqliteSaver(_conn)
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver
    _checkpointer = MemorySaver()
    import logging as _logging
    _logging.getLogger("FuseMap").warning(
        "langgraph-checkpoint-sqlite not installed; "
        "conversation history will be lost on server restart. "
        "Run: uv add langgraph-checkpoint-sqlite"
    )


def extract_tool_descriptions(tools) -> str:
    """Extract descriptions from tool functions automatically.

    Works with both function-based (LangChain Tool) and class-based tools.
    """
    descriptions = []

    for i, tool in enumerate(tools, 1):
        if hasattr(tool, "func"):
            # Function-based tool (e.g. LangChain's Tool)
            func = tool.func
            name = getattr(tool, "name", func.__name__)
            doc = func.__doc__ or "No description available"
        else:
            # Class-based tool (custom tools)
            func = getattr(tool, "_run", None)
            name = getattr(tool, "name", tool.__class__.__name__)
            doc = getattr(tool, "description", None) or (
                func.__doc__ if func else "No description available"
            )

        if func:
            sig = inspect.signature(func)
            params = [p for p in sig.parameters if p != "log"]
        else:
            params = []

        param_str = ", ".join(params)
        description = f"{i}. {name}({param_str}): {doc.strip()}"
        descriptions.append(description)

    return "\n".join(descriptions)


# ── Sub-agent tool-level trace callback ───────────────────────────────────────

class SubAgentTraceCallback(BaseCallbackHandler):
    """
    Hooks into a sub-agent's (AtlasAgent, CodingAgent, etc.) internal ReAct loop
    and emits structured log_progress() entries for every tool call and result.

    This makes the internal tool-level execution visible in the live trace panel
    in app.py, completing the \"Sub-agent 内部工具调用 Callback\" Sprint 2.3 item.

    One instance is created per sub-agent with the matching agent_name so that
    trace entries are correctly attributed in the timeline.
    """

    def __init__(self, agent_name: str):
        super().__init__()
        self._agent_name = agent_name

    # ── standardised log helpers ──────────────────────────────────────────────

    def log_tool_call(self, tool_name: str, input_preview: str) -> None:
        """Emit a structured TOOL-CALL entry visible in the trace timeline."""
        from agent_setup.progress_utils import log_progress
        preview = str(input_preview)[:100].replace("\n", " ")
        if len(str(input_preview)) > 100:
            preview += "…"
        log_progress(f"🔬 [{self._agent_name}] → {tool_name}({preview})")

    def log_tool_result(self, tool_name: str, output_preview: str) -> None:
        """Emit a structured TOOL-RESULT entry visible in the trace timeline."""
        from agent_setup.progress_utils import log_progress
        preview = str(output_preview)[:100].replace("\n", " ")
        if len(str(output_preview)) > 100:
            preview += "…"
        log_progress(f"📊 [{self._agent_name}] ← {tool_name}: {preview}")

    # ── LangChain callback hooks ──────────────────────────────────────────────

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "tool")
        self.log_tool_call(tool_name, input_str)

    def on_tool_end(self, output, **kwargs):
        # tool name not directly available here; use generic label
        self.log_tool_result("tool", output)

    def on_tool_error(self, error, **kwargs):
        from agent_setup.progress_utils import log_progress
        log_progress(f"❌ [{self._agent_name}] Tool error: {str(error)[:120]}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        from agent_setup.progress_utils import log_progress
        log_progress(f"🧠 [{self._agent_name}] Reasoning...")

    def on_chain_end(self, outputs, **kwargs):
        pass


class AgentWrapper:
    """
    Wraps a LangGraph CompiledStateGraph agent to provide a consistent
    interface compatible with the existing codebase.

    The `.invoke({\"input\": ...})` method returns `{\"output\": ...}` to
    match the old AgentExecutor interface.

    When a `memory` (ConversationBufferMemory) is provided, prior turns are
    injected into the message list before each invocation, enabling genuine
    multi-round conversation history.
    """

    def __init__(self, graph, memory=None):
        self._graph = graph
        self._memory = memory
        self._callbacks = []   # populated by supervisor_graph.py or create_model_agnostic_agent

    def invoke(self, inputs: dict, config: dict = None) -> dict:
        """
        Run the agent with the given input.

        Args:
            inputs: Dict with "input" key containing the user message.
            config: Optional config dict.

        Returns:
            Dict with "output" key containing the agent's final response.
        """
        user_input = inputs.get("input", "")

        # Build message list including conversation history from memory
        messages = []
        if self._memory:
            try:
                mem_vars = self._memory.load_memory_variables({})
                history = mem_vars.get("memory", [])
                for msg in history:
                    if hasattr(msg, "type") and hasattr(msg, "content"):
                        if msg.type == "human":
                            messages.append(("user", msg.content))
                        elif msg.type == "ai":
                            messages.append(("assistant", msg.content))
            except Exception:
                pass

        # Append current user message
        messages.append(("user", user_input))

        # Prepare input with history-aware messages
        agent_input = {
            "input": user_input,
            "intermediate_steps": [],
            "messages": messages,
        }

        # LangGraph's default recursion_limit (25) is too low for multi-step
        # FuseMap pipelines that chain map_molCCF → integrate → finalize → annotate.
        merged_config = {"recursion_limit": 100}
        if config:
            merged_config.update(config)
        # Forward any registered callbacks (e.g. SupervisorTraceCallback, SubAgentTraceCallback)
        if self._callbacks:
            merged_config["callbacks"] = self._callbacks

        result = self._graph.invoke(agent_input, config=merged_config)

        # Extract the final AI message from the result
        result_messages = result.get("messages", [])
        output = ""
        if result_messages:
            final_msg = result_messages[-1]
            if hasattr(final_msg, "content"):
                output = final_msg.content
            else:
                output = str(final_msg)

        # Persist the exchange to memory so subsequent turns see it
        if self._memory:
            try:
                self._memory.save_context(
                    {"input": user_input},
                    {"output": output},
                )
            except Exception:
                pass

        return {"output": output}

    def run(self, input_text: str) -> str:
        """Legacy `.run()` interface for backward compatibility."""
        result = self.invoke({"input": input_text})
        return result.get("output", "")


def create_model_agnostic_agent(
    llm,
    tools: list,
    system_message: str,
    memory=None,
    verbose: bool = True,
    enable_memory: bool = False,
    agent_name: str = "",
) -> AgentWrapper:
    """
    Create an agent that works with both OpenAI and Anthropic models.

    Uses LangGraph's `create_react_agent` which leverages each provider's
    native tool-calling API (OpenAI function calling / Anthropic tool use).

    Args:
        llm: The language model instance (ChatOpenAI or ChatAnthropic).
        tools: List of LangChain tools the agent can use.
        system_message: The system prompt string for the agent.
        memory: ConversationBufferMemory for multi-turn conversation history.
                When provided, prior turns are injected into every invocation
                so the agent genuinely remembers previous exchanges.
        verbose: Whether to enable debug mode.
        enable_memory: If True, attach the module-level SqliteSaver checkpointer
                       so the top-level supervisor graph remembers state across
                       Streamlit server restarts (requires thread_id in config).
                       Sub-agents should leave this False.
        agent_name: If non-empty, attaches a SubAgentTraceCallback so that every
                    internal tool call made by this agent is emitted as a structured
                    log_progress() entry visible in the live trace timeline.
                    Pass the display name (e.g. "AtlasAgent", "CodingAgent").

    Returns:
        An AgentWrapper instance with .invoke() and .run() methods.
    """
    graph = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message,
        checkpointer=_checkpointer if enable_memory else None,
    )

    wrapper = AgentWrapper(graph, memory=memory)

    # Attach sub-agent trace callback when agent_name is provided
    if agent_name:
        wrapper._callbacks = [SubAgentTraceCallback(agent_name=agent_name)]

    return wrapper
