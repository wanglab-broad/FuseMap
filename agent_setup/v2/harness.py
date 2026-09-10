"""FuseMap Agent v2 harness: single CodeAct agent on LangChain 1.x / LangGraph.

One main agent whose primary tool is a persistent Jupyter kernel armed with the
FuseMap foundation model + molCCF atlas. Thread-scoped SQLite checkpointing;
context compaction via summarization middleware when available.
"""

import contextvars
import os
from pathlib import Path

from agent_setup.v2.kernel import get_session
from agent_setup.v2.prompts import build_system_prompt

V2_DIR = Path(__file__).parent
CURRENT_THREAD = contextvars.ContextVar("fusemap_thread", default="default")
CURRENT_TAVILY_KEY = contextvars.ContextVar("fusemap_tavily", default=None)


# ---------------------------------------------------------------- tools ----
def _build_tools():
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class RunPythonArgs(BaseModel):
        code: str = Field(description="Python code to execute in the persistent kernel. "
                                      "One focused logical step per call.")
        purpose: str = Field(description="One line: what this block is checking or producing.")

    @tool(args_schema=RunPythonArgs)
    def run_python(code: str, purpose: str) -> str:
        """Execute Python in the persistent FuseMap kernel (variables survive across
        calls; fusemap/molCCF/atlas helpers pre-loaded). Returns stdout (long output
        auto-saved to an artifact file with a preview), saved figure paths, and the
        traceback on error."""
        sess = get_session(CURRENT_THREAD.get())
        r = sess.execute(code)
        parts = []
        if r["text"].strip():
            parts.append(r["text"].strip())
        for img in r["images"]:
            parts.append(f"[figure saved: {img}]")
        if r["error"]:
            parts.append(f"ERROR:\n{r['error']}")
        return "\n".join(parts) or "(no output)"

    class LoadSkillArgs(BaseModel):
        name: str = Field(description="Skill file stem, e.g. 'map_and_annotate_new_data'.")

    @tool(args_schema=LoadSkillArgs)
    def load_skill(name: str) -> str:
        """Load the full text of a workflow skill (validated step-by-step playbook).
        Always load the relevant skill before a non-trivial FuseMap workflow."""
        f = V2_DIR / "skills" / f"{name}.md"
        if not f.exists():
            avail = ", ".join(p.stem for p in (V2_DIR / "skills").glob("*.md"))
            return f"skill '{name}' not found. Available: {avail}"
        return f.read_text()

    class LitArgs(BaseModel):
        query: str = Field(description="Literature question, in English.")
        max_results: int = Field(default=5, ge=1, le=10)

    @tool(args_schema=LitArgs)
    def literature_search(query: str, max_results: int = 5) -> str:
        """Search the scientific literature/web (Tavily). Returns titles, URLs and
        content snippets - synthesize and cite them in your answer."""
        key = CURRENT_TAVILY_KEY.get()
        if not key:
            return "No Tavily API key provided for this session - tell the user literature search is unavailable."
        from tavily import TavilyClient
        hits = TavilyClient(api_key=key).search(query, max_results=max_results)
        out = []
        for h in hits.get("results", []):
            out.append(f"- {h['title']}\n  {h['url']}\n  {h['content'][:400]}")
        return "\n".join(out) or "(no results)"

    return [run_python, load_skill, literature_search]


# ---------------------------------------------------------------- model ----
# No model picker: detect the provider from the key and auto-select the newest
# flagship it offers, so new releases are picked up with zero code changes.
_PREFERENCE = {
    "anthropic": ["claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
                  "claude-sonnet-4-6", "claude-sonnet-4-5"],
    "openai": ["gpt-6-astra", "gpt-6", "gpt-5.5", "gpt-5.2", "gpt-5.1", "gpt-5",
               "gpt-5-mini", "gpt-4o"],
    "google": ["gemini-3.1-pro", "gemini-3-pro", "gemini-2.5-pro"],
}


def detect_provider(api_key):
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("AIza"):
        return "google"
    return "openai"


def newest_model(api_key, provider=None, base_url=None):
    """Pick the newest flagship the account can actually use."""
    provider = provider or detect_provider(api_key)
    prefs = _PREFERENCE[provider]
    try:
        if provider == "openai":
            from openai import OpenAI
            avail = {m.id for m in OpenAI(api_key=api_key, base_url=base_url or None).models.list()}
            for m in prefs:
                if m in avail:
                    return provider, m
            gpts = sorted((m for m in avail if m.startswith("gpt-")), reverse=True)
            if gpts:
                return provider, gpts[0]
        elif provider == "anthropic":
            import anthropic
            avail = {m.id for m in anthropic.Anthropic(api_key=api_key).models.list()}
            for m in prefs:
                hit = next((a for a in sorted(avail, reverse=True)
                            if a == m or a.startswith(m + "-")), None)
                if hit:
                    return provider, hit
    except Exception:
        pass
    return provider, prefs[0]  # optimistic default; errors surface on first call


def make_llm(api_key, base_url=None):
    provider, model = newest_model(api_key, base_url=base_url)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, max_tokens=8000), model
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key), model
    from langchain_openai import ChatOpenAI
    kw = {"model": model, "api_key": api_key}
    if base_url:
        kw["base_url"] = base_url
    if model.startswith(("gpt-5", "gpt-6", "o3", "o4")):
        # reasoning models require the Responses API for function tools
        kw["use_responses_api"] = True
    return ChatOpenAI(**kw), model


# ---------------------------------------------------------------- agent ----
def build_agent(llm, checkpoint_db=None):
    from langchain.agents import create_agent

    kwargs = {}
    try:  # compaction middleware (API present in langchain>=1.0)
        from langchain.agents.middleware import SummarizationMiddleware
        kwargs["middleware"] = [SummarizationMiddleware(model=llm,
                                                        max_tokens_before_summary=60000)]
    except Exception:
        pass

    checkpointer = None
    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        db = checkpoint_db or str(V2_DIR / "workspace" / "checkpoints.sqlite")
        conn = sqlite3.connect(db, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return create_agent(
        model=llm,
        tools=_build_tools(),
        system_prompt=build_system_prompt(),
        checkpointer=checkpointer,
        **kwargs,
    )


def message_text(msg):
    """Normalize message content to plain text (Responses API returns block lists)."""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and b.get("type") in ("text", "output_text"):
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return str(c)


def _repair_pending_tool_calls(agent, cfg):
    """If the previous turn was interrupted mid-tool (e.g. the user typed a new
    message and Streamlit reran), the thread ends with an assistant message whose
    tool_calls never got outputs - the API rejects the next turn. Close them
    with synthetic interruption notices."""
    from langchain_core.messages import ToolMessage
    try:
        state = agent.get_state(cfg)
        msgs = (state.values or {}).get("messages", [])
    except Exception:
        return
    if not msgs:
        return
    answered = {m.tool_call_id for m in msgs if type(m).__name__ == "ToolMessage"}
    patches = []
    for m in msgs:
        for tc in (getattr(m, "tool_calls", None) or []):
            if tc["id"] not in answered:
                patches.append(ToolMessage(
                    content="[execution interrupted by the user before it finished; "
                            "the kernel may still hold partial state - re-run if needed]",
                    tool_call_id=tc["id"]))
    if patches:
        agent.update_state(cfg, {"messages": patches})


def run_turn(agent, user_text, thread_id, tavily_key=None):
    """Stream one turn; yields (kind, payload) events for the UI."""
    CURRENT_THREAD.set(thread_id)
    CURRENT_TAVILY_KEY.set(tavily_key)
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    _repair_pending_tool_calls(agent, cfg)
    for chunk in agent.stream({"messages": [("user", user_text)]}, cfg,
                              stream_mode="updates"):
        for node, update in chunk.items():
            for msg in ((update or {}).get("messages") or []):
                yield node, msg
