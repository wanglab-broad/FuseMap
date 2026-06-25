"""
Thread-safe progress reporting for Streamlit + LangGraph.

LangGraph's ToolNode always runs tools in a ThreadPoolExecutor, so tools
cannot call st.* directly (no ScriptRunContext in threads).

Pattern:
  - Main Streamlit thread calls init_queue() before launching the agent.
  - Background threads (tools / sub-agents) call log_progress() freely.
  - Main thread polls the queue and updates st.empty() with new messages.

TraceEntry format:
  {
    "ts":      "HH:MM:SS",       # wall-clock timestamp
    "level":   "info|success|error|warning",
    "agent":   "Supervisor|AtlasAgent|...",  # parsed from bracket prefix
    "message": "raw message text",
  }
"""

import queue
import threading
import datetime
import re

_lock = threading.Lock()
_current_queue: queue.Queue = None


def init_queue() -> queue.Queue:
    """Create a fresh queue for this session. Must be called from the main thread."""
    global _current_queue
    q = queue.Queue()
    with _lock:
        _current_queue = q
    return q


def get_queue() -> queue.Queue:
    """Return the current queue. Safe to call from any thread."""
    with _lock:
        return _current_queue


# ── Emoji → level mapping ──────────────────────────────────────────────────────
_LEVEL_MAP = {
    "✅": "success",
    "❌": "error",
    "⚠️": "warning",
    "🔍": "info",
    "🧠": "info",
    "💻": "info",
    "🗺️": "info",
    "🔧": "info",
    "📊": "info",
    "🚀": "info",
    "⏳": "info",
    "📝": "info",
    "🔬": "info",   # tool-call (SubAgentTraceCallback.log_tool_call)
    "📊": "info",   # tool-result (SubAgentTraceCallback.log_tool_result)
    "🧬": "info",   # FuseMapAgent
}

# Agent name extraction: looks for "[AgentName]" or "[AgentName " pattern
_AGENT_RE = re.compile(r"\[([A-Za-z]+(?:Agent|Supervisor)?)\]")


def _parse_entry(message: str) -> dict:
    """Convert a raw log string into a structured TraceEntry dict."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # Determine level from leading emoji
    level = "info"
    for emoji, lvl in _LEVEL_MAP.items():
        if message.startswith(emoji):
            level = lvl
            break

    # Extract agent name
    m = _AGENT_RE.search(message)
    agent = m.group(1) if m else "Supervisor"

    return {"ts": ts, "level": level, "agent": agent, "message": message}


def log_progress(message: str) -> None:
    """Put a structured progress entry into the queue. Safe to call from any thread."""
    q = get_queue()
    if q is not None:
        entry = _parse_entry(message)
        q.put(entry)
