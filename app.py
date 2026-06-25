### --- app.py ---
import openai
import streamlit as st
import os
import datetime
from tempfile import NamedTemporaryFile
from agent_setup.config import create_llm, SUPPORTED_MODELS
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from audio_recorder_streamlit import audio_recorder
from agent_setup.supervisor_graph import create_supervisor_graph
from agent_setup.progress_utils import init_queue
import re
import threading
import time
from pathlib import Path
import pandas as pd
import shutil
from agent_setup.prompt import supervisor_agent_prompt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress NoSessionContext warnings from LangChain callbacks invoked in background threads.
# These warnings are expected and non-fatal: sub-agents run in ThreadPoolExecutor threads
# which lack Streamlit session context. Raising the level to ERROR hides the noise.
import logging
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)


# ── Agent Trace Timeline renderer ──────────────────────────────────────────────

# Level styling: (icon, CSS colour class label we inject)
_LEVEL_STYLE = {
    "success": ("✅", "#22c55e"),   # green
    "error":   ("❌", "#ef4444"),   # red
    "warning": ("⚠️", "#f59e0b"),  # amber
    "info":    ("",   "#94a3b8"),   # slate
}

# Per-agent accent colours (used as left-border stripe)
_AGENT_COLOURS = {
    "Supervisor":     "#6366f1",   # indigo
    "AtlasAgent":     "#0ea5e9",   # sky blue
    "ResearchAgent":  "#8b5cf6",   # violet
    "CodingAgent":    "#10b981",   # emerald
    "FuseMapAgent":   "#f97316",   # orange
}


def _agent_colour(agent: str) -> str:
    return _AGENT_COLOURS.get(agent, "#64748b")


def render_agent_trace(trace: list[dict], turn_label: str = ""):
    """
    Render a structured agent trace as an expandable timeline panel.

    Features:
      - Agent multiselect filter (show/hide per-agent rows)
      - Keyword search box (instant client-side filter)
      - Full timeline with coloured left-border per agent

    Each entry in `trace` is a dict:
        { ts, level, agent, message }
    """
    if not trace:
        return

    n_steps = len(trace)
    agents_used = list(dict.fromkeys(e["agent"] for e in trace))   # preserve order

    # Build badge HTML for agents used
    badges = " ".join(
        f'<span style="background:{_agent_colour(a)};color:#fff;'
        f'border-radius:4px;padding:1px 7px;font-size:0.72rem;font-weight:600;">{a}</span>'
        for a in agents_used
    )

    summary_html = (
        f'<span style="font-size:0.82rem;color:#94a3b8;">'
        f'{n_steps} steps · {badges}</span>'
    )

    label = f"🔍 Agent Trace{' — ' + turn_label if turn_label else ''}"

    with st.expander(label, expanded=False):
        st.markdown(summary_html, unsafe_allow_html=True)

        # ── Filter controls ────────────────────────────────────────────────
        col_f, col_s = st.columns([1, 2])
        with col_f:
            selected_agents = st.multiselect(
                "Filter by Agent",
                options=agents_used,
                default=agents_used,
                key=f"_trace_agents_{turn_label}",
                label_visibility="collapsed",
            )
        with col_s:
            keyword = st.text_input(
                "Search",
                placeholder="🔍 Search trace...",
                key=f"_trace_kw_{turn_label}",
                label_visibility="collapsed",
            ).strip().lower()
        # ──────────────────────────────────────────────────────────────────

        st.markdown("---")

        displayed = 0
        for entry in trace:
            ts      = entry.get("ts", "")
            level   = entry.get("level", "info")
            agent   = entry.get("agent", "Supervisor")
            message = entry.get("message", "")

            # Apply filters
            if selected_agents and agent not in selected_agents:
                continue
            if keyword and keyword not in message.lower():
                continue

            displayed += 1
            icon, colour = _LEVEL_STYLE.get(level, ("", "#94a3b8"))[:2]
            border = _agent_colour(agent)
            clean_msg = message.strip()

            row_html = f"""
<div style="
    display:flex;align-items:flex-start;gap:10px;
    border-left:3px solid {border};
    padding:5px 0 5px 10px;
    margin-bottom:4px;
">
  <span style="font-size:0.72rem;color:#64748b;white-space:nowrap;min-width:54px;padding-top:2px;">{ts}</span>
  <span style="font-size:0.8rem;font-weight:600;color:{border};white-space:nowrap;min-width:96px;padding-top:2px;">{agent}</span>
  <span style="font-size:0.82rem;color:{colour};flex:1;">{clean_msg}</span>
</div>"""
            st.markdown(row_html, unsafe_allow_html=True)

        if displayed == 0:
            st.caption("No entries match the current filter.")



# ── Trace export helpers ───────────────────────────────────────────────────────

import json as _json


def _build_trace_export(agent_traces: dict) -> tuple[bytes, bytes]:
    """
    Serialise all stored agent traces to JSON and CSV bytes for download.

    Returns:
        (json_bytes, csv_bytes)
    """
    # Flatten all traces into a single list with turn metadata
    rows = []
    for ai_idx, trace in sorted(agent_traces.items()):
        turn_num = ai_idx // 2
        for entry in trace:
            rows.append({
                "turn":    turn_num,
                "ts":      entry.get("ts", ""),
                "agent":   entry.get("agent", ""),
                "level":   entry.get("level", ""),
                "message": entry.get("message", ""),
            })

    json_bytes = _json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")

    # CSV via pandas (already imported)
    if rows:
        df = pd.DataFrame(rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
    else:
        csv_bytes = b"turn,ts,agent,level,message\n"

    return json_bytes, csv_bytes


def setup_sidebar():
    st.set_page_config(page_title="Spatial Brain AI Agent", page_icon="🧠")

    ### convert the sidebar to a markdown text
    st.sidebar.markdown("""
    ## 🧠 Spatial Brain AI Agent
    """)
    
    model_choice = st.sidebar.radio(
        "Choose a model:", tuple(SUPPORTED_MODELS.keys())
    )
    
    # Determine default API key based on model choice
    default_api_key = ""
    if "gpt" in model_choice.lower():
        default_api_key = os.getenv("OPENAI_API_KEY", "")
    elif "claude" in model_choice.lower():
        default_api_key = os.getenv("ANTHROPIC_API_KEY", "")

    api_key = st.sidebar.text_input("Enter your API key:", value=default_api_key, type="password")
    base_url = st.sidebar.text_input("Enter your base URL (optional):", value=os.getenv("BASE_URL", ""), type="password")
    tavily_api_key = st.sidebar.text_input("Enter your Tavily API key:", value=os.getenv("TAVILY_API_KEY", ""), type="password")

    return model_choice, api_key, tavily_api_key, base_url

def main():
    save_path = './output/'
    # Note: output subdirectories (data/, figures/, fusemap/) are created
    # on demand by each tool when it actually writes a file, so we do not
    # pre-create them here.  Pre-creating them caused empty ghost folders
    # at the output/ root whenever the user chose a custom output_dir
    # (e.g. output/my_run/) instead of the default output/.

    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = []

    # agent_traces: list of lists — one trace (list[dict]) per AI turn.
    # Index aligns with AI message index so we can look it up on replay.
    if "agent_traces" not in st.session_state:
        st.session_state.agent_traces = {}   # {ai_msg_index: [TraceEntry, ...]}

    model_choice, api_key, tavily_api_key, base_url = setup_sidebar()
    
    # Create LLM using the new function
    if api_key and tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        # Ensure the specific provider key is also set in environ for LangChain
        if "gpt" in model_choice.lower():
            os.environ["OPENAI_API_KEY"] = api_key
        elif "claude" in model_choice.lower():
            os.environ["ANTHROPIC_API_KEY"] = api_key
            
        llm = create_llm(model_choice=model_choice, api_key=api_key, base_url=base_url)
    else:
        st.error("Please provide an API key and a Tavily API key (via .env or sidebar) to continue.")
        st.stop()

        
    prompt = None

    # st.title("Spatial Brain AI Agent")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="memory", output_key="output"
    )

    # Derive a stable thread_id for this browser tab + conversation epoch.
    # Bumping turn_count on reset creates a new thread_id, which effectively
    # starts a fresh conversation in the MemorySaver without deleting old data.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
        _ctx = _get_ctx()
        _session_id = _ctx.session_id if _ctx else "default"
    except Exception:
        _session_id = "default"
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0
    thread_id = f"{_session_id}_{st.session_state.turn_count}"

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}
        st.session_state.agent_traces = {}
        # Bump turn_count → new thread_id → fresh supervisor memory context
        st.session_state.turn_count += 1
        thread_id = f"{_session_id}_{st.session_state.turn_count}"
        # Clear cached agents so they are rebuilt with the next prompt
        for _k in [k for k in st.session_state if k.startswith("_supervisor_")]:
            del st.session_state[_k]
        prompt = None

    # ── Sidebar: Trace Export panel ───────────────────────────────────────────
    if st.session_state.get("agent_traces"):
        st.sidebar.markdown("---")
        with st.sidebar.expander("📥 Trace Export", expanded=False):
            st.caption("Download the full agent trace for this session.")
            json_bytes, csv_bytes = _build_trace_export(st.session_state.agent_traces)
            ts_label = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            col_j, col_c = st.columns(2)
            with col_j:
                st.download_button(
                    label="⬇ JSON",
                    data=json_bytes,
                    file_name=f"agent_trace_{ts_label}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_c:
                st.download_button(
                    label="⬇ CSV",
                    data=csv_bytes,
                    file_name=f"agent_trace_{ts_label}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    # ─────────────────────────────────────────────────────────────────────────

    avatars = {"human": "user", "ai": "assistant"}

    # ── Replay historical messages including persisted traces ──────────────────
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type], avatar=f'agent_setup/img/{avatars[msg.type]}.jpeg'):
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                    st.write(step[0].log)
                    st.write(f"**{step[1]}**")

            # Show persisted agent trace for AI messages
            if msg.type == "ai" and idx in st.session_state.agent_traces:
                render_agent_trace(
                    st.session_state.agent_traces[idx],
                    turn_label=f"Turn {idx // 2}"
                )

            st.write(msg.content)


    if not prompt:
        prompt = st.chat_input(placeholder="What would you like to know?")

    # Cache the compiled supervisor graph in session_state keyed by model.
    # The graph is rebuilt only when the model changes or after "Reset chat history".
    _supervisor_key = f"_supervisor_{model_choice}"
    if _supervisor_key not in st.session_state:
        st.session_state[_supervisor_key] = create_supervisor_graph(llm)
    supervisor_graph = st.session_state[_supervisor_key]

    if prompt:
        st.chat_message("user", avatar='agent_setup/img/user.jpeg').write(prompt)

        # ── Build conversation history for multi-turn context ─────────────────
        # msgs is only populated by explicit add_*_message() calls below.
        # Read BEFORE adding the current user message so we only get prior turns.
        # msgs starts with an AI greeting, so pair by type rather than by index.
        _pairs = []
        _pending_user = None
        for _msg in msgs.messages:
            if _msg.type == "human":
                _pending_user = _msg.content
            elif _msg.type == "ai" and _pending_user is not None:
                _pairs.append(f"User: {_pending_user}\nAssistant: {_msg.content}")
                _pending_user = None
        _history_context = "\n\n".join(_pairs[-3:])  # keep last 3 exchanges max
        augmented_query = (
            f"Conversation history (refer to this when the user says 'those sections', "
            f"'the same cells', etc.):\n{_history_context}\n\nCurrent question: {prompt}"
            if _history_context else prompt
        )
        msgs.add_user_message(prompt)   # persist current user turn
        # ─────────────────────────────────────────────────────────────────────

        # ── Intent classification removed ─────────────────────────────────────
        # The ReAct Supervisor dynamically decides which agents to call based
        # on the query — no pre-classification needed.
        # ─────────────────────────────────────────────────────────────────────

        # Extract output_dir from the user message.
        import re as _re
        _od_match = _re.search(
            r"(?:save results? at|output (?:path|dir)|save (?:to|at))\s+['\"]?([^\s'\"]+)['\"]?",
            prompt, _re.IGNORECASE
        )
        output_dir = _od_match.group(1) if _od_match else save_path

        # ── Security: validate that the extracted path is inside the
        # allowed output root.  Malicious prompt injection could embed
        # a path like "/path/to/raw_data/" and trick all Atlas
        # tools into writing there.  Any out-of-scope path is silently
        # reset to save_path and a warning is added to the trace queue.
        _allowed_root = os.path.abspath("output")
        _resolved_od  = os.path.abspath(output_dir)
        if _resolved_od != _allowed_root and \
                not _resolved_od.startswith(_allowed_root + os.sep):
            from agent_setup.progress_utils import log_progress
            log_progress(
                f"⚠️ [Security] Requested output_dir '{output_dir}' is outside "
                f"the allowed root. Results will be saved to '{save_path}' instead."
            )
            output_dir = save_path

        # Append the (possibly reset) output directory to the query so all sub-agents see it.
        augmented_query += f"\n\nSave all output files to '{output_dir}'"

        with st.chat_message("assistant", avatar='agent_setup/img/assistant.jpeg'):
            # Initialize a fresh progress queue for this request.
            # Background threads (LangGraph nodes) write to it via log_progress();
            # the main Streamlit thread polls it here and updates the UI safely.
            progress_queue = init_queue()

            result_holder = {}
            error_holder = {}
            # Accumulate ALL trace entries for this turn so they can be persisted.
            current_trace: list[dict] = []

            def _run_agent():
                try:
                    result_holder["result"] = supervisor_graph.invoke(
                        {"input": augmented_query},
                        config={"recursion_limit": 50},
                    )
                except Exception as e:
                    error_holder["error"] = e

            agent_thread = threading.Thread(target=_run_agent, daemon=True)
            agent_thread.start()

            # ── Live streaming trace panel ─────────────────────────────────────
            # We render a compact live feed in a placeholder while the agent
            # runs, then replace it with the full persistent timeline expander
            # once done.
            live_placeholder = st.empty()

            with st.spinner("Analyzing..."):
                while agent_thread.is_alive():
                    try:
                        changed = False
                        while not progress_queue.empty():
                            entry = progress_queue.get_nowait()
                            current_trace.append(entry)
                            changed = True

                        if changed:
                            # Show last 6 entries as a compact live feed
                            lines = []
                            for e in current_trace[-6:]:
                                colour = _AGENT_COLOURS.get(e["agent"], "#64748b")
                                lines.append(
                                    f'<span style="color:{colour};font-weight:600;">[{e["agent"]}]</span> '
                                    f'<span style="color:#94a3b8;font-size:0.78rem;">{e["ts"]}</span> '
                                    f'<code style="font-size:0.8rem;">{e["message"]}</code>'
                                )
                            live_placeholder.markdown(
                                "<br>".join(lines), unsafe_allow_html=True
                            )
                    except Exception:
                        pass
                    time.sleep(0.3)

            agent_thread.join()
            # Drain any remaining entries
            try:
                while not progress_queue.empty():
                    entry = progress_queue.get_nowait()
                    current_trace.append(entry)
            except Exception:
                pass

            # Clear the live feed placeholder
            live_placeholder.empty()
            # ──────────────────────────────────────────────────────────────────

            if "error" in error_holder:
                st.error(str(error_holder["error"]))
                response = ""
            else:
                result = result_holder.get("result", {})
                # AgentWrapper returns {"output": ...}
                response = result.get("output", "")

            # Persist AI response so the next turn can read conversation history
            if response:
                msgs.add_ai_message(response)

            # ── Persist and render the full trace as an expandable timeline ────
            # The AI message was just appended; its index is len(msgs.messages) - 1
            ai_msg_idx = len(msgs.messages) - 1
            if current_trace:
                st.session_state.agent_traces[ai_msg_idx] = current_trace
                render_agent_trace(current_trace, turn_label=f"Turn {ai_msg_idx // 2}")
            # ──────────────────────────────────────────────────────────────────

            # Show response
            st.write(response)

            # Show buffered logs
            if "log_buffer" in st.session_state and st.session_state.log_buffer:
                for entry in st.session_state.log_buffer:
                    st.markdown(f"🧠 {entry}")
                st.session_state.log_buffer.clear()

            # Auto-display table if CSV exists
            if f"{save_path}/data/2D_section_ids.csv" in response:
                st.markdown("### 📊 2D Section Summary Table")
                df = pd.read_csv(f"{save_path}/data/2D_section_ids.csv")
                st.dataframe(df)

            escaped_save_path = re.escape(f"{save_path}/figures/")


            # --- Show 3D region HTML plots ---
            region_matches = re.findall(rf"{escaped_save_path}/Brain_region_(.*?)_subregion_3D\.html", response)
            for region in region_matches:
                html_path = f"{save_path}/figures/Brain_region_{region}_subregion_3D.html"
                if Path(html_path).exists():
                    st.markdown(f"### 🧠 3D Cell Plot - Region: `{region}`")
                    st.components.v1.html(open(html_path).read(), height=600)

            # --- Show 3D gene expression HTML plots ---
            gene_matches = re.findall(rf"{escaped_save_path}/Brain_region_(.*?)_gene_(.*?)_3D\.html", response)
            for region, gene in gene_matches:
                html_path = f"{save_path}/figures/Brain_region_{region}_gene_{gene}_3D.html"
                if Path(html_path).exists():
                    st.markdown(f"### 🧬 3D Gene Plot - `{gene}` in Region `{region}`")
                    st.components.v1.html(open(html_path).read(), height=600)

            # --- Show 2D Section gene plots ---
            gene2d_matches = re.findall(rf"{escaped_save_path}/Section_(.*?)_gene_(.*?)\.png", response)
            for sid, gene in gene2d_matches:
                img_path = f"{save_path}//figures/Section_{sid}_gene_{gene}.png"
                if Path(img_path).exists():
                    st.markdown(f"### 🔬 Gene Expression in Section `{sid}` - `{gene}`")
                    st.image(img_path)

            # --- Show 2D Section cell type plots ---
            celltype_matches = re.findall(rf"{escaped_save_path}/Section_(.*?)_cell_type\.png", response)
            for sid in celltype_matches:
                img_path = f"{save_path}/figures/Section_{sid}_cell_type.png"
                if Path(img_path).exists():
                    st.markdown(f"### 🧫 Cell Types in Section `{sid}`")
                    st.image(img_path)

            # --- Show 2D Section brain region plots ---
            region2d_matches = re.findall(rf"{escaped_save_path}/Section_(.*?)_brain_region\.png", response)
            for sid in region2d_matches:
                img_path = f"{save_path}/figures/Section_{sid}_brain_region.png"
                if Path(img_path).exists():
                    st.markdown(f"### 🧠 Brain Regions in Section `{sid}`")
                    st.image(img_path)



if __name__ == "__main__":
    main()
