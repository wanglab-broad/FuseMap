"""FuseMap Agent v2 - Streamlit UI.

Run with the v2 harness environment:
    /ewsc/yhe/miniconda3/envs/fusemap_agent_v2/bin/streamlit run app_v2.py
"""

import re
import uuid

import streamlit as st

from agent_setup.v2.harness import build_agent, make_llm, message_text, run_turn

st.set_page_config(page_title="FuseMap Agent v2", page_icon="🧠", layout="wide")

FIG_RE = re.compile(r"\[figure saved: (.+?)\]")


def sidebar():
    st.sidebar.markdown("## 🧠 FuseMap Agent v2")
    st.sidebar.caption("CodeAct agent armed with the FuseMap foundation model "
                       "and the molCCF atlas (10.4M cells).")
    api_key = st.sidebar.text_input("Model API key", type="password")
    base_url = st.sidebar.text_input("Base URL (optional, for gateways)", type="password")
    tavily = st.sidebar.text_input("Tavily key (optional, literature)", type="password")
    if st.sidebar.button("New session"):
        st.session_state.pop("thread_id", None)
        st.session_state.pop("history", None)
        st.rerun()
    return (api_key or "").strip(), (base_url or "").strip(), (tavily or "").strip()


def render_tool_message(msg):
    text = message_text(msg)
    figs = FIG_RE.findall(text)
    body = FIG_RE.sub("", text).strip()
    if body:
        st.code(body[:4000], language=None)
    for f in figs:
        st.image(f)


def main():
    api_key, base_url, tavily = sidebar()
    if not api_key:
        st.info("Paste your model API key in the sidebar to start - the newest "
                "flagship model for your provider is selected automatically.")
        st.stop()

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid.uuid4().hex[:12]
        st.session_state.history = []

    if "agent" not in st.session_state or st.session_state.get("agent_key") != (api_key, base_url):
        llm, model = make_llm(api_key, base_url=base_url or None)
        st.session_state.agent = build_agent(llm)
        st.session_state.agent_key = (api_key, base_url)
        st.session_state.model_name = model
    host = (base_url or "api.openai.com / api.anthropic.com").split("//")[-1].split("/")[0]
    st.sidebar.caption(f"model: **{st.session_state.model_name}**  \nendpoint: {host}")

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    q = st.chat_input("Ask about the brain atlas, or bring your own data (give a path)...")
    if not q:
        st.stop()

    st.session_state.history.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    final = ""
    with st.chat_message("assistant"):
      try:
        status = None
        for node, msg in run_turn(st.session_state.agent, q,
                                  st.session_state.thread_id, tavily_key=tavily or None):
            mtype = type(msg).__name__
            if mtype == "AIMessage":
                for tc in (getattr(msg, "tool_calls", None) or []):
                    label = tc["args"].get("purpose") or tc["name"]
                    status = st.status(f"⏳ {tc['name']}: {label}", state="running")
                    if tc["name"] == "run_python":
                        status.code(tc["args"].get("code", ""), language="python")
                txt = message_text(msg)
                if txt.strip():
                    final = txt
            elif mtype == "ToolMessage":
                body = message_text(msg)
                if status is not None:
                    status.update(state="complete")
                    with status:
                        st.code(FIG_RE.sub("", body).strip()[:4000] or "(no output)", language=None)
                for f in FIG_RE.findall(body):
                    st.image(f)
      except Exception as e:
        st.error(f"Model call failed ({type(e).__name__}): {str(e)[:300]}\n\n"
                 f"Check the API key and Base URL in the sidebar (no spaces/newlines).")
        st.stop()
      if True:
        if final:
            # resolve artifact references (agent saves under its thread workspace)
            from agent_setup.v2.kernel import REPO
            import os
            ws = os.path.join(REPO, "agent_setup", "v2", "workspace",
                              st.session_state.thread_id)
            shown = st.session_state.setdefault("shown_figs", set())
            st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", final))
            for m in re.findall(r"[\w./-]+\.(?:png|jpg|svg)", final):
                cand = m if os.path.isabs(m) else os.path.join(ws, m)
                if os.path.exists(cand) and cand not in shown:
                    st.image(cand); shown.add(cand)
    st.session_state.history.append(("assistant", final or "(see steps above)"))


main()
