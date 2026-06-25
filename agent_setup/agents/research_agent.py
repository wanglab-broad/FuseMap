### ---research_agent.py--- ###

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from agent_setup.progress_utils import log_progress
import os


def create_research_agent(llm):
    """
    Create a research agent that can search for information using Tavily.
    This version is designed to work outside of Streamlit context (e.g., in tests).
    """
    from agent_setup.agent_utils import AgentWrapper

    tavily_tool = TavilySearchResults(max_results=5,
                                    tavily_api_key=os.environ["TAVILY_API_KEY"])

    system_prompt = (
        "You are a helpful AI research assistant. "
        "Use the search tool to find information about scientific topics, "
        "including cell types, genes, and tissue regions in diseases. "
        "Provide comprehensive answers with references."
    )

    # Create agent using LangGraph's create_react_agent
    graph = create_react_agent(
        llm,
        tools=[tavily_tool],
        prompt=system_prompt,
    )

    return AgentWrapper(graph)


def research_agent_tool(llm):

    tavily_tool = TavilySearchResults(max_results=5,
                                    tavily_api_key=os.environ["TAVILY_API_KEY"])


    def make_system_prompt(suffix: str) -> str:
        return (
            "You are a helpful AI assistant collaborating with another assistant.\n"
            "Use tools to answer questions.\n"
            "Prefix your final answer with 'FINAL ANSWER'.\n"
            f"{suffix}"
        )

    search_sub_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        prompt=make_system_prompt("""
    You are a neuroscience research assistant. When given a disease or condition, you should analyze cell types, genes, and tissue regions involved in diseases.
    Template structure output:
    1. Affected Cell Types
    2. Affected Genes, with specific gene names
    3. Affected Tissue Regions
    You must include links to referred scientific terminology, references, and citations.
    You cannot change output from the tool. Directly output the information.
    """),
        name="research_agent"
    )

    def get_final_message_from_stream(input_text):
        # Do NOT call st.* here: this runs inside a background thread where
        # Streamlit has no session context. In bare mode, st.* calls succeed
        # silently but corrupt the frontend delta queue → white screen.
        log_progress("🔍 [ResearchAgent] Starting literature search...")

        stream = search_sub_agent.stream(
            {"messages": [("user", input_text)]},
            stream_mode="values"
        )

        final = None
        step_count = 0
        for step in stream:
            step_count += 1
            messages = step.get("messages", [])
            if messages:
                last = messages[-1]
                msg_type = type(last).__name__
                # Emit a brief live update for each ReAct step
                if hasattr(last, "content") and last.content:
                    preview = str(last.content)[:80].replace("\n", " ")
                    if len(str(last.content)) > 80:
                        preview += "…"
                    log_progress(f"🔍 [ResearchAgent] Step {step_count} ({msg_type}): {preview}")
                final = last

        log_progress("✅ [ResearchAgent] Research complete.")
        
        if isinstance(final, tuple):
            return str(final[1])
        elif hasattr(final, "content"):
            return final.content
        else:
            return str(final)

        
    # Wrap sub-agent as a tool for main agent
    return Tool(
        name="ResearchAgent",
        func=lambda input: "FINAL ANSWER: " + get_final_message_from_stream(input),
        description="""Search scientific literature for neuroscience background knowledge.

        Capabilities:
        1. Find disease mechanisms, pathology, and affected brain regions/cell types
        2. Identify relevant genes and markers for a disease or condition
        3. Explain neuroscience concepts, gene functions, and cell type biology
        4. Retrieve references and citations from scientific literature

        Use this agent FIRST when:
        - The user asks about a disease (e.g. Alzheimer's, Parkinson's)
        - Scientific context is needed before running atlas or analysis tools
        - The user wants background on genes, cell types, or brain regions

        Input: the user's question or topic to search.
        Output: structured summary with affected cell types, genes, brain regions, and references.

        Do not change output message of the search agent.
    """
    )

