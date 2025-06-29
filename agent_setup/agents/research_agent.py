### ---research_agent.py--- ###

from langchain_community.tools.tavily_search import TavilySearchResults
import openai
# from agent_setup.config import llm
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

import tempfile
import datetime
from tempfile import NamedTemporaryFile
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import os


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
    You are a neuroscience research assistant. When given a disease or condition, analyze cell types, genes, and tissue regions involved in diseases.
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
        container = st.container()
        container.markdown("🔍 Running Research Agent...")

        progress = container.progress(0, text="Starting research...")

        stream = search_sub_agent.stream(
            {"messages": [("user", input_text)]},
            stream_mode="values"
        )

        final = None
        for i, step in enumerate(stream):
            progress.progress(min(100, (i + 1) * 10), text="Analyzing literature...")
            messages = step.get("messages", [])
            if messages:
                final = messages[-1]

        progress.progress(100, text="Research complete.")
        
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
        description="""This tool is used to search for information about change of gene expression, cell types, and brain regions, e.g., under brain diseases and conditions with references,
        under the help of a dedicated search agent.
        Do not change output message of the search agent.
    """
    )

