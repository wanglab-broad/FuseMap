### --- app.py ---
import openai
import streamlit as st
import os
import datetime
from tempfile import NamedTemporaryFile
from agent_setup.config import create_llm
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from audio_recorder_streamlit import audio_recorder
from agent_setup.agents.atlas_agent import atlas_agent_tool
from agent_setup.agents.research_agent import research_agent_tool
from agent_setup.agents.fusemap_agent import fusemap_agent_tool
import re
from pathlib import Path
import pandas as pd
import shutil
from agent_setup.prompt import supervisor_agent_prompt

def setup_sidebar():
    st.set_page_config(page_title="Spatial Brain AI Agent", page_icon="🧠")

    ### convert the sidebar to a markdown text
    st.sidebar.markdown("""
    ## 🧠 Spatial Brain AI Agent

    A collaborative multi-agent system that helps you explore:
    - Brain regions, cell types, and gene expression from the 3D mouse brain atlas at single-cell resolution
    - Changes of cell states with diseases, aging, and other conditions
    - Your own spatial transcriptomics data, integrated with the reference atlas

    ---

    ### 🔄 Modes of Use

    **1. Autonomous Mode**
                            
    Let the AI agent handle everything end-to-end:
    - *"How do the cell state changes in mouse hippocampus with Alzheimer's disease? I have measured spatial transcriptomics datasets of mouse brain with two Alzheimer's disease model at path 'path/to/data/disease'. Help me analyze the cell types."*

    **2. Interactive Mode**
                        
    Guide the analysis step-by-step:
    - *"Search literature about how the cell state changes in mouse hippocampus with Alzheimer's."*
    - *"Based on the literature / I'm interested in dopaminergic neurons and genes APP, PSEN1, PSEN2, find the corresponding spatial transcriptomics data in the 3D mouse brain atlas."*
    - *"I have measured spatial transcriptomics datasets of E15.5 mouse embryo at 'path/to/data/embryo'. Use FuseMap to analyze tissue regions."*
                            
    ---

    You can change the model or reset the chat below.
    """)
    
    # st.sidebar.info("You can choose different models or reset chat history. ")

    model_choice = st.sidebar.radio("Choose a model:", ("gpt-4o", "claude-3.7"))
    api_key = st.sidebar.text_input("Enter your API key:", type="password")
    base_url = st.sidebar.text_input("Enter your base URL (optional):", type="password")
    tavily_api_key = st.sidebar.text_input("Enter your Tavily API key:", type="password")

    # path_to_molCCF = st.sidebar.text_input("Enter the path to molCCF:", type="password")
    return model_choice, api_key, tavily_api_key, base_url





def tool_log(message: str):
    if "log_buffer" in st.session_state:
        st.session_state.log_buffer.append(message)


def main():

    save_path = './output/'
    ### remove save_path if it exists
    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path)

    os.makedirs(f'{save_path}/data', exist_ok=True)
    os.makedirs(f'{save_path}/figures', exist_ok=True)
    os.makedirs(f'{save_path}/fusemap', exist_ok=True)

    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = []

    model_choice, api_key, tavily_api_key, base_url = setup_sidebar()
    
    
    # Create LLM using the new function
    if api_key and tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key
        llm = create_llm(model_choice=model_choice, api_key=api_key, base_url=base_url)
    else:
        st.error("Please provide an OPENAI API key and a TAVILY API key to continue.")
        st.stop()
        
    prompt = None

    # st.title("Spatial Brain AI Agent")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs, return_messages=True, memory_key="memory", output_key="output"
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}
        prompt = None

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type], avatar=f'agent_setup/img/{avatars[msg.type]}.jpeg'):
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                    st.write(step[0].log)
                    st.write(f"**{step[1]}**")
            st.write(msg.content)


    if not prompt:
        prompt = st.chat_input(placeholder="What would you like to know?")

    if prompt:
        st.chat_message("user", avatar='agent_setup/img/user.jpeg').write(prompt)

        current_time = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z%z")

        content = f"""{supervisor_agent_prompt}
        Current Time: {current_time_str}.
        """

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": SystemMessage(content=content),
        }

        ### research_agent_tool
        research_agent_tool_instance = research_agent_tool(llm)

        ### atlas_agent_tool
        atlas_agent_tool_instance = atlas_agent_tool(llm)

        ### fusemap_agent_tool
        fusemap_agent_tool_instance = fusemap_agent_tool(llm)

        supervisor_agent = initialize_agent(
            [research_agent_tool_instance, atlas_agent_tool_instance, fusemap_agent_tool_instance],
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            agent_kwargs=agent_kwargs,
            memory=memory,
            verbose=False
        )

        with st.chat_message("assistant", avatar='agent_setup/img/assistant.jpeg'):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = supervisor_agent.run(input=prompt, callbacks=[st_cb])

            # Show response
            st.write(response)

            # Show buffered logs
            if st.session_state.log_buffer:
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
