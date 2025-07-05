### ---atlas_agent.py--- ###

# from agent_setup.config import llm
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
import agent_setup.tools.brain_atlas_tools as brain_atlas_tools
import streamlit as st
import inspect
import datetime
from agent_setup.tools.brain_atlas_tools import match_brain_region_tool, match_brain_type_tool

# Global memory variable to record all outputs from each agent and tool
atlas_agent_memory = []

def add_to_atlas_memory(agent_name: str, tool_name: str, output: str):
    """Add an entry to the atlas agent memory"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "agent": agent_name,
        "tool": tool_name,
        "output": output
    }
    atlas_agent_memory.append(entry)
    return entry

def get_atlas_memory_summary() -> str:
    """Get a summary of all recorded atlas agent outputs"""
    if not atlas_agent_memory:
        return "No atlas agent outputs recorded yet."
    
    summary = "=== ATLAS AGENT MEMORY SUMMARY ===\n\n"
    for entry in atlas_agent_memory:
        summary += f"[{entry['timestamp']}] {entry['agent']} - {entry['tool']}:\n"
        summary += f"{entry['output'][:200]}...\n\n"
    return summary

def clear_atlas_memory():
    """Clear the atlas agent memory"""
    global atlas_agent_memory
    atlas_agent_memory = []

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


def create_atlas_tools(llm):
    # List of available tools
    atlas_tools = [
        match_brain_region_tool(llm),
        match_brain_type_tool(llm),
        brain_atlas_tools.plot_region_distribution,
        brain_atlas_tools.find_section_ids,
        brain_atlas_tools.read_one_section,
        brain_atlas_tools.explain_cell_type
    ]
    return atlas_tools


def create_atlas_agent(llm):
    # Setup memory
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    atlas_tools = create_atlas_tools(llm)

    tool_descriptions = extract_tool_descriptions(atlas_tools)

    # Define the system prompt
    atlas_agent_system_prompt = SystemMessage(content=f"""
    You are an expert AI assistant for searching and doing basic analysis of spatial transcriptomics data in the spatial mouse brain atlas.

    You have access to a set of tools that allow you to analyze mouse brain spatial transcriptomics data.
    {tool_descriptions}
                                              
    When a user asks to find information about specific brain regions, cell types, or genes, you must excute all the following steps:
    1. Identify gene names, cell types, and brain regions mentioned in the input.
    2. Match cell types, and brain regions to find the matched cell types symbols and brain regions symbols in the spatial brain atlas using `match_brain_regions` and `match_brain_type`.
    3. Plot 3D region and gene expression using `plot_region_distribution`.
    4. Use `find_section_ids` to find relevant 2D tissue sections with given cell types and brain regions.
    5. Use `read_one_section` to generate detailed 2D section adata and visualizations with given section id.
    If one step is not successful, you should retry the step.
    Perform all steps unless the user requests something specific.

    When the user specifically requires something like visualizing a 2D section, you can perform the corresponding tool.
    Before you finish, if there is cell type symbols from the spatial brain atlas in the input or output, you should explain the symbols with detailed information.
    If input is not specific, ask the user to provide more information.
    Keep path of the output files complete and unchanged.
    Keep all output information unchanged when you return the final answer.
    Always return a 'FINAL ANSWER' with a complete summary of all actions taken.
    """)

    system_prompt = atlas_agent_system_prompt 


    # Initialize the agent
    atlas_agent = initialize_agent(
        tools=atlas_tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs={
            "system_message": system_prompt,
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        },
        memory=memory,
        verbose=True
    )

    return atlas_agent


def atlas_agent_tool(llm):
    atlas_agent = create_atlas_agent(llm)

    def get_final_message_from_stream(input_text):
        container = st.container()
        container.markdown("🧠 Running Atlas Agent...")

        progress = container.progress(0, text="Initializing...")

        # Add input to memory
        add_to_atlas_memory("AtlasAgent", "Input", input_text)

        # Simulated stepwise progress based on the AtlasAgent's known workflow
        try:
            progress.progress(10, text="🔍 Step 1: Identifying genes, cell types, and brain regions...")
            add_to_atlas_memory("AtlasAgent", "Step1", "Identifying genes, cell types, and brain regions")
            # You could extract some summary here if needed

            progress.progress(30, text="🧬 Step 2: Matching brain regions and cell types in atlas...")
            add_to_atlas_memory("AtlasAgent", "Step2", "Matching brain regions and cell types in atlas")
            # Simulate delay or intermediate output if available

            progress.progress(50, text="🧠 Step 3: Plotting 3D regions and gene expression...")
            add_to_atlas_memory("AtlasAgent", "Step3", "Plotting 3D regions and gene expression")
            # Optionally, st.write() partial results if your tools return them

            progress.progress(70, text="🧾 Step 4: Finding matching 2D tissue sections...")
            add_to_atlas_memory("AtlasAgent", "Step4", "Finding matching 2D tissue sections")

            progress.progress(90, text="🖼️ Step 5: Generating 2D gene/cell/region plots...")
            add_to_atlas_memory("AtlasAgent", "Step5", "Generating 2D gene/cell/region plots")

            # Actual execution of all steps (this is when your tools run)
            result = atlas_agent.run(input_text)
            
            # Add final result to memory
            add_to_atlas_memory("AtlasAgent", "FinalResult", result)

            progress.progress(100, text="✅ Atlas analysis complete.")
            add_to_atlas_memory("AtlasAgent", "Completion", "Atlas analysis completed successfully")

        except Exception as e:
            progress.progress(100, text="❌ Atlas Agent failed.")
            container.error(f"Error: {str(e)}")
            add_to_atlas_memory("AtlasAgent", "Error", f"Atlas Agent failed with error: {str(e)}")
            return "FINAL ANSWER: Error occurred."

        return result
    
    return Tool(
        name="AtlasAgent",
        func=lambda input: "FINAL ANSWER: " + get_final_message_from_stream(input),
        description="""This tool is used to query, read, and analyze spatial brain atlas data.
            The functions include:
            1. Find information about specific brain regions.
            2. Find information about specific cell types.
            3. give matched region, type, genes, 3d plot of regions, cell types, genes. and save plots
            4. give matched region, type, genes, find corresponding 2d section ids. 
            5. give section id. extract and save data. plot 2d region, type, genes
            Keep path of the output files complete and unchanged.
            Do not change output message of the search agent.
            """

        # Uses a dedicated search agent to look up brain atlas: cell types, genes, regions."
    )