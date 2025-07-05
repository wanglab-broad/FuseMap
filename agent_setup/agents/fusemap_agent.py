### ---fusemap_agent.py--- ###
# from agent_setup.config import llm
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
import agent_setup.tools.fusemap_tool as fusemap_tool
import streamlit as st
import inspect
from agent_setup.tools.fusemap_tool import finalize_mainlevel_tool, annotate_sublevel_tool


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


# def extract_tool_descriptions(tools):
#     """Extract descriptions from tool functions automatically"""
#     descriptions = []
    
#     for i, tool_func in enumerate(tools, 1):
#         # Get the function signature
#         sig = inspect.signature(tool_func.func)
#         params = list(sig.parameters.keys())
        
#         # Remove the 'log' parameter as it's internal
#         if 'log' in params:
#             params.remove('log')
        
#         # Get the docstring
#         doc = tool_func.func.__doc__ or "No description available"
        
#         # Format the description
#         param_str = ", ".join(params)
#         description = f"{i}. {tool_func.name}({param_str}): {doc.strip()}"
#         descriptions.append(description)
    
#     return "\n".join(descriptions)


def create_fusemap_tools(llm):
    """Create fusemap tools with the provided LLM"""
    # Create tools that accept the LLM parameter
    tools = [
        fusemap_tool.map_molCCF,
        fusemap_tool.fusemap_integrate,
        # Create wrapper functions for tools that need LLM
        finalize_mainlevel_tool(llm),
        annotate_sublevel_tool(llm),
    ]
    return tools



def create_fusemap_agent(llm):
    """Create and return a fusemap agent with the provided LLM"""
    # Setup memory
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    # Create tools with LLM
    tools = create_fusemap_tools(llm)

    # Automatically extract tool descriptions
    tool_descriptions = extract_tool_descriptions(tools)


    # Define the system prompt
    fusemap_agent_system_prompt = SystemMessage(content=f"""
    You are an expert AI assistant for using FuseMap model, a deep-learning-based framework for spatial transcriptomics that bridges single-cell or single-spot gene expression with spatial contexts and consolidates various gene panels across spatial transcriptomics atlases.
                                as well as integrating query spatial transcriptomics data with the reference brain atlas and analyze.

    You have access to the following tools:
    {tool_descriptions}

    When a user asks to analyze spatial transcriptomics data, you should decide which tools to use.
    If the new sample has all cell types included in normal adult mouse brain, 
                                like a part of the mouse brain or aging mouse brain, 
                                only use molCCF mapping to transfer both main-level and sub-level cell types.

    If the new sample has partial cell types included in normal adult mouse brain, 
                                like diseased mouse brain, 
                                first use molCCF mapping to transfer only main-level,
                                then based on the results, apply spatial_integrate and annotate_new_sample.

    If the new sample has totally different cell types from normal adult mouse brain, 
                                like another organ liver, 
                                apply spatial_integrate and annotate_new_sample.

    After this, finalize main-level cell types.                        
    Then annotate sublevel cell types.

    Finally, you can call plotting functions to show cell type composition changes across conditions.

    If the user specifically request to run a tool, you should just call the tool with correct input.

    The final answer should be in the following format:
    'FINAL ANSWER'
    first take action 1, get result 1, detailed output is ... 
    then take action 2, get result 2, detailed output is ...
    then take action 3, get result 3, detailed output is ...
    """)



    # Initialize the agent
    fusemap_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs={
            "system_message": fusemap_agent_system_prompt,
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        },
        memory=memory,
        verbose=True
    )
    
    return fusemap_agent


def fusemap_agent_tool(llm):
    """Create a fusemap agent tool with the provided LLM"""
    agent = create_fusemap_agent(llm)
    
    def get_final_message_from_stream(input_text):
        container = st.container()
        container.markdown("🧠 Running FuseMap Agent...")

        progress = container.progress(0, text="Initializing...")

        try:
            result = agent.run(input_text)
            progress.progress(100, text="✅ FuseMap analysis complete.")

        except Exception as e:
            progress.progress(100, text="❌ FuseMap analysis failed.")
            container.error(f"Error: {str(e)}")
            return "FINAL ANSWER: Error occurred."

        return result


    return Tool(
        name="FuseMapAgent",
        func=lambda input: "FINAL ANSWER: " + get_final_message_from_stream(input),
        description="""This tool is used to run FuseMap analysis.
        The functions is to analyze new spatial transcriptomics data, 
        Input informationneed to specify the data path, 
        need to specify the tissue description, including normal mouse brain, at certain conditions like disease, or other organs or liver,
        need to specify if you want to analyze cell types or tissue regions.
        If the new data is related to mouse brain, input also need the corresponding section id. You can get the information from the atlas agent.
        For example:
        I have measured spatial transcriptomics datasets of E15.5 mouse embryo at path '/home/gaofeng/Desktop/gaofeng/FuseMap/user_data/embryo'. 
        Help me analyze the tissue regions.

        Another example:
        I have measured spatial transcriptomics datasets of mouse brain during aging at path '/home/gaofeng/Desktop/gaofeng/FuseMap/user_data/aging'. 
        This corresponds to the section id: '157'. 
        Help me analyze the cell types.
            """
    )
