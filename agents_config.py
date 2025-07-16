from __future__ import annotations

"""Agent configuration utilities."""

from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

from PHM_tools.Retrieval.retriever import create_retriever_tool
from utils.registry import get_agent, get_tool


def create_manager_agent(model: Model) -> CodeAgent:
    """Return a manager agent with preconfigured sub-agents.

    Parameters
    ----------
    model:
        Model instance created via :func:`model_config.configure_model`.
    """
    # Search agent with web tools
    search_agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        description="This agent can perform web searches and fetch webpages.",
        return_full_result=True,
    )

    # PHM analysis agent using registered tools
    FeatureTools = get_tool("FeatureExtractorTools")
    SignalTools = get_tool("SignalProcessingTools")
    PHMAgentCls = ToolCallingAgent # get_agent("PHMAgent")
    phm_agent = PHMAgentCls(
        tools=[FeatureTools(), SignalTools()],
        model=model,
        name="phm_agent",
        description="Agent for PHM signal analysis.",
        return_full_result=True,
        add_base_tools=True,
    )

    # Retrieval agent utilizing semantic search
    retriever_tool = create_retriever_tool()
    retrieval_agent = ToolCallingAgent(
        tools=[retriever_tool],
        model=model,
        name="retrieval_agent",
        description="Agent that performs document retrieval via vector search.",
        return_full_result=True,
        add_base_tools=True,
    )

    # Manager agent orchestrating both sub agents
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent, phm_agent, retrieval_agent],
        return_full_result=True,
        add_base_tools=True,
    )
    return manager_agent


# def create_rag_agent(model: Model, vector_store) -> CodeAgent:
#     """Return a simple RAG agent using :class:`RetrieverTool`."""

#     RetrieverTool = get_tool("RetrieverTool")
#     retriever = RetrieverTool(vector_store)

#     rag_agent = CodeAgent(
#         tools=[retriever],
#         model=model,
#         max_steps=4,
#         verbosity_level=2,
#         stream_outputs=True,
#     )
#     return rag_agent
