from __future__ import annotations

"""Agent configuration utilities."""

from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

from PHM_tools import load_default_vector_store

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
    RetrieverTool = get_tool("RetrieverTool")
    PHMAgentCls = get_agent("PHMAgent")

    retriever = RetrieverTool(load_default_vector_store())
    phm_agent = PHMAgentCls(
        tools=[FeatureTools(), SignalTools(), retriever],
        model=model,
        name="phm_agent",
        description="Agent for PHM signal analysis.",
        return_full_result=True,
    )

    # Manager agent orchestrating both sub agents
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent, phm_agent],
        return_full_result=True,
    )
    return manager_agent
