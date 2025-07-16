from __future__ import annotations

"""Agent configuration utilities."""

from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

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
    PHMAgentCls = get_agent("PHMAgent")
    phm_agent = PHMAgentCls(
        tools=[FeatureTools(), SignalTools()],
        model=model,
        name="phm_agent",
        description="Agent for PHM signal analysis.",
        return_full_result=True,
    )

    # Retrieval agent utilizing semantic search
    RetrieverTool = get_tool("RetrieverTool")
    retrieval_agent = ToolCallingAgent(
        tools=[RetrieverTool()],
        model=model,
        name="retrieval_agent",
        description="Agent that performs document retrieval via vector search.",
        return_full_result=True,
    )

    # Manager agent orchestrating both sub agents
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent, phm_agent, retrieval_agent],
        return_full_result=True,
    )
    return manager_agent
