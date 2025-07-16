"""Agent configuration utilities."""

from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

from utils.registry import get_agent, get_tool


def create_manager_agent(model: Model, vector_store=None) -> CodeAgent:
    """Return a manager agent with preconfigured sub-agents."""
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

    # Knowledge retrieval agent for RAG
    RetrieverTool = get_tool("RetrieverTool")
    if vector_store is not None:
        retriever_tool = RetrieverTool(vector_store)
    else:
        retriever_tool = RetrieverTool(None)
    retriever_agent = ToolCallingAgent(
        tools=[retriever_tool],
        model=model,
        name="retriever_agent",
        description="Agent that performs semantic document retrieval.",
        return_full_result=True,
    )

    # Manager agent orchestrating sub agents
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent, phm_agent, retriever_agent],
        return_full_result=True,
    )
    return manager_agent
