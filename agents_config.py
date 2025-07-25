from __future__ import annotations

"""Agent configuration utilities."""

from smolagents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

from agent_leader.manager_agent import ManagerAgent

from PHM_tools.Retrieval.retriever import create_retriever_tool
from utils.registry import get_agent, get_tool
from agent import create_deep_research_agent

# ---------------------------------------------------------------------------
# Sub-agent factories
# ---------------------------------------------------------------------------

def _make_search_agent(model: Model) -> ToolCallingAgent:
    """Return a simple web search assistant."""
    return ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="search_agent",
        description="Agent that performs targeted PHM web searches.",
        instructions="Search online sources for PHM related data and articles.",
        return_full_result=True,
    )


def _make_phm_agent(model: Model) -> ToolCallingAgent:
    """Return the PHM analysis agent from the registry."""
    time_feat = get_tool("extract_time_features")
    freq_feat = get_tool("extract_frequency_features")
    normalize_tool = get_tool("normalize")
    PHMAgentCls = get_agent("PHMAgent")
    return PHMAgentCls(
        tools=[time_feat, freq_feat, normalize_tool],
        model=model,
        name="phm_agent",
        description="Agent for PHM signal analysis.",
        return_full_result=True,
        add_base_tools=True,
    )


def _make_retrieval_agent(model: Model) -> ToolCallingAgent:
    """Return the document retrieval helper."""
    retriever_tool = create_retriever_tool()
    return ToolCallingAgent(
        tools=[retriever_tool],
        model=model,
        name="retrieval_agent",
        description="Agent that performs document retrieval via vector search.",
        instructions="Use the retriever tool to access the knowledge base.",
        return_full_result=True,
        add_base_tools=True,
    )


AGENT_BUILDERS = {
    "search_agent": _make_search_agent,
    "phm_agent": _make_phm_agent,
    "retrieval_agent": _make_retrieval_agent,
    "deep_research_agent": create_deep_research_agent,
}


def create_manager_agent(model: Model, config=None) -> ManagerAgent:
    """Return the high-level :class:`ManagerAgent` used by the demos.

    The previous implementation dynamically composed a manager from several
    sub-agents. The new architecture encapsulates this logic inside
    :class:`ManagerAgent`, so this function simply instantiates that class.

    Args:
        model: Configured language model instance.
        config: Optional configuration object (unused).

    Returns:
        Instance of :class:`ManagerAgent` ready for interaction.
    """

    _ = config
    return ManagerAgent(model)


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


def create_report_agent(model: Model) -> CodeAgent:
    """Return an agent specialized in PHM report writing."""
    search_agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="research_agent",
        description="Performs web searches and gathers information.",
        instructions="Gather recent PHM publications and data from the web.",
        return_full_result=True,
    )
    retrieval_tool = create_retriever_tool()
    retrieval_agent = ToolCallingAgent(
        tools=[retrieval_tool],
        model=model,
        name="retrieval_agent",
        description="Retrieves documents from the knowledge base.",
        instructions="Search the knowledge base for relevant materials.",
        return_full_result=True,
        add_base_tools=True,
    )
    report_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[search_agent, retrieval_agent],
        name="report_agent",
        description="Expert PHM research report writer.",
        instructions="Compose a clear PHM report summarising all findings.",
        return_full_result=True,
        add_base_tools=True,
    )
    return report_agent
