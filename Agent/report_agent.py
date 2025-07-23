from __future__ import annotations

"""PHM research report writing agent.

This agent orchestrates web search and knowledge retrieval sub-agents to
gather information, analyse data processing pipelines and produce a concise
report. It can be extended to summarise code snippets and interact with other
agents in the system.
"""

from typing import Any

from smolagents.agents import CodeAgent, ToolCallingAgent, VisitWebpageTool, WebSearchTool
from smolagents.models import Model

from PHM_tools.Retrieval.retriever import create_retriever_tool
from utils.registry import register_agent


@register_agent("ReportAgent")
class ReportAgent(CodeAgent):
    """Agent specialized in writing PHM research reports."""

    def __init__(self, model: Model, **kwargs: Any) -> None:
        search_agent = ToolCallingAgent(
            tools=[WebSearchTool(), VisitWebpageTool()],
            model=model,
            name="research_agent",
            description="Performs web searches and gathers information.",
            instructions=(
                "You are a research assistant specialised in PHM. "
                "Search the web for relevant literature and data to support the analysis."
            ),
            return_full_result=True,
        )
        retrieval_agent = ToolCallingAgent(
            tools=[create_retriever_tool()],
            model=model,
            name="retrieval_agent",
            description="Retrieves knowledge base documents.",
            instructions="Use semantic search to find supporting documentation.",
            return_full_result=True,
            add_base_tools=True,
        )
        instructions = (
            "You are an expert PHM analyst tasked with writing concise "
            "technical reports. Combine information from web searches and "
            "the knowledge base to summarise findings and suggest next steps."
        )
        super().__init__(
            tools=[],
            model=model,
            managed_agents=[search_agent, retrieval_agent],
            name="report_agent",
            description="Expert agent that compiles PHM research reports.",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )
