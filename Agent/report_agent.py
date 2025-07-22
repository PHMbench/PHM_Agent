from __future__ import annotations

"""PHM research report writing agent."""

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
            return_full_result=True,
        )
        retrieval_agent = ToolCallingAgent(
            tools=[create_retriever_tool()],
            model=model,
            name="retrieval_agent",
            description="Retrieves knowledge base documents.",
            return_full_result=True,
            add_base_tools=True,
        )
        super().__init__(
            tools=[],
            model=model,
            managed_agents=[search_agent, retrieval_agent],
            name="report_agent",
            description="Expert agent that compiles PHM research reports.",
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )
