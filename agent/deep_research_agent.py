from __future__ import annotations

"""DeepResearchAgent creation utilities."""

import os
from typing import List

from smolagents import GoogleSearchTool, ToolCallingAgent
from smolagents.models import LiteLLMModel

from PHM_tools.text_web_browser import (
    SimpleTextBrowser,
    VisitTool,
    PageUpTool,
    PageDownTool,
    FinderTool,
    FindNextTool,
    ArchiveSearchTool,
)
from PHM_tools.text_inspector_tool import TextInspectorTool


SYSTEM_PROMPT = (
    """A highly specialized team member for conducting deep-dive technical research.
Deploy this agent for any task that requires searching the web, analyzing technical papers, or summarizing complex documents.
Provide it with a clear, detailed research question, not just keywords.
Example: 'Find recent studies comparing the effectiveness of CWT and STFT for diagnosing bearing faults in high-speed machinery.'
It will autonomously browse, read, and synthesize information, returning a structured summary with sources."""
)

TASK_INSTRUCTIONS = (
    """Your Persona: You are DeepResearch, an advanced research analyst. Your expertise lies in navigating the web to find and analyze technical information for Predictive Health Management (PHM). You are methodical, precise, and always cite your sources.

Your Mission: Given a research question, you must thoroughly investigate it using the provided web Browse and inspection tools. Your goal is to gather data, identify key findings, and compile a structured, evidence-based summary.

Your Workflow:
1. **Deconstruct & Search**: Break down the user's question into searchable queries. Use `GoogleSearchTool` to find relevant articles, technical papers (.pdf), and documentation.
2. **Visit & Inspect**: Use `VisitTool` to access web pages. For long pages, navigate with `PageUpTool` and `PageDownTool`. Use `FinderTool` to locate specific keywords.
3. **Analyze Content**: If you encounter a link to a file (like a .pdf or .txt), use `TextInspectorTool` to extract its text content for analysis. This is your primary method for reading documents.
4. **Synthesize & Summarize**: As you gather information, continuously build a summary of your findings.
5. **Final Output**: Your final response should be a well-structured summary of the findings, complete with URLs as citations for all claims. If you cannot find a definitive answer, state what you found and why the information is inconclusive. If you need more information from the user, use `final_answer` to ask a clarifying question."""
)


def _build_browser() -> SimpleTextBrowser:
    """Return a configured :class:`SimpleTextBrowser`."""

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
    return SimpleTextBrowser(
        viewport_size=1024 * 5,
        downloads_folder="downloads_folder",
        serpapi_key=os.getenv("SERPAPI_API_KEY"),
        request_kwargs={"headers": {"User-Agent": user_agent}, "timeout": 300},
    )


def _build_tools(browser: SimpleTextBrowser, model: LiteLLMModel) -> List:
    """Return the list of tools available to the agent."""

    return [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit=100000),
    ]


def create_deep_research_agent(model: LiteLLMModel) -> ToolCallingAgent:
    """Build and return the Deep Research agent."""

    browser = _build_browser()
    tools = _build_tools(browser, model)

    agent = ToolCallingAgent(
        model=model,
        tools=tools,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="deep_research_agent",
        description="Agent that performs in-depth PHM research on the web.",
        instructions=SYSTEM_PROMPT,
        provide_run_summary=True,
    )
    agent.prompt_templates["managed_agent"]["task"] += (
        TASK_INSTRUCTIONS
        + "\nYou can navigate to .txt online files. If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it."
        + " Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request more information."
    )
    return agent

__all__ = ["create_deep_research_agent"]


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = create_deep_research_agent(DummyModel())
    print("DeepResearch agent tools:", [t.name for t in agent.tools[:3]], "...")
