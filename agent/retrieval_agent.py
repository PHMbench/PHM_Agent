from __future__ import annotations

"""Agent for querying the PHM knowledge base."""

from typing import Any

from smolagents.agents import ToolCallingAgent
from smolagents.models import Model

from PHM_tools.Retrieval import create_local_retriever_tool
from utils.registry import register_agent


@register_agent("RetrievalAgent")
class RetrievalAgent(ToolCallingAgent):
    """Simple retrieval agent using vector search."""

    def __init__(self, model: Model, kb_directory: str = "./kb", **kwargs: Any) -> None:
        tool = create_local_retriever_tool(kb_directory)
        instructions = "Use the retriever tool to fetch documents related to a query."
        super().__init__(
            tools=[tool],
            model=model,
            name="retrieval_agent",
            description="Queries the vector knowledge base",
            instructions=instructions,
            return_full_result=True,
            add_base_tools=True,
            **kwargs,
        )


if __name__ == "__main__":
    class DummyModel:
        def __call__(self, messages):
            return type("Obj", (), {"content": "ok"})()

    agent = RetrievalAgent(DummyModel(), kb_directory="./")
    print("Retriever tool name:", agent.tools[0].name)
