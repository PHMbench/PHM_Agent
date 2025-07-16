from __future__ import annotations

"""Simple retrieval tool based on a language model vector store."""

from typing import Any

from smolagents import Tool
from utils.registry import register_tool


@register_tool("RetrieverTool")
class RetrieverTool(Tool):
    """Retrieve relevant documents using semantic search."""

    name = "retriever"
    description = (
        "Uses semantic search to fetch documentation snippets relevant to a query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "Search query. Should be semantically related to the desired documents."
            ),
        }
    }
    output_type = "string"

    def __init__(self, vector_store: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )
