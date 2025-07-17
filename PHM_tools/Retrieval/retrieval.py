from __future__ import annotations

"""Retrieval-augmented tools for PHM agents."""

from typing import Iterable

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import Tool
from transformers import AutoTokenizer

from utils.registry import register_tool


@register_tool("RetrieverTool")
class RetrieverTool(Tool):
    """Semantic search over a vector store.

    Args:
        vector_store: Chroma database for document similarity search.
    """

    name = "retriever"
    description = (
        "Uses semantic search to retrieve documentation snippets relevant to a query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Query used for similarity search.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store: Chroma, **kwargs) -> None:
        """Initialize the tool.

        Args:
            vector_store: Prebuilt vector store used for similarity search.
        """
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n".join(doc.page_content for doc in docs)


def load_default_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """Load the demo knowledge base as a Chroma vector store.

    Args:
        persist_directory: Directory for persisting the vector store.

    Returns:
        Initialized Chroma instance.
    """
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed: list[Document] = []
    seen: set[str] = set()
    for doc in source_docs:
        for new_doc in splitter.split_documents([doc]):
            if new_doc.page_content not in seen:
                seen.add(new_doc.page_content)
                docs_processed.append(new_doc)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory=persist_directory)
    return vector_store


def create_retriever_tool(persist_directory: str = "./chroma_db") -> RetrieverTool:
    """Convenience function building a :class:`RetrieverTool`.

    Args:
        persist_directory: Directory for persisting the vector store.

    Returns:
        :class:`RetrieverTool` connected to the vector store.
    """
    store = load_default_vector_store(persist_directory)
    return RetrieverTool(store)
