from __future__ import annotations

"""Retriever tool built on Chroma and LangChain utilities."""

from typing import List

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer
from smolagents import Tool

from utils.registry import register_tool


# ---------------------------------------------------------------------------
# Vector store creation
# ---------------------------------------------------------------------------


def build_vector_store(persist_directory: str = "./chroma_db") -> Chroma:
    """Load documentation dataset and build a Chroma vector store."""
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed: List[Document] = []
    unique_texts = {}
    for doc in tqdm(source_docs, desc="Splitting documents"):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory=persist_directory)
    return vector_store


# ---------------------------------------------------------------------------
# Retriever tool
# ---------------------------------------------------------------------------


@register_tool("RetrieverTool")
class RetrieverTool(Tool):
    """Semantic search tool over a documentation corpus."""

    name = "retriever"
    description = (
        "Uses semantic search to retrieve documentation passages relevant to a query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. Use an affirmative statement rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store: Chroma, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )
