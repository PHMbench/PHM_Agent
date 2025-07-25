from __future__ import annotations

"""Utilities for building a vector store from local documents."""

from pathlib import Path
from typing import Iterable, Sequence

from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from .retriever import RetrieverTool

__all__ = ["build_local_vector_store", "create_local_retriever_tool"]


def _load_text_from_file(path: Path) -> str:
    """Return the textual content of a file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _gather_documents(directory: Path) -> list[Document]:
    """Load supported documents from ``directory``."""
    docs: list[Document] = []
    for file in directory.rglob("*"):
        if file.suffix.lower() not in {".pdf", ".md"}:
            continue
        text = _load_text_from_file(file)
        docs.append(Document(page_content=text, metadata={"source": file.name}))
    return docs


def _split_documents(docs: Sequence[Document]) -> list[Document]:
    """Split ``docs`` into non-overlapping chunks."""
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
    )
    unique: set[str] = set()
    processed: list[Document] = []
    for doc in docs:
        for chunk in splitter.split_documents([doc]):
            if chunk.page_content not in unique:
                unique.add(chunk.page_content)
                processed.append(chunk)
    return processed


def build_local_vector_store(
    directory: str, persist_directory: str = "./local_chroma_db"
) -> Chroma:
    """Create a Chroma vector store from local PDF and Markdown files.

    Args:
        directory: Folder containing documents.
        persist_directory: Directory used to store the vector database.

    Returns:
        Initialized Chroma instance populated with the documents.
    """
    docs = _gather_documents(Path(directory))
    processed = _split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    store = Chroma.from_documents(
        processed, embeddings, persist_directory=persist_directory
    )
    return store


def create_local_retriever_tool(
    directory: str, persist_directory: str = "./local_chroma_db"
) -> RetrieverTool:
    """Build a :class:`RetrieverTool` for documents under ``directory``.

    Args:
        directory: Folder containing the knowledge documents.
        persist_directory: Directory used to store the vector database.

    Returns:
        Configured :class:`RetrieverTool` instance.
    """
    store = build_local_vector_store(directory, persist_directory=persist_directory)
    return RetrieverTool(store)


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.txt"
        path.write_text("simple demo document")
        store = build_local_vector_store(tmp, persist_directory=tmp)
        tool = RetrieverTool(store)
        print(tool("demo"))
