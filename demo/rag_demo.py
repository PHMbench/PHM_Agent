"""Minimal RAG example using the RetrieverTool."""

from __future__ import annotations

import os
from pathlib import Path

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer

from smolagents import CodeAgent, LiteLLMModel

from PHM_tools.Retrieval import RetrieverTool


DATASET_NAME = "m-ric/huggingface_doc"


def build_vector_store() -> Chroma:
    """Download and embed docs into a Chroma vector store."""
    knowledge_base = datasets.load_dataset(DATASET_NAME, split="train")
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=200,
        chunk_overlap=20,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs, desc="Splitting"):
        for new_doc in splitter.split_documents([doc]):
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store_dir = Path("./chroma_db")
    vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory=str(store_dir))
    return vector_store


def main() -> None:
    vector_store = build_vector_store()
    retriever_tool = RetrieverTool(vector_store)
    model = LiteLLMModel(model_id="gpt-4o")
    agent = CodeAgent(tools=[retriever_tool], model=model, max_steps=4, verbosity_level=2, stream_outputs=True)
    result = agent.run("How can I push a model to the Hub?")
    print("Final output:")
    print(result)


if __name__ == "__main__":
    main()
