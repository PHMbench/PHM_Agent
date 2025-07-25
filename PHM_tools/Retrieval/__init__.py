from .retriever import build_vector_store, create_retriever_tool, RetrieverTool
from .local_knowledge import build_local_vector_store, create_local_retriever_tool
from .knowledge_base import VectorKnowledgeBaseManager

__all__ = [
    "build_vector_store",
    "create_retriever_tool",
    "RetrieverTool",
    "build_local_vector_store",
    "create_local_retriever_tool",
    "VectorKnowledgeBaseManager",
]


if __name__ == "__main__":
    from langchain.docstore.document import Document
    from langchain.vectorstores import Chroma

    class RandomEmbeddings:
        def embed_documents(self, docs):
            import numpy as np

            return [np.random.random(8).tolist() for _ in docs]

        def embed_query(self, _q):
            import numpy as np

            return np.random.random(8).tolist()

    docs = [Document(page_content=f"doc {i}") for i in range(3)]
    store = Chroma.from_documents(docs, RandomEmbeddings())
    tool = RetrieverTool(store)
    print(tool("doc"))
