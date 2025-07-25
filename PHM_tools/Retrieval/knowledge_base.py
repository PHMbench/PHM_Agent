from __future__ import annotations

"""Vector knowledge base manager for storing PHM findings."""

from typing import Any, Dict, List

import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorKnowledgeBaseManager:
    """Manage and query vectorized findings."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.embedding_model = SentenceTransformer(embedding_model_name)
        dim = self.embedding_model.get_sentence_embedding_dimension()
        self.vector_index = faiss.IndexFlatL2(dim)
        self.findings_metadata: List[Dict[str, Any]] = []

    def add_finding(
        self, finding_text: str, source_agent: str, confidence: float, evidence_keys: List[str]
    ) -> str:
        """Store ``finding_text`` with metadata and return its ID."""
        finding_id = f"finding_{len(self.findings_metadata) + 1}_{int(datetime.datetime.now().timestamp())}"
        record = {
            "id": finding_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "text": finding_text,
            "source": source_agent,
            "confidence": confidence,
            "evidence": evidence_keys,
        }
        self.findings_metadata.append(record)
        vec = self.embedding_model.encode([finding_text]).astype("float32")
        self.vector_index.add(vec)
        return finding_id

    def query_similar_findings(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return findings most similar to ``query_text``."""
        if self.vector_index.ntotal == 0:
            return []
        q_vec = self.embedding_model.encode([query_text]).astype("float32")
        distances, indices = self.vector_index.search(q_vec, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            result = self.findings_metadata[idx].copy()
            result["similarity_score"] = 1 - float(distances[0][i])
            results.append(result)
        return results


if __name__ == "__main__":
    kb = VectorKnowledgeBaseManager()
    fid = kb.add_finding("demo finding", "demo_agent", 0.8, ["e1"])
    print("Added finding", fid)
    print("Query result:", kb.query_similar_findings("demo"))
