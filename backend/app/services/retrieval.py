"""Retrieval service.

* numpy backend (default): vectors are kept in memory (models/index/embeddings.npy).
* pgvector backend: the dense search is executed inside PostgreSQL with the
  cosine-distance operator  embedding <=> :query  (see docker-compose.yml).
"""
from typing import Optional

import numpy as np
from sqlalchemy import text

from ml.retriever import Retriever

from ..core.database import SessionLocal


class PgVectorRetriever(Retriever):
    """Same API as ml.retriever.Retriever, dense part served by pgvector."""

    def __init__(self, base: Retriever, candidates: int = 200):
        super().__init__(base.chunks, base.embeddings, base.encoder, base.bm25)
        self.candidates = candidates
        self._pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def dense_scores(self, query: str, qvec=None) -> np.ndarray:
        q = (self.encoder.encode([query])[0] if qvec is None else qvec).tolist()
        scores = np.full(len(self.chunks), -1.0, dtype=np.float32)
        sql = text("SELECT chunk_id, 1 - (embedding <=> CAST(:q AS vector)) AS sim FROM document_chunks "
                   "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k")
        with SessionLocal() as db:
            for cid, sim in db.execute(sql, {"q": str(q), "k": self.candidates}):
                i = self._pos.get(cid)
                if i is not None:
                    scores[i] = float(sim)
        return scores

    def add_chunks(self, new_chunks):
        super().add_chunks(new_chunks)
        self._pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

    def remove_document(self, doc_id):
        n = super().remove_document(doc_id)
        self._pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        return n


def make_retriever(base: Retriever, backend: str) -> Retriever:
    return PgVectorRetriever(base) if backend == "pgvector" else base
