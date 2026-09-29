"""Embedding service: wraps our own trained SubwordEncoder (ml/embeddings.py)."""
from typing import List, Sequence

from ml.embeddings import SubwordEncoder


class EmbeddingService:
    def __init__(self, encoder: SubwordEncoder):
        self.encoder = encoder

    @property
    def dim(self) -> int:
        return self.encoder.dim

    def embed_query(self, text: str) -> List[float]:
        return self.encoder.encode([text])[0].tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.encoder.encode(list(texts)).tolist()
