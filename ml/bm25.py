"""Okapi BM25 keyword retriever implemented from scratch (sparse matrices)."""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse

from .text import stems


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.matrix: sparse.csr_matrix | None = None  # docs x terms, BM25 weights

    def fit(self, texts: Sequence[str]) -> "BM25":
        tokenized = [stems(t) for t in texts]
        df: Counter = Counter()
        for toks in tokenized:
            df.update(set(toks))
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(tokenized)
        self.idf = np.array([np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in sorted(df)])
        lengths = np.array([len(t) for t in tokenized], dtype=np.float64)
        avgdl = lengths.mean() if n else 1.0
        rows, cols, vals = [], [], []
        for i, toks in enumerate(tokenized):
            tf = Counter(toks)
            norm = self.k1 * (1 - self.b + self.b * lengths[i] / avgdl)
            for t, c in tf.items():
                j = self.vocab[t]
                rows.append(i); cols.append(j)
                vals.append(self.idf[j] * c * (self.k1 + 1) / (c + norm))
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)))
        return self

    def query_vector(self, query: str) -> sparse.csr_matrix:
        q = Counter(t for t in stems(query) if t in self.vocab)
        cols = [self.vocab[t] for t in q]
        vals = [1.0 for _ in q]  # binary query term weights
        return sparse.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.vocab)))

    def scores(self, query: str) -> np.ndarray:
        qv = self.query_vector(query)
        if qv.nnz == 0:
            return np.zeros(self.matrix.shape[0])
        return np.asarray((self.matrix @ qv.T).todense()).ravel()

    def search(self, query: str, k: int = 10) -> List[tuple]:
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "BM25":
        with open(path, "rb") as f:
            return pickle.load(f)
