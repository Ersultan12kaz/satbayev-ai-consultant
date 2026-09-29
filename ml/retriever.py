"""Dense, keyword (BM25) and hybrid retrieval over the chunk index."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bm25 import BM25
from .embeddings import SubwordEncoder
from .lexicon import CONCEPT_STEMS, detect_concepts, expand_query


@dataclass
class Candidate:
    idx: int
    dense: float = 0.0
    bm25: float = 0.0
    dense_norm: float = 0.0
    bm25_norm: float = 0.0
    hybrid: float = 0.0
    dense_rank: int = 999
    bm25_rank: int = 999
    rerank: float = 0.0
    chunk: Dict = field(default_factory=dict)


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class Retriever:
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray, encoder: SubwordEncoder, bm25: BM25):
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.encoder = encoder
        self.bm25 = bm25

    # ------------------------------------------------------------ io
    @classmethod
    def load(cls, index_dir: str | Path, encoder: Optional[SubwordEncoder] = None) -> "Retriever":
        d = Path(index_dir)
        with open(d / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        emb = np.load(d / "embeddings.npy")
        enc = encoder or SubwordEncoder.load(d.parent / "embedder.pkl")
        return cls(chunks, emb, enc, BM25.load(d / "bm25.pkl"))

    def save(self, index_dir: str | Path) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(d / "embeddings.npy", self.embeddings)
        self.bm25.save(d / "bm25.pkl")

    @classmethod
    def build(cls, chunks: List[Dict], encoder: SubwordEncoder) -> "Retriever":
        texts = [f"{c.get('section', '')}. {c['text']}" for c in chunks]
        emb = encoder.encode(texts)
        bm = BM25().fit([f"{c['title']} {c.get('section', '')} {c['text']}" for c in chunks])
        return cls(chunks, emb, encoder, bm)

    # ------------------------------------------------------------ search
    def dense_scores(self, query: str, qvec: Optional[np.ndarray] = None) -> np.ndarray:
        q = self.encoder.encode([query])[0] if qvec is None else qvec
        return self.embeddings @ q

    def search(self, query: str, mode: str = "hybrid", k: int = 5, alpha: float = 0.7, lang: Optional[str] = None,
               lang_boost: float = 0.08, pool: int = 100, qvec: Optional[np.ndarray] = None,
               expand: bool = True, concept_boost: float = 0.25) -> List[Candidate]:
        """mode: 'dense' | 'bm25' | 'hybrid'.  Hybrid = alpha*dense + (1-alpha)*bm25 (min-max normalised).

        expand=True adds the official terms of colloquial words ("общага" -> "общежитие жатақхана dormitory"),
        see ml/lexicon.py.  A precomputed `qvec` must be the encoding of the expanded query.
        """
        n = len(self.chunks)
        if n == 0:
            return []
        concepts = detect_concepts(query) if expand else set()
        if expand:
            query = expand_query(query)
        if mode in ("dense", "hybrid"):
            dense = self.dense_scores(query) if qvec is None else self.dense_scores(query, qvec)
        else:
            dense = np.zeros(n)
        bm = self.bm25.scores(query) if mode in ("bm25", "hybrid") else np.zeros(n)
        pool = min(pool, n)
        cand = set(np.argsort(-dense)[:pool].tolist()) if mode != "bm25" else set()
        if mode != "dense":
            cand |= set(np.argsort(-bm)[:pool].tolist())
        cand = np.array(sorted(cand))
        d_c, b_c = dense[cand], bm[cand]
        dn, bn = _minmax(d_c), _minmax(b_c)
        if mode == "dense":
            score = d_c.copy()
        elif mode == "bm25":
            score = b_c.copy()
        else:
            score = alpha * dn + (1 - alpha) * bn
        if concepts and mode == "hybrid":
            # topic boost: chunks that talk about the concept of the question ("общага" -> dormitory) go first,
            # so generic words ("документы", "правила") do not pull unrelated regulations to the top
            hit = np.zeros(len(cand))
            for c in concepts:
                cols = [self.bm25.vocab[t] for t in CONCEPT_STEMS[c] if t in self.bm25.vocab]
                if cols:
                    hit += (np.asarray(self.bm25.matrix[cand][:, cols].sum(axis=1)).ravel() > 0)
            score = score + concept_boost * hit / len(concepts)
        if lang:
            same = np.array([self.chunks[i].get("lang") == lang for i in cand], dtype=np.float64)
            score = score + lang_boost * same * (1 if mode == "hybrid" else max(score.max(), 1e-6))
        d_rank = {int(i): r for r, i in enumerate(np.argsort(-dense)[:pool])}
        b_rank = {int(i): r for r, i in enumerate(np.argsort(-bm)[:pool])}
        order = np.argsort(-score)[:k]
        out = []
        for j in order:
            i = int(cand[j])
            out.append(Candidate(idx=i, dense=float(d_c[j]), bm25=float(b_c[j]), dense_norm=float(dn[j]),
                                 bm25_norm=float(bn[j]), hybrid=float(score[j]), dense_rank=d_rank.get(i, pool),
                                 bm25_rank=b_rank.get(i, pool), chunk=self.chunks[i]))
        return out

    # ------------------------------------------------------------ updates (admin)
    def add_chunks(self, new_chunks: List[Dict]) -> None:
        if not new_chunks:
            return
        emb = self.encoder.encode([f"{c.get('section', '')}. {c['text']}" for c in new_chunks])
        self.chunks = self.chunks + list(new_chunks)
        self.embeddings = np.vstack([self.embeddings, emb]) if len(self.embeddings) else emb
        self.bm25 = BM25().fit([f"{c['title']} {c.get('section', '')} {c['text']}" for c in self.chunks])

    def remove_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.embeddings = self.embeddings[keep]
            self.bm25 = BM25().fit([f"{c['title']} {c.get('section', '')} {c['text']}" for c in self.chunks])
        return removed
