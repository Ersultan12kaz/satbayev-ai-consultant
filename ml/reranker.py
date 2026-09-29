"""Learning-to-rank reranker.

For every (question, candidate chunk) pair we compute interpretable features
(dense similarity, BM25, ranks, lexical overlap, heading overlap, language
match, ...) and train a logistic-regression model (own gradient-descent
implementation) that predicts P(chunk answers the question).  The probability
is also used as the answer confidence and for the "information not found"
decision (hallucination guard).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .retriever import Candidate
from .lexicon import query_stem_groups, support
from .text import content_tokens, stems

FEATURES = ["dense", "dense_norm", "bm25_norm", "hybrid", "inv_dense_rank", "inv_bm25_rank", "overlap",
            "bigram_overlap", "heading_overlap", "title_overlap", "lang_match", "log_len", "bias"]


def _bigrams(tokens: List[str]) -> set:
    return set(zip(tokens, tokens[1:]))


def features(query: str, cands: Sequence[Candidate], lang: Optional[str] = None) -> np.ndarray:
    qs = stems(query)
    qset = set(qs)
    groups = query_stem_groups(query)      # colloquial words also match their official synonyms
    qbi = _bigrams(qs)
    rows = []
    for c in cands:
        ch = c.chunk
        cs = stems(ch["text"])
        cset = set(cs)
        hs = set(stems(ch.get("section", "")))
        ts = set(stems(ch.get("title", "")))
        ov = support(groups, cset)
        bov = len(qbi & _bigrams(cs)) / len(qbi) if qbi else 0.0
        hov = support(groups, hs)
        tov = support(groups, ts)
        rows.append([c.dense, c.dense_norm, c.bm25_norm, c.hybrid, 1.0 / (1 + c.dense_rank),
                     1.0 / (1 + c.bm25_rank), ov, bov, hov, tov,
                     1.0 if (lang and ch.get("lang") == lang) else 0.0,
                     math.log1p(ch.get("n_words", len(cs))) / 6.0, 1.0])
    return np.asarray(rows, dtype=np.float64)


class Reranker:
    def __init__(self):
        self.w = np.zeros(len(FEATURES))
        self.mu = np.zeros(len(FEATURES))
        self.sd = np.ones(len(FEATURES))
        self.history: List[Dict] = []
        self.trained = False

    def _z(self, X):
        Z = (X - self.mu) / self.sd
        Z[:, -1] = 1.0
        return Z

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 0.1, l2: float = 1e-3,
            verbose: bool = True) -> "Reranker":
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        self.mu[-1], self.sd[-1] = 0.0, 1.0
        Z = self._z(X)
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1)
        sw = np.where(y == 1, pos_w, 1.0)
        sw = sw / sw.mean()
        self.w = np.zeros(Z.shape[1])
        for ep in range(1, epochs + 1):
            p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
            g = Z.T @ ((p - y) * sw) / len(y) + l2 * self.w
            self.w -= lr * g
            if ep % 25 == 0 or ep == 1:
                pc = np.clip(p, 1e-9, 1 - 1e-9)
                loss = float(-(sw * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).mean())
                acc = float(((p >= 0.5) == y).mean())
                self.history.append({"epoch": ep, "loss": loss, "acc": acc})
                if verbose and (ep % 100 == 0 or ep == 1):
                    print(f"  epoch {ep:4d} | weighted_logloss={loss:.4f} | acc={acc:.4f}")
        self.trained = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return 1 / (1 + np.exp(-np.clip(self._z(X) @ self.w, -30, 30)))

    def rerank(self, query: str, cands: List[Candidate], lang: Optional[str] = None) -> List[Candidate]:
        if not cands:
            return cands
        if not self.trained:
            for c in cands:
                c.rerank = c.hybrid
            return cands
        p = self.predict_proba(features(query, cands, lang))
        for c, s in zip(cands, p):
            c.rerank = float(s)
        return sorted(cands, key=lambda c: -c.rerank)

    def weights(self) -> Dict[str, float]:
        return {f: float(w) for f, w in zip(FEATURES, self.w)}

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                       "features": FEATURES, "history": self.history}, f, indent=1)

    @classmethod
    def load(cls, path: str | Path) -> "Reranker":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        r = cls()
        r.w, r.mu, r.sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
        r.history = d.get("history", [])
        r.trained = True
        return r
