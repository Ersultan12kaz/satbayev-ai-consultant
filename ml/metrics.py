"""Evaluation metrics for retrieval and answers."""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Sequence

import numpy as np

from .text import stems


def relevance(retrieved_titles: Sequence[str], is_relevant: Callable[[str], bool]) -> List[int]:
    return [1 if is_relevant(t) else 0 for t in retrieved_titles]


def recall_at_k(rel: Sequence[int], k: int) -> float:
    """Hit-rate style Recall@K: 1 if at least one relevant document is in the top-K."""
    return 1.0 if any(rel[:k]) else 0.0


def precision_at_k(rel: Sequence[int], k: int) -> float:
    return sum(rel[:k]) / k if k else 0.0


def reciprocal_rank(rel: Sequence[int]) -> float:
    for i, r in enumerate(rel):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def token_f1(prediction: str, reference: str) -> float:
    p, r = stems(prediction), stems(reference)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(r)
    return 2 * prec * rec / (prec + rec)


def keyword_hit(answer: str, keywords: Sequence[str]) -> float:
    """Share of expected key words (stems) found in the answer."""
    if not keywords:
        return float("nan")
    a = set(stems(answer))
    hits = sum(1 for k in keywords if set(stems(k)) & a)
    return hits / len(keywords)


def latency_stats(ms: Sequence[float]) -> Dict[str, float]:
    if not ms:
        return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0}
    a = np.asarray(ms)
    return {"avg_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


def make_title_matcher(patterns: Sequence[str]) -> Callable[[str], bool]:
    regs = [re.compile(p, re.IGNORECASE) for p in patterns]
    return lambda title: any(r.search(title or "") for r in regs)
