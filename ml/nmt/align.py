"""Parallel sentences from the university's own trilingual documents (our own aligner).

Many regulations are published in Kazakh, Russian and English.  We
1. pair documents that are translations of each other: same site section, similar
   sequence of numbers (clause numbers "4.1", dates, years, amounts) and similar length;
2. align their sentences with a length-based dynamic programme (Gale & Church, 1993)
   extended with a number-agreement score (numbers are copied unchanged by translators),
   allowing 1-1, 1-2, 2-1, 2-2 and skip beads, inside a band around the diagonal;
3. keep only confident pairs (same numbers, plausible length ratio, normal sentence length).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

from ..text import split_sentences, tokenize

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_YEAR_RE = re.compile(r"(19|20)\d\d")


def numbers(text: str) -> List[str]:
    return [n.replace(",", ".").rstrip(".") for n in _NUM_RE.findall(text)]


def doc_signature(text: str) -> Counter:
    return Counter(n for n in numbers(text) if len(n) >= 2 or "." in n)


def pair_documents(docs: Dict[str, Dict], min_score: float = 0.3) -> List[Tuple[str, str, float]]:
    """docs: doc_id -> {"lang", "category", "title", "text"}.  Returns (id_a, id_b, score) for translations."""
    sig = {d: doc_signature(v["text"]) for d, v in docs.items()}
    size = {d: max(1, len(v["text"])) for d, v in docs.items()}
    by_cat = defaultdict(list)
    for d, v in docs.items():
        by_cat[v.get("category", "")].append(d)
    cands = []
    langs = ("ru", "kk", "en")
    for cat, ids in by_cat.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                la, lb = docs[a]["lang"], docs[b]["lang"]
                if la == lb or la not in langs or lb not in langs:
                    continue
                sa, sb = sig[a], sig[b]
                if not sa or not sb:
                    continue
                inter = sum((sa & sb).values()); union = sum((sa | sb).values())
                jac = inter / max(1, union)
                ratio = min(size[a], size[b]) / max(size[a], size[b])
                ya = set(_YEAR_RE.findall(docs[a]["title"])); yb = set(_YEAR_RE.findall(docs[b]["title"]))
                year = 0.1 if (ya and ya == yb) else (-0.1 if (ya and yb and ya != yb) else 0.0)
                score = jac * (0.5 + 0.5 * ratio) + year
                if score >= min_score:
                    cands.append((score, a, b))
    cands.sort(reverse=True)
    used = set(); out = []
    for score, a, b in cands:                      # one partner per language for every document
        ka = (a, docs[b]["lang"]); kb = (b, docs[a]["lang"])
        if ka in used or kb in used:
            continue
        used.add(ka); used.add(kb)
        out.append((a, b, round(score, 3)))
    return out


# ------------------------------------------------------------------ sentence alignment
_BEADS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (0, 1)]
_PRIOR = {(1, 1): 0.0, (1, 2): 2.0, (2, 1): 2.0, (2, 2): 3.5, (1, 0): 5.0, (0, 1): 5.0}


def align_sentences(src: Sequence[str], tgt: Sequence[str], band: int = 40) -> List[Tuple[str, str, float]]:
    n, m = len(src), len(tgt)
    if not n or not m:
        return []
    ls = [len(s) for s in src]; lt = [len(t) for t in tgt]
    c = sum(lt) / max(1, sum(ls))                  # average length ratio of this document pair
    ns = [set(numbers(s)) for s in src]; nt = [set(numbers(t)) for t in tgt]
    INF = float("inf")
    cost = {(0, 0): (0.0, None)}

    def bead_cost(i, j, di, dj):
        if di == 0 or dj == 0:
            return _PRIOR[(di, dj)]
        a = sum(ls[i - di:i]); b = sum(lt[j - dj:j])
        z = (b - c * a) / math.sqrt(max(1.0, 6.8 * (a + b) / 2))
        pen = z * z / 2 + _PRIOR[(di, dj)]
        na = set().union(*ns[i - di:i]); nb = set().union(*nt[j - dj:j])
        if na or nb:
            pen += -2.0 if na == nb else 3.0
        return pen

    for i in range(n + 1):
        center = i * m / n
        lo, hi = max(0, int(center - band)), min(m, int(center + band))
        for j in range(lo, hi + 1):
            if (i, j) == (0, 0):
                continue
            best = (INF, None)
            for di, dj in _BEADS:
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0 or (pi, pj) not in cost:
                    continue
                v = cost[(pi, pj)][0] + bead_cost(i, j, di, dj)
                if v < best[0]:
                    best = (v, (di, dj))
            if best[1] is not None:
                cost[(i, j)] = best
    if (n, m) not in cost:
        return []
    out, i, j = [], n, m
    while (i, j) != (0, 0):
        v, (di, dj) = cost[(i, j)]
        pv = cost[(i - di, j - dj)][0]
        if di and dj:
            out.append((" ".join(src[i - di:i]), " ".join(tgt[j - dj:j]), v - pv))
        i, j = i - di, j - dj
    return out[::-1]


def good_pair(a: str, b: str, bead_cost: float) -> bool:
    ta, tb = len(tokenize(a)), len(tokenize(b))
    if not (4 <= ta <= 80 and 4 <= tb <= 80):
        return False
    r = len(a) / max(1, len(b))
    if not (0.5 <= r <= 2.0):
        return False
    if set(numbers(a)) != set(numbers(b)):
        return False
    if re.search(r"\.{4,}|_{4,}", a + b):
        return False
    return bead_cost < 4.0


def parallel_from_documents(docs: Dict[str, Dict], verbose: bool = True) -> List[Dict]:
    """-> [{"src_lang", "tgt_lang", "src", "tgt", "doc_a", "doc_b"}]"""
    pairs = pair_documents(docs)
    if verbose:
        print(f"  document pairs (translations of each other): {len(pairs)}")
    out = []
    for a, b, score in pairs:
        sa = [s for s in split_sentences(docs[a]["text"]) if len(s) > 3]
        sb = [s for s in split_sentences(docs[b]["text"]) if len(s) > 3]
        band = max(40, int(abs(len(sa) - len(sb)) * 1.2) + 20)
        for x, y, cst in align_sentences(sa, sb, band=band):
            if good_pair(x, y, cst):
                out.append({"src_lang": docs[a]["lang"], "tgt_lang": docs[b]["lang"], "src": x, "tgt": y,
                            "doc_a": a, "doc_b": b})
    return out
