"""Translation quality metrics implemented by us: BLEU (Papineni et al., 2002) and chrF (Popović, 2015)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence


def _words(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower())


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    match = [0] * max_n; total = [0] * max_n; hl = rl = 0
    for h, r in zip(hyps, refs):
        ht, rt = _words(h), _words(r)
        hl += len(ht); rl += len(rt)
        for n in range(1, max_n + 1):
            hn, rn = _ngrams(ht, n), _ngrams(rt, n)
            match[n - 1] += sum((hn & rn).values()); total[n - 1] += max(0, len(ht) - n + 1)
    if min(total) == 0 or min(match) == 0:
        return 0.0
    logp = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(1, hl))
    return 100 * bp * math.exp(logp)


def chrf(hyps: Sequence[str], refs: Sequence[str], n: int = 6, beta: float = 2.0) -> float:
    prec = rec = 0.0; cnt = 0
    for h, r in zip(hyps, refs):
        h = re.sub(r"\s+", "", h); r = re.sub(r"\s+", "", r)
        ps, rs = [], []
        for k in range(1, n + 1):
            hn, rn = _ngrams(list(h), k), _ngrams(list(r), k)
            m = sum((hn & rn).values())
            if sum(hn.values()):
                ps.append(m / sum(hn.values()))
            if sum(rn.values()):
                rs.append(m / sum(rn.values()))
        if ps and rs:
            prec += sum(ps) / len(ps); rec += sum(rs) / len(rs); cnt += 1
    if not cnt:
        return 0.0
    p, r = prec / cnt, rec / cnt
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
