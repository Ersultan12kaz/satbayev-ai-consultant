"""Byte-Pair Encoding tokenizer (our own implementation, Sennrich et al., 2016).

A joint sub-word vocabulary for Kazakh, Russian and English.  Words are split on
whitespace/punctuation; the first piece of every word carries the "▁" marker so the
text can be restored exactly.  Training uses incremental pair statistics, so
16 000 merges over a few hundred thousand distinct words take a few minutes.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "<2kk>", "<2ru>", "<2en>"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3
LANG_TOKEN = {"kk": 4, "ru": 5, "en": 6}
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
MARK = "▁"


def pre_tokenize(text: str) -> List[str]:
    """'Оқу ақысы, 2024 ж.' -> ['▁Оқу', '▁ақысы', ',', '▁2024', '▁ж', '.']  (▁ = a space before the token)."""
    out = []
    for m in _WORD_RE.finditer(text):
        st = m.start()
        out.append((MARK + m.group(0)) if (st == 0 or text[st - 1].isspace()) else m.group(0))
    return out


class BPE:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inv: List[str] = []
        self._rank: Dict[Tuple[str, str], int] = {}
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------ training
    def train(self, texts: Iterable[str], vocab_size: int = 16000, min_freq: int = 2, verbose: bool = True) -> "BPE":
        freq: Counter = Counter()
        for t in texts:
            freq.update(pre_tokenize(t))
        words = [(tuple(w), c) for w, c in freq.items() if c >= min_freq or len(w) <= 3]
        chars = Counter()
        for w, c in words:
            for ch in w:
                chars[ch] += c
        alphabet = [ch for ch, c in chars.most_common() if c >= min_freq]
        symbols = [list(w) for w, _ in words]
        counts = [c for _, c in words]
        pairs: Counter = Counter()
        where: Dict[Tuple[str, str], set] = defaultdict(set)
        for i, s in enumerate(symbols):
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += counts[i]
                where[(a, b)].add(i)
        n_merges = vocab_size - len(SPECIALS) - len(alphabet)
        for step in range(max(0, n_merges)):
            if not pairs:
                break
            best, bc = max(pairs.items(), key=lambda kv: kv[1])
            if bc < min_freq:
                break
            a, b = best
            new = a + b
            self.merges.append(best)
            for i in list(where.pop(best, ())):
                s = symbols[i]
                c = counts[i]
                for x, y in zip(s, s[1:]):                   # remove old pairs of this word
                    pairs[(x, y)] -= c
                    if pairs[(x, y)] <= 0:
                        del pairs[(x, y)]
                    where[(x, y)].discard(i)
                j, merged = 0, []
                while j < len(s):
                    if j < len(s) - 1 and s[j] == a and s[j + 1] == b:
                        merged.append(new); j += 2
                    else:
                        merged.append(s[j]); j += 1
                symbols[i] = merged
                for x, y in zip(merged, merged[1:]):          # add new pairs
                    pairs[(x, y)] += c
                    where[(x, y)].add(i)
            pairs.pop(best, None)
            if verbose and (step + 1) % 2000 == 0:
                print(f"    BPE merges: {step + 1}/{n_merges}", flush=True)
        self._build(alphabet)
        return self

    def _build(self, alphabet: Sequence[str]):
        toks = list(SPECIALS) + list(alphabet) + [a + b for a, b in self.merges]
        seen, self.inv = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); self.inv.append(t)
        self.vocab = {t: i for i, t in enumerate(self.inv)}
        self._rank = {m: r for r, m in enumerate(self.merges)}
        self._cache = {}

    # ------------------------------------------------------------ encode / decode
    def _word(self, w: str) -> List[int]:
        if w in self._cache:
            return self._cache[w]
        s = list(w)
        while len(s) > 1:
            ranked = [(self._rank.get((s[i], s[i + 1]), 1 << 30), i) for i in range(len(s) - 1)]
            r, i = min(ranked)
            if r == 1 << 30:
                break
            s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        ids = [self.vocab.get(p, UNK) for p in s]
        if len(self._cache) < 500_000:
            self._cache[w] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for w in pre_tokenize(text):
            out.extend(self._word(w))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        pieces = [self.inv[i] for i in ids if i >= len(SPECIALS) and i < len(self.inv)]
        text = "".join(pieces).replace(MARK, " ").strip()
        return re.sub(r"\s+", " ", text)

    def __len__(self):
        return len(self.inv)

    # ------------------------------------------------------------ io
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges, "vocab": self.inv}, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        b = cls()
        b.merges = [tuple(m) for m in d["merges"]]
        b.inv = d["vocab"]
        b.vocab = {t: i for i, t in enumerate(b.inv)}
        b._rank = {m: r for r, m in enumerate(b.merges)}
        return b
