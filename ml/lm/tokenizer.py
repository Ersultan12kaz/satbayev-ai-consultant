"""Sub-word tokenizer of our language model (byte-pair encoding, trained by us).

Same idea as ml/nmt/bpe.py, but made for a large corpus:
  * training counts words on a sample of the corpus and picks merges with a heap (fast for 32k merges);
  * newlines are kept (token "⏎"), so the model learns paragraphs;
  * special tokens for the chat format used in fine-tuning.
"""
from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>", "⏎",
            "<|question|>", "<|context|>", "<|answer|>", "<|kk|>", "<|ru|>", "<|en|>", "<|sep|>"]
PAD, BOS, EOS, UNK, NL = 0, 1, 2, 3, 4
Q_TOK, CTX_TOK, ANS_TOK = 5, 6, 7
LANG_TOK = {"kk": 8, "ru": 9, "en": 10}
SEP_TOK = 11
MARK = "▁"
_WORD_RE = re.compile(r"\n|\w+|[^\w\s]", re.UNICODE)


def pre_tokenize(text: str) -> List[str]:
    out = []
    for m in _WORD_RE.finditer(text):
        g = m.group(0)
        if g == "\n":
            out.append("\n")
            continue
        st = m.start()
        out.append((MARK + g) if (st == 0 or text[st - 1] in " \t") else g)
    return out


class LMTokenizer:
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.inv: List[str] = list(SPECIALS)
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.inv)}
        self._rank: Dict[Tuple[str, str], int] = {}
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------ training
    def train(self, texts: Iterable[str], vocab_size: int = 32000, min_freq: int = 3, verbose: bool = True):
        freq: Counter = Counter()
        for t in texts:
            freq.update(w for w in pre_tokenize(t) if w != "\n")
        words = [(tuple(w), c) for w, c in freq.items() if c >= min_freq]
        chars: Counter = Counter()
        for w, c in words:
            for ch in w:
                chars[ch] += c
        base = (MARK + "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі" + "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі".upper()
                + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper() + "0123456789"
                + ".,;:!?()[]{}-–—«»\"'/%№§+=*&#@$<>_|~^`…°•")
        alphabet = list(dict.fromkeys(list(base) + [ch for ch, c in chars.most_common() if c >= min_freq][:3000]))
        alpha = set(alphabet)
        symbols = [list(w) for w, _ in words]
        counts = [c for _, c in words]
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        where: Dict[Tuple[str, str], set] = defaultdict(set)
        for i, s in enumerate(symbols):
            for a, b in zip(s, s[1:]):
                if a in alpha and b in alpha:
                    pairs[(a, b)] += counts[i]
                    where[(a, b)].add(i)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)
        n_merges = vocab_size - len(SPECIALS) - len(alphabet)
        done = 0
        while done < n_merges and heap:
            negc, best = heapq.heappop(heap)
            if pairs.get(best, 0) != -negc:          # stale heap entry
                continue
            if -negc < min_freq:
                break
            a, b = best
            new = a + b
            self.merges.append(best)
            alpha.add(new)
            changed = set()
            for i in list(where.pop(best, ())):
                s, c = symbols[i], counts[i]
                for x, y in zip(s, s[1:]):
                    if (x, y) in pairs:
                        pairs[(x, y)] -= c
                        changed.add((x, y))
                        where[(x, y)].discard(i)
                j, merged = 0, []
                while j < len(s):
                    if j < len(s) - 1 and s[j] == a and s[j + 1] == b:
                        merged.append(new); j += 2
                    else:
                        merged.append(s[j]); j += 1
                symbols[i] = merged
                for x, y in zip(merged, merged[1:]):
                    if x in alpha and y in alpha:
                        pairs[(x, y)] += c
                        where[(x, y)].add(i)
                        changed.add((x, y))
            pairs.pop(best, None)
            for p in changed:
                c = pairs.get(p, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, p))
                else:
                    pairs.pop(p, None)
            done += 1
            if verbose and done % 2000 == 0:
                print(f"    tokenizer merges: {done}/{n_merges}", flush=True)
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
        ids = self._cache.get(w)
        if ids is not None:
            return ids
        s = list(w)
        while len(s) > 1:
            best, bi = 1 << 30, -1
            for i in range(len(s) - 1):
                r = self._rank.get((s[i], s[i + 1]), 1 << 30)
                if r < best:
                    best, bi = r, i
            if bi < 0:
                break
            s = s[:bi] + [s[bi] + s[bi + 1]] + s[bi + 2:]
        ids = [self.vocab.get(p, UNK) for p in s]
        if len(self._cache) < 2_000_000:
            self._cache[w] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for w in pre_tokenize(text):
            if w == "\n":
                out.append(NL)
            else:
                out.extend(self._word(w))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        parts = []
        for i in ids:
            if i == NL:
                parts.append("\n")
            elif len(SPECIALS) <= i < len(self.inv):
                parts.append(self.inv[i])
        text = "".join(parts).replace(MARK, " ")
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r" *\n *", "\n", text).strip()

    def __len__(self):
        return len(self.inv)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges, "vocab": self.inv}, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LMTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        t = cls()
        t.merges = [tuple(m) for m in d["merges"]]
        t.inv = d["vocab"]
        t.vocab = {x: i for i, x in enumerate(t.inv)}
        t._rank = {m: r for r, m in enumerate(t.merges)}
        return t
