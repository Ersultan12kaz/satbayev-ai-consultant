"""Automatic construction of training pairs from the document chunks.

* Inverse Cloze Task (ICT, Lee et al., 2019): a sentence taken from a chunk is a
  pseudo-question; the chunk (with that sentence removed most of the time) is
  the positive passage.
* Heading pairs: a section heading / document title is a query for the chunk.
* Question-style pairs: a heading is wrapped into a natural question template
  in the language of the chunk ("Каков порядок ...?", "... қандай?", "What is ...?").
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .text import split_sentences, tokenize

Q_TEMPLATES = {
    "ru": ["Что говорится о {h}?", "Каков порядок: {h}?", "{h} - какие правила?", "Расскажи: {h}"],
    "kk": ["{h} туралы не айтылған?", "{h} тәртібі қандай?", "{h} қандай?", "{h} туралы айтып бер"],
    "en": ["What does the document say about {h}?", "What is the procedure: {h}?", "{h} - what are the rules?",
           "Tell me about {h}"],
}


def _clean_heading(h: str) -> str:
    h = h.strip().strip(".:")
    h = h.lstrip("0123456789. ")
    return h[:1].lower() + h[1:] if h and not h.isupper() else h.capitalize()


def make_pairs(chunks: Sequence[Dict], max_pairs: int = 40000, seed: int = 7,
               remove_prob: float = 0.9) -> List[Tuple[str, str, int]]:
    """Returns (query, passage_text, chunk_index)."""
    rnd = random.Random(seed)
    pairs: List[Tuple[str, str, int]] = []
    for i, ch in enumerate(chunks):
        sents = [s for s in split_sentences(ch["text"]) if 6 <= len(tokenize(s)) <= 45]
        if len(sents) >= 2:
            s = rnd.choice(sents)
            passage = ch["text"].replace(s, " ") if rnd.random() < remove_prob else ch["text"]
            pairs.append((s, passage, i))
        sec = ch.get("section") or ""
        if sec and 2 <= len(tokenize(sec)) <= 15:
            h = _clean_heading(sec)
            tpl = rnd.choice(Q_TEMPLATES.get(ch.get("lang", "ru"), Q_TEMPLATES["ru"]))
            pairs.append((tpl.format(h=h), ch["text"], i))
    rnd.shuffle(pairs)
    return pairs[:max_pairs]


def split_pairs(pairs, val_ratio: float = 0.1, max_val: int = 1500, seed: int = 3):
    """Split by chunk so that no validation chunk is seen in training."""
    rnd = random.Random(seed)
    chunk_ids = sorted({p[2] for p in pairs})
    rnd.shuffle(chunk_ids)
    n_val = min(max_val, int(len(chunk_ids) * val_ratio))
    val_ids = set(chunk_ids[:n_val])
    train = [p for p in pairs if p[2] not in val_ids]
    val, seen = [], set()
    for p in pairs:
        if p[2] in val_ids and p[2] not in seen:  # one pair per chunk -> unique positives
            val.append(p); seen.add(p[2])
    return train, val
