"""Grounded answer generation with citations.

Baseline generator ("extractive"): selects the most relevant sentences from the
top reranked chunks and composes an answer in the language of the question.
Because every sentence is copied from an official document, the answer is
100 % faithful by construction and every statement carries a citation [n].
The main generator is ml/composer.py (question type + trained sentence ranker).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .lexicon import expand_query, query_stem_groups, support
from .retriever import Candidate
from .text import split_sentences, stems, tokenize

MESSAGES = {
    "not_found": {
        "ru": "В официальных документах университета информация по этому вопросу не найдена. "
              "Попробуйте переформулировать вопрос или обратитесь в Центр обслуживания студентов.",
        "kk": "Университеттің ресми құжаттарынан бұл сұрақ бойынша ақпарат табылмады. "
              "Сұрақты басқаша қойып көріңіз немесе Студенттерге қызмет көрсету орталығына хабарласыңыз.",
        "en": "I couldn't find this information in the official university documents. "
              "Please rephrase the question or contact the Student Service Center.",
    },
    "out_of_scope": {
        "ru": "Я являюсь виртуальным консультантом Satbayev University и отвечаю только на вопросы, "
              "связанные с университетом: обучение, экзамены, стипендии, общежитие, практика, документы и т.д.",
        "kk": "Мен Satbayev University-дің виртуалды кеңесшісімін және тек университетке қатысты сұрақтарға "
              "жауап беремін: оқу, емтихандар, шәкіртақы, жатақхана, практика, құжаттар және т.б.",
        "en": "I am the Satbayev University virtual consultant and I answer only university-related questions: "
              "studies, exams, scholarships, dormitories, internships, documents, etc.",
    },
    "intro": {
        "ru": "Согласно официальным документам Satbayev University:",
        "kk": "Satbayev University-дің ресми құжаттарына сәйкес:",
        "en": "According to the official documents of Satbayev University:",
    },
    "other_lang": {
        "ru": "(Фрагменты приведены на языке оригинала документа.)",
        "kk": "(Үзінділер құжаттың түпнұсқа тілінде берілген.)",
        "en": "(Fragments are quoted in the original language of the document.)",
    },
    "confidence": {
        "ru": {"high": "высокая", "medium": "средняя", "low": "низкая"},
        "kk": {"high": "жоғары", "medium": "орташа", "low": "төмен"},
        "en": {"high": "high", "medium": "medium", "low": "low"},
    },
    "page": {"ru": "стр.", "kk": "бет", "en": "p."},
}

@dataclass
class Answer:
    text: str
    lang: str
    sources: List[Dict] = field(default_factory=list)
    confidence: str = "low"
    confidence_score: float = 0.0
    status: str = "answered"          # answered | not_found | out_of_scope
    generator: str = "extractive"
    used_sentences: List[Dict] = field(default_factory=list)


def confidence_label(p: float) -> str:
    return "high" if p >= 0.75 else "medium" if p >= 0.45 else "low"


def _source(c: Candidate, n: int) -> Dict:
    ch = c.chunk
    return {"n": n, "doc_id": ch["doc_id"], "title": ch["title"], "page": ch["page"], "url": ch.get("url", ""),
            "lang": ch.get("lang", ""), "section": ch.get("section", ""), "chunk_id": ch.get("chunk_id"),
            "score": round(c.rerank, 4)}


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, encoder=None, max_sentences: int = 4, max_sources: int = 3):
        self.encoder = encoder
        self.max_sentences = max_sentences
        self.max_sources = max_sources

    def generate(self, query: str, lang: str, cands: Sequence[Candidate]) -> Answer:
        top = cands[0].rerank
        evidence = [c for c in cands if c.rerank >= max(0.5 * top, 0.15)][: self.max_sources] or list(cands[:1])
        groups = query_stem_groups(query)
        sent_rows = []
        for ci, c in enumerate(evidence):
            for si, s in enumerate(split_sentences(c.chunk["text"])):
                ntok = len(tokenize(s))
                if ntok < 6 or ntok > 80:
                    continue          # headings / table dumps are not answers
                letters = [ch for ch in s if ch.isalpha()]
                if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.5:
                    continue          # UPPER-CASE headings
                ov = support(groups, set(stems(s)))
                sent_rows.append({"ci": ci, "si": si, "text": s, "overlap": ov, "chunk_score": c.rerank})
        if not sent_rows:
            return Answer(text=MESSAGES["not_found"][lang], lang=lang, status="not_found")
        dense = np.zeros(len(sent_rows))
        if self.encoder is not None:
            qv = self.encoder.encode([expand_query(query)])[0]
            sv = self.encoder.encode([r["text"] for r in sent_rows])
            dense = sv @ qv
        for r, d in zip(sent_rows, dense):
            r["score"] = 0.45 * r["overlap"] + 0.35 * float(d) + 0.20 * r["chunk_score"]
        ranked, chosen, seen = sorted(sent_rows, key=lambda r: -r["score"]), [], []
        for r in ranked:                      # skip near-duplicates (same rule repeated in several documents)
            st = set(stems(r["text"]))
            if any(len(st & o) / max(1, len(st | o)) > 0.7 for o in seen):
                continue
            chosen.append(r); seen.append(st)
            if len(chosen) >= self.max_sentences:
                break
        best = chosen[0]["score"]
        chosen = [r for r in chosen if r["score"] >= 0.55 * best]
        # add the following sentence when a chosen sentence introduces a list ("...:")
        extra = []
        for r in chosen:
            if r["text"].rstrip().endswith(":"):
                nxt = [x for x in sent_rows if x["ci"] == r["ci"] and x["si"] == r["si"] + 1]
                extra.extend(nxt)
        chosen = {(r["ci"], r["si"]): r for r in chosen + extra}.values()
        chosen = sorted(chosen, key=lambda r: (r["ci"], r["si"]))

        used_ci = sorted({r["ci"] for r in chosen})
        num = {ci: n + 1 for n, ci in enumerate(used_ci)}
        sources = [_source(evidence[ci], num[ci]) for ci in used_ci]
        lines = [MESSAGES["intro"][lang]]
        for r in chosen:
            lines.append(f"• {r['text'].strip()} [{num[r['ci']]}]")
        if any(evidence[ci].chunk.get("lang") != lang for ci in used_ci):
            lines.append(MESSAGES["other_lang"][lang])
        return Answer(text="\n".join(lines), lang=lang, sources=sources, confidence=confidence_label(top),
                      confidence_score=round(top, 4), generator=self.name,
                      used_sentences=[{"text": r["text"], "source": num[r["ci"]]} for r in chosen])


def supported_ratio(answer: str, contexts: Sequence[str], min_overlap: float = 0.6) -> float:
    """Faithfulness: share of answer sentences whose content stems are covered by the context."""
    ctx = set()
    for c in contexts:
        ctx |= set(stems(c))
    from .composer import template_phrases
    phrases = [p for g in ("intro", "other_lang") for p in MESSAGES[g].values()] + template_phrases()
    for phrase in sorted(phrases, key=len, reverse=True):   # template phrases are not factual statements
        answer = answer.replace(phrase, " ")
    answer = re.sub(r"(?m)^\s*\d+\.\s", " ", answer)
    answer = answer.replace("•", " ")
    sents = [s for s in split_sentences(re.sub(r"\[\d+\]", "", answer)) if len(tokenize(s)) >= 4]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        st = set(stems(s))
        if st and len(st & ctx) / len(st) >= min_overlap:
            ok += 1
    return ok / len(sents)
