"""Answer composer (our own models, no pre-trained LLM).

1. The question-type classifier (ml/qtype.py) says what kind of answer is expected:
   steps, a list of documents, a deadline, an amount, a place, conditions, a definition.
2. The sentence ranker (logistic regression trained by us, see `train_sentence_ranker`)
   scores every sentence of the best document fragments:
   P(sentence answers the question | features).
3. The composer cleans the chosen sentences (clause numbers "4.1", "(далее - ...)",
   broken line joins) and builds a short, readable answer in the language of the
   question: a short answer first, then numbered steps / a list, with citations [n].

Every sentence still comes from an official document, so the answer cannot contain
invented facts (faithfulness by construction).
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .generator import MESSAGES, Answer, _source, confidence_label
from .lexicon import CONCEPT_LABELS, detect_concepts, expand_query, query_stem_groups, support
from .qtype import CUES, TYPES, cue_score
from .text import content_tokens, fold, split_sentences, stems, tokenize

SENT_FEATURES = ["overlap", "dense", "chunk_score", "inv_chunk_rank", "cue", "has_number", "position",
                 "log_len", "lang_match", "listy", "bias"]

LABELS = {
    "short": {"kk": "Қысқаша жауап", "ru": "Коротко", "en": "In short"},
    "procedure": {"kk": "Қалай жасалады", "ru": "Как это сделать", "en": "How to do it"},
    "documents": {"kk": "Қажетті құжаттар мен талаптар", "ru": "Какие документы и требования",
                  "en": "Documents and requirements"},
    "deadline": {"kk": "Мерзімдері", "ru": "Сроки", "en": "Deadlines"},
    "amount": {"kk": "Мөлшері / сомасы", "ru": "Размер / сумма", "en": "Amount"},
    "place": {"kk": "Қайда жүгіну керек", "ru": "Куда обращаться", "en": "Where to apply"},
    "condition": {"kk": "Шарттары", "ru": "Условия", "en": "Conditions"},
    "definition": {"kk": "Толығырақ", "ru": "Подробнее", "en": "More details"},
    "general": {"kk": "Толығырақ", "ru": "Подробнее", "en": "More details"},
}
OTHER_LANG_NOTE = {
    "kk": "(Бұл ереже құжатта басқа тілде жазылған, сондықтан үзінділер құжаттың түпнұсқа тілінде берілді.)",
    "ru": "(Это правило в документе написано на другом языке, поэтому фрагменты приведены на языке оригинала.)",
    "en": "(This rule is written in another language in the document, so the fragments are quoted in the original.)",
}
SEE_SOURCE = {"kk": "Толық мәтін — төмендегі дереккөзде.", "ru": "Полный текст — в источнике ниже.",
              "en": "Full text is in the source below."}


NATURAL = {   # template heading -> a natural lead-in (the chat shows answers without citation markers)
    "procedure": {"kk": "Ол үшін:", "ru": "Для этого:", "en": "To do this:"},
    "documents": {"kk": "Қажет болатыны:", "ru": "Понадобится:", "en": "You will need:"},
    "deadline": {"kk": "Мерзімдеріне келсек:", "ru": "По срокам:", "en": "As for the deadlines:"},
    "amount": {"kk": "Мөлшеріне келсек:", "ru": "По размеру:", "en": "As for the amount:"},
    "place": {"kk": "Қайда жүгіну керек:", "ru": "Куда обращаться:", "en": "Where to go:"},
    "condition": {"kk": "Шарттары мынадай:", "ru": "Условия такие:", "en": "The conditions are:"},
    "definition": {"kk": "Сонымен қатар:", "ru": "Кроме того:", "en": "Also:"},
    "general": {"kk": "Сонымен қатар:", "ru": "Кроме того:", "en": "Also:"},
}
NATURAL_OTHER_LANG = {"kk": "(Бұл туралы ереже орыс/ағылшын тілінде жазылған.)",
                      "ru": "(Это правило написано на другом языке.)", "en": "(This rule is written in another language.)"}


def natural_text(text: str, lang: str) -> str:
    """The composed answer as plain text: no "[1]" markers, no "Қысқаша жауап:" / "Тақырып — Қалай жасалады:"
    template headings (the sources stay available in the API response)."""
    if not text:
        return text
    t = re.sub(r"\s*\[\d+\]", "", text)
    for d in (LABELS["short"],):
        for v in d.values():
            t = t.replace(v + ": ", "")
    for kind, d in LABELS.items():
        if kind == "short":
            continue
        for lg, v in d.items():
            nat = NATURAL.get(kind, {}).get(lang) or NATURAL.get(kind, {}).get(lg, "")
            t = re.sub(r"(?m)^(?:[^\n]{0,80}? — )?" + re.escape(v) + r":", nat, t)
    for lg, v in OTHER_LANG_NOTE.items():
        t = t.replace(v, NATURAL_OTHER_LANG.get(lang, ""))
    for v in SEE_SOURCE.values():
        t = t.replace(v, "")
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def template_phrases() -> List[str]:
    """Phrases added by the composer (not facts) - removed by the faithfulness metric."""
    out = []
    for d in LABELS.values():
        out += [v + ":" for v in d.values()]
    return out + list(OTHER_LANG_NOTE.values()) + list(SEE_SOURCE.values())


# ------------------------------------------------------------------ sentence cleaning
_CLAUSE_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[а-яa-zәғқңөұүһі]\)|\d+\)|[-•–—])\s+", re.I)
_BROKEN_DOT_RE = re.compile(r"(?<=[А-Яа-яA-Za-zӘәҒғҚқҢңӨөҰұҮүҺһІі,])\.\s+(?=[a-zа-яәғқңөұүһі])")
_HEREINAFTER_RE = re.compile(r"\((?:далее|бұдан (?:әрі|кейін)|hereinafter)[^)]{0,120}\)", re.I)


_INNER_CLAUSE_RE = re.compile(r"\s\d+\.\d+(?:\.\d+)*\.?\s+(?=[А-ЯA-ZӘҒҚҢӨҰҮҺІ])")


_LEGAL_REFS = [
    # references to laws / orders do not help a student: "..., утвержденных приказом Министра ... от ... № ..."
    re.compile(r",?\s*утвержденн\w*\s+(?:приказом|постановлением|решением|распоряжением)[^.;:]*", re.I),
    re.compile(r",?\s*approved by (?:the )?(?:order|decree|resolution|decision)[^.;:]*", re.I),
    re.compile(r"\s*\(с изменениями[^)]*\)", re.I),
    re.compile(r",?\s*(?:в соответствии с|согласно)\s+(?:приказом|постановлением|законом|Законом|Правилами)"
               r"[^,.;:]{0,160}(?=[,.;:])", re.I),
    re.compile(r",?\s*in accordance with (?:the )?(?:order|law|decree)[^,.;:]{0,160}(?=[,.;:])", re.I),
    re.compile(r"\s*№\s*\d+[\w\-/]*", re.I),
    re.compile(r"(?:ҚР\.?\s+)?[А-ЯӘҒҚҢӨҰҮҺІ][\wӘәҒғҚқҢңӨөҰұҮүҺһІі]{1,10}\s+бұйрығымен\s+бекітілген\s+"),
]


def simplify(s: str) -> str:
    """Shorter, easier sentence: legal references and long bracket notes are removed (facts stay)."""
    for rx in _LEGAL_REFS:
        s = rx.sub("", s)
    s = re.sub(r"\s*\((?=[^()]{40,}\))(?![^()]*\d)[^()]*\)", "", s)     # long bracket note without numbers
    s = re.sub(r"\s{2,}", " ", s)
    return s


def clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(.{5,80}?) \1(?=\W|$)", r"\1", s)       # "Университет тарихы Университет тарихы ..." (page title + h1)
    m = _INNER_CLAUSE_RE.search(s)
    if m and m.start() < 160:
        s = s[m.end():]                                 # "<heading> 4.1 Конкурсная комиссия ..." -> clause text
    m = re.match(r"^[^()]{1,25}\)\s+", s)
    if m:
        s = s[m.end():]                                 # "Алматы) секретарь ..." - tail of a broken bracket
    s = _BROKEN_DOT_RE.sub(" ", s)                      # "в. общежитии" -> "в общежитии" (line-join artefact)
    for _ in range(2):
        s = _CLAUSE_RE.sub("", s)                       # "4.1 ", "1) ", "а) ", "- "
    s = _HEREINAFTER_RE.sub("", s)
    s = simplify(s)
    s = re.sub(r"\s+([,.;:])", r"\1", s).strip(" ;,")
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?:":
        s += "."
    return s


def as_list(s: str) -> str:
    """'a; - b; - c' (an enumeration inside one sentence) -> one item per line."""
    items = [x.strip(" ;-–•") for x in re.split(r";\s*[-–•]\s+|;\s+(?=[-–•]?\s*[А-ЯA-ZӘҒҚҢӨҰҮҺІ])", s)]
    items = [x for x in items if x]
    if len(items) < 3:
        return s
    return "\n" + "\n".join(f"   • {x[:1].upper() + x[1:].rstrip('.')}" for x in items)


# ------------------------------------------------------------------ sentence ranker (our model)
class SentenceRanker:
    """Logistic regression over SENT_FEATURES, trained with our own gradient descent."""

    def __init__(self):
        self.w = np.zeros(len(SENT_FEATURES)); self.mu = np.zeros(len(SENT_FEATURES))
        self.sd = np.ones(len(SENT_FEATURES)); self.trained = False; self.history: List[Dict] = []

    def _z(self, X):
        Z = (X - self.mu) / self.sd
        Z[:, -1] = 1.0
        return Z

    def fit(self, X, y, epochs: int = 400, lr: float = 0.2, l2: float = 1e-3, verbose: bool = True):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        self.mu[-1], self.sd[-1] = 0.0, 1.0
        Z = self._z(X)
        pos_w = (len(y) - y.sum()) / max(1, y.sum())
        sw = np.where(y == 1, pos_w, 1.0); sw /= sw.mean()
        self.w = np.zeros(Z.shape[1])
        for ep in range(1, epochs + 1):
            p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
            self.w -= lr * (Z.T @ ((p - y) * sw) / len(y) + l2 * self.w)
            if ep % 50 == 0 or ep == 1:
                pc = np.clip(p, 1e-9, 1 - 1e-9)
                loss = float(-(sw * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).mean())
                self.history.append({"epoch": ep, "loss": loss})
                if verbose:
                    print(f"    epoch {ep:4d}  loss={loss:.4f}", flush=True)
        self.trained = True
        return self

    def predict_proba(self, X):
        if not self.trained:        # before training: a sensible hand-made combination
            w = dict(zip(SENT_FEATURES, [2.0, 2.0, 1.0, 0.5, 1.0, 0.0, -0.3, 0.2, 0.3, 0.2, -3.0]))
            z = np.asarray(X) @ np.array([w[f] for f in SENT_FEATURES])
            return 1 / (1 + np.exp(-z))
        return 1 / (1 + np.exp(-np.clip(self._z(np.asarray(X, dtype=np.float64)) @ self.w, -30, 30)))

    def weights(self) -> Dict[str, float]:
        return {f: round(float(v), 4) for f, v in zip(SENT_FEATURES, self.w)}

    def save(self, path):
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                                          "features": SENT_FEATURES, "history": self.history}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SentenceRanker":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = cls()
        m.w, m.mu, m.sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
        m.history = d.get("history", []); m.trained = True
        return m


def sentence_rows(query: str, qtype_probs: Dict[str, float], chunks: Sequence[Dict], scores: Sequence[float],
                  lang: Optional[str], encoder=None, qvec=None) -> List[Dict]:
    """Candidate sentences of the given chunks with their feature vectors."""
    groups = query_stem_groups(query)
    rows = []
    for ci, (ch, sc) in enumerate(zip(chunks, scores)):
        sents = split_sentences(ch["text"])
        for si, s in enumerate(sents):
            ntok = len(tokenize(s))
            if ntok < 5 or ntok > 90:
                continue
            if s[:1].islower():
                continue                                    # a fragment cut at the chunk border
            if re.search(r"\.{4,}|…{2,}|_{4,}|(?:\s\.){3,}", s):
                continue                                    # table of contents / blank form lines
            letters = [c for c in s if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) > 0.5:
                continue                                    # UPPER-CASE headings are not answers
            words = s.split()
            if sum(1 for w in words if len(w) >= 4 and w.isupper()) >= 2 or \
                    sum(1 for w in words if re.fullmatch(r"[\d=/+\-–%()]+", w)) > 0.2 * len(words) or "=" in s:
                continue                                    # flattened tables / diagrams ("МЕРЗІМ 1 семестр = 15 апта")
            if re.search(r"[а-яәіңғүұқөһё][А-ЯӘІҢҒҮҰҚӨҺЁ]{2,}|[а-яәіңғүұқөһё][А-ЯӘІҢҒҮҰҚӨҺЁ][а-яәіңғүұқөһё][А-ЯӘІҢҒҮҰҚӨҺЁ]", s):
                continue                                    # broken PDF text layer ("мемЛеКеТТІК ШәКІРТАҚЫ")
            cue = sum(p * cue_score(s, t) for t, p in qtype_probs.items())
            rows.append({"ci": ci, "si": si, "text": s, "f": [
                support(groups, set(stems(s))), 0.0, float(sc), 1.0 / (1 + ci), cue,
                1.0 if re.search(r"\d", s) else 0.0, si / max(1, len(sents) - 1),
                math.log1p(ntok) / 4.0, 1.0 if (lang and ch.get("lang") == lang) else 0.0,
                1.0 if re.match(r"^\s*(\d+\)|[а-яa-z]\)|[-•–])", s) or s.rstrip().endswith(":") else 0.0, 1.0]})
    if encoder is not None and rows:
        qv = encoder.encode([expand_query(query)])[0] if qvec is None else qvec
        sv = encoder.encode([r["text"] for r in rows])
        for r, d in zip(rows, sv @ qv):
            r["f"][1] = float(d)
    return rows


class Composer:
    name = "composer"

    def __init__(self, encoder=None, ranker: Optional[SentenceRanker] = None, qtype_model=None):
        self.encoder = encoder
        self.ranker = ranker or SentenceRanker()
        self.qtype_model = qtype_model

    def question_type(self, question: str) -> Dict[str, float]:
        if self.qtype_model is None:
            return {"general": 1.0}
        p = self.qtype_model.predict_proba([question])[0]
        order = np.argsort(-p)[:2]
        return {TYPES[i]: float(p[i]) for i in order}

    def generate(self, query: str, lang: str, cands: Sequence, more: bool = False) -> Answer:
        top = cands[0].rerank
        evidence = [c for c in cands if c.rerank >= max((0.3 if more else 0.5) * top, 0.1)][:8 if more else 4] \
            or list(cands[:1])
        qprobs = self.question_type(query)
        qtype = max(qprobs, key=qprobs.get)
        if "history" in detect_concepts(query):
            qtype = "general"                    # "университет тарихы қандай" is not a question about documents
        rows = sentence_rows(query, qprobs, [c.chunk for c in evidence], [c.rerank for c in evidence], lang,
                             self.encoder)
        if not rows:
            return Answer(text=MESSAGES["not_found"][lang], lang=lang, status="not_found")
        probs = self.ranker.predict_proba(np.array([r["f"] for r in rows]))
        for r, p in zip(rows, probs):
            r["p"] = float(p)
            if qtype == "definition" and r["f"][4] >= 0.5 and r["f"][0] >= 0.5:
                r["p"] += 0.5                    # "<term> – <definition>" sentence about the asked term
        # answer in the student's language when the documents have this rule in that language
        same = [r for r in rows if evidence[r["ci"]].chunk.get("lang") == lang]
        if same and len(same) < len(rows) and max(r["p"] for r in same) >= 0.8 * max(r["p"] for r in rows):
            rows = same
        ranked = sorted(rows, key=lambda r: -r["p"])
        best = ranked[0]["p"]
        limit = (10 if more else 5) if qtype in ("procedure", "documents", "condition") else (8 if more else 3)
        chosen, seen = [], []
        for r in ranked:
            if r["p"] < (0.15 if more else 0.45) * best or len(chosen) >= limit:
                break
            st = set(stems(r["text"]))
            if any(len(st & o) / max(1, len(st | o)) > 0.6 for o in seen):
                continue                                    # the same rule repeated in several documents
            chosen.append(r); seen.append(st)
        # a sentence that introduces a list ("...:") brings its next items
        extra = []
        for r in chosen:
            if r["text"].rstrip().endswith(":"):
                extra += [x for x in rows if x["ci"] == r["ci"] and r["si"] < x["si"] <= r["si"] + 4]
        chosen_map = {(r["ci"], r["si"]): r for r in chosen + extra}
        lead = chosen[0]
        rest = sorted([r for k, r in chosen_map.items() if k != (lead["ci"], lead["si"])],
                      key=lambda r: (r["ci"], r["si"]))

        used_ci = sorted({lead["ci"]} | {r["ci"] for r in rest})
        num = {ci: n + 1 for n, ci in enumerate(used_ci)}
        lines = [f"{LABELS['short'][lang]}: {as_list(clean_sentence(lead['text']))} [{num[lead['ci']]}]"]
        if rest:
            lines.append("")
            topic = next((CONCEPT_LABELS[c][lang] for c in sorted(detect_concepts(query)) if c in CONCEPT_LABELS
                          and c != "documents"), "")
            lines.append((f"{topic} — " if topic else "") + LABELS[qtype][lang] + ":")
            for i, r in enumerate(rest, 1):
                lines.append(f"{i}. {as_list(clean_sentence(r['text']))} [{num[r['ci']]}]")
        if any(evidence[ci].chunk.get("lang") != lang for ci in used_ci):
            lines += ["", OTHER_LANG_NOTE[lang]]
        sources = [_source(evidence[ci], num[ci]) for ci in used_ci]
        used = [lead] + rest
        return Answer(text="\n".join(lines), lang=lang, sources=sources, confidence=confidence_label(top),
                      confidence_score=round(top, 4), generator=self.name,
                      used_sentences=[{"text": r["text"], "source": num[r["ci"]], "p": round(r["p"], 3)}
                                      for r in used])


# ------------------------------------------------------------------ self-supervised training data
TYPE_PREFIX = {
    "procedure": {"ru": "как оформить", "kk": "қалай рәсімдеймін", "en": "how do I"},
    "documents": {"ru": "какие документы нужны", "kk": "қандай құжаттар керек", "en": "which documents are needed"},
    "deadline": {"ru": "какой срок", "kk": "мерзімі қандай", "en": "what is the deadline"},
    "amount": {"ru": "какой размер", "kk": "қанша", "en": "how much"},
    "place": {"ru": "куда обращаться", "kk": "қайда бару керек", "en": "where to apply"},
    "condition": {"ru": "какие условия", "kk": "шарттары қандай", "en": "what are the conditions"},
}


def make_sentence_training(chunks: Sequence[Dict], retriever, qtype_model, encoder, n: int = 2500,
                           seed: int = 21):
    """(X, y, groups): a pseudo-question is made from a sentence (a random part of its content words, sometimes
    without Kazakh letters, with a question-type phrase when the sentence carries that kind of information);
    the candidates are all sentences of its chunk and of the best retrieved other chunks; label 1 = the source."""
    rnd = random.Random(seed)
    idx = list(range(len(chunks)))
    rnd.shuffle(idx)
    X, y, groups, made = [], [], [], 0
    for ci in idx:
        if made >= n:
            break
        ch = chunks[ci]
        sents = [s for s in split_sentences(ch["text"]) if 6 <= len(tokenize(s)) <= 60]
        if len(sents) < 3:
            continue
        target = rnd.choice(sents)
        words = content_tokens(target)
        if len(words) < 4:
            continue
        k = max(1, int(len(words) * rnd.uniform(0.15, 0.4)))     # few words -> the task is not trivial
        q_words = rnd.sample(words, k)
        lang = ch.get("lang", "ru")
        types = [t for t in TYPE_PREFIX if cue_score(target, t) >= 0.5]
        q = " ".join(q_words)
        if types and rnd.random() < 0.85:
            q = f"{TYPE_PREFIX[rnd.choice(types)][lang]} {q}"
        if rnd.random() < 0.3:
            q = fold(q)
        qprobs = {TYPES[i]: float(p) for i, p in enumerate(qtype_model.predict_proba([q])[0])} \
            if qtype_model is not None else {"general": 1.0}
        top2 = dict(sorted(qprobs.items(), key=lambda kv: -kv[1])[:2])
        others = [c.chunk for c in retriever.search(q, k=4) if c.chunk.get("chunk_id") != ch.get("chunk_id")][:2]
        cl = [ch] + others
        rows = sentence_rows(q, top2, cl, [1.0] + [0.6] * len(others), lang, encoder)
        labels = [1 if (r["ci"] == 0 and r["text"] == target) else 0 for r in rows]
        if sum(labels) != 1:
            continue
        for r, lab in zip(rows, labels):
            X.append(r["f"]); y.append(lab); groups.append(made)
        made += 1
    return np.array(X), np.array(y, dtype=np.float64), np.array(groups)
