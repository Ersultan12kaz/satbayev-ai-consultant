"""Language identification model (RU / KZ / EN), implemented from scratch.

Model: multinomial Naive Bayes over character n-grams (1..3) with
Laplace smoothing + a Kazakh-letter prior.  Trained on text chunks of the
official documents whose language is known from the website section
(/ru/, /kk/, /en/).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .text import KAZAKH_LETTERS, fold, normalize

# Kazakh words typed without Kazakh letters ("калай", "кандай", "керек" ...) - strong signal for colloquial questions
KK_MARKERS = set(fold("""қалай қандай қанша керек болады бола ма ме ба бе қайда қашан үшін туралы бойынша және
    алам аламын алуға беру бер берші өтініш тапсыру оқу оқыту өткізіледі өткізу жатақхана шәкіртақы емтихан
    құжат құжаттар деген дегеніміз неге кімге кім жоқ бар қажет мүмкін""").split()) - {"не", "ма", "ме", "бар"}

LANGS = ("ru", "kk", "en")


def char_ngrams(text: str, n_max: int = 3) -> List[str]:
    text = " " + normalize(text).lower() + " "
    grams = []
    for n in range(1, n_max + 1):
        grams.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return grams


class LanguageDetector:
    def __init__(self, n_max: int = 3, alpha: float = 0.5, max_features: int = 20000):
        self.n_max = n_max
        self.alpha = alpha
        self.max_features = max_features
        self.log_prior: Dict[str, float] = {}
        self.log_prob: Dict[str, Dict[str, float]] = {}
        self.log_unk: Dict[str, float] = {}
        self.vocab: set = set()

    # ------------------------------------------------------------ training
    def fit(self, texts: Sequence[str], labels: Sequence[str], augment_folded: bool = True) -> "LanguageDetector":
        """augment_folded: Kazakh texts are also added without Kazakh letters ("жатакханага калай"),
        the way students often type them on a Russian keyboard."""
        if augment_folded:
            texts, labels = list(texts), list(labels)
            extra = [(fold(t), y) for t, y in zip(texts, labels) if y == "kk"]
            texts += [t for t, _ in extra]; labels += [y for _, y in extra]
        counts: Dict[str, Counter] = defaultdict(Counter)
        docs = Counter(labels)
        total_counter: Counter = Counter()
        for t, y in zip(texts, labels):
            g = char_ngrams(t, self.n_max)
            counts[y].update(g)
            total_counter.update(g)
        self.vocab = {g for g, _ in total_counter.most_common(self.max_features)}
        V = len(self.vocab)
        n_docs = sum(docs.values())
        for y in docs:
            self.log_prior[y] = math.log(docs[y] / n_docs)
            tot = sum(c for g, c in counts[y].items() if g in self.vocab)
            denom = tot + self.alpha * V
            self.log_prob[y] = {g: math.log((counts[y][g] + self.alpha) / denom)
                                for g in self.vocab if counts[y][g] > 0}
            self.log_unk[y] = math.log(self.alpha / denom)
        return self

    # ----------------------------------------------------------- inference
    def scores(self, text: str) -> Dict[str, float]:
        grams = [g for g in char_ngrams(text, self.n_max) if g in self.vocab]
        out = {}
        for y in self.log_prior:
            lp = self.log_prob[y]
            unk = self.log_unk[y]
            out[y] = self.log_prior[y] + sum(lp.get(g, unk) for g in grams)
        # Kazakh-specific letters are a very strong signal
        kz = sum(ch in KAZAKH_LETTERS for ch in text)
        if kz and "kk" in out:
            out["kk"] += 4.0 * kz
        words = re.findall(r"\w+", fold(text))
        markers = sum(w in KK_MARKERS for w in words)
        if markers and "kk" in out:
            out["kk"] += 15.0 * markers
            if markers >= 2 or markers / max(1, len(words)) >= 0.2:     # "калай", "канша", "керек" decide
                out["kk"] = max(out.values()) + 1.0
        return out

    def predict_proba(self, text: str) -> Dict[str, float]:
        s = self.scores(text)
        m = max(s.values())
        exp = {k: math.exp(v - m) for k, v in s.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def predict(self, text: str) -> str:
        if not text.strip():
            return "ru"
        s = self.scores(text)
        return max(s, key=s.get)

    # --------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"n_max": self.n_max, "alpha": self.alpha, "log_prior": self.log_prior,
                       "log_prob": self.log_prob, "log_unk": self.log_unk,
                       "vocab": sorted(self.vocab)}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "LanguageDetector":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = cls(n_max=d["n_max"], alpha=d["alpha"])
        m.log_prior, m.log_prob, m.log_unk = d["log_prior"], d["log_prob"], d["log_unk"]
        m.vocab = set(d["vocab"])
        return m


_KK_COLLOQUIAL = re.compile(
    r"\b(калай|кандай|канша|кайда|кашан|керек|керекпе|болады|болама|бола ма|деген|дегенимиз|туралы|ушин|неге|"
    r"кимге|жане|мен бе|алам|аламын|алуга|беру|береди|откизиледи|откизу|тапсыру|тапсырамын|жасау|бар ма|"
    r"жатакхана\w*|шакиртакы\w*|емтихан\w*|кужат\w*|студенттер\w*|"
    # mixed Kazakh-Russian messages ("академка алсам грант сохраняется ма", "расписание кайдан корем")
    r"кайдан|кайдан\w+|корем\w*|коруге|корсем|алсам|берсем|барсам|болса\w*|айтшы|айтып|жайлы|неше|"
    r"кайтадан|кайта|жок|бизде|маган|сизде|керекпи|ма|ме|ба|бе|па|пе)\b")


def kazakh_without_letters(text: str) -> bool:
    """Kazakh typed on a Russian keyboard ("дуальный обучение калай откизиледи"): Kazakh function words and
    suffixes written with Russian letters."""
    t = text.lower().replace("ё", "е")
    return len(_KK_COLLOQUIAL.findall(t)) >= 1


def heuristic_language(text: str) -> str:
    """Rule-based language guess (used for label cleaning and before the model is trained).

    Kazakh-specific letters make up ~4-8 % of the letters of a Kazakh text, while Russian
    and English documents of the university contain only a few of them (the name "Сәтбаев",
    headers in Kazakh).  So we use the *share* of Kazakh letters, not their mere presence.
    """
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return "ru"
    kz = sum(c in KAZAKH_LETTERS for c in letters) / len(letters)
    cyr = sum("\u0400" <= c <= "\u04ff" for c in letters) / len(letters)
    if kz >= 0.015 or (len(letters) < 150 and kz > 0):
        return "kk"
    return "ru" if cyr > 0.5 else "en"


def evaluate(model: LanguageDetector, texts: Iterable[str], labels: Iterable[str]) -> Tuple[float, Dict]:
    conf: Dict[str, Counter] = defaultdict(Counter)
    ok = n = 0
    for t, y in zip(texts, labels):
        p = model.predict(t)
        conf[y][p] += 1
        ok += p == y
        n += 1
    return (ok / n if n else 0.0), {k: dict(v) for k, v in conf.items()}
