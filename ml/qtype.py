"""Question-type classifier (our own model): what kind of answer does the student expect?

Types: procedure ("how"), documents ("which documents"), deadline ("when / how long"),
amount ("how much"), place ("where / whom to contact"), condition ("can I / who is eligible /
requirements"), definition ("what is"), general.

Model: TF-IDF (word 1-2 grams + character 2-5 grams, after letter folding) ->
multinomial logistic regression (softmax) trained with our own mini-batch gradient
descent (numpy).  Training questions are generated from templates x university topics
in Kazakh, Russian and English (+ colloquial spellings without Kazakh letters).

The type is used by the answer composer (ml/composer.py): it chooses sentences that
contain the expected kind of information (a list of documents, a deadline, an amount ...)
and the form of the answer (numbered steps, a list, a short fact).
"""
from __future__ import annotations

import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .scope import UNI_TOPICS
from .text import fold, normalize, stems

TYPES = ["procedure", "documents", "deadline", "amount", "place", "condition", "definition", "general"]

TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "procedure": {
        "ru": ["Как оформить {t}?", "Как получить {t}?", "Какой порядок {t}?", "Что нужно сделать для {t}?",
               "Как подать заявление на {t}?", "Как проходит {t}?", "Как мне оформить {t}", "как получить {t}",
               "Какие шаги для {t}?", "Как происходит {t} в университете?"],
        "kk": ["{t} қалай рәсімделеді?", "{t} қалай алуға болады?", "{t} тәртібі қандай?",
               "{t} үшін не істеу керек?", "{t} қалай алам?", "{t} калай алуга болады", "{t} қалай өтеді?",
               "{t} қалай тапсырамын?", "{t} алу үшін не істеймін", "{t} рәсімдеу тәртібі"],
        "en": ["How do I apply for {t}?", "How can I get {t}?", "What is the procedure for {t}?",
               "What should I do to get {t}?", "How does {t} work?", "What are the steps for {t}?",
               "how to get {t}"],
    },
    "documents": {
        "ru": ["Какие документы нужны для {t}?", "Какие документы нужны на {t}?", "Что нужно предоставить для {t}?",
               "Какие справки нужны для {t}?", "Список документов для {t}", "Что сдать для {t}?",
               "документы для {t}"],
        "kk": ["{t} үшін қандай құжаттар керек?", "{t} қандай құжаттар қажет?", "{t} қандай документтер керек",
               "{t} үшін не тапсыру керек?", "{t} үшін қандай анықтама керек?", "{t} құжаттар тізімі",
               "{t} кандай кужаттар керек"],
        "en": ["Which documents are required for {t}?", "What documents do I need for {t}?",
               "What papers should I submit for {t}?", "List of documents for {t}"],
    },
    "deadline": {
        "ru": ["Когда {t}?", "Какие сроки {t}?", "До какого числа {t}?", "В какие сроки подается {t}?",
               "Сколько дней длится {t}?", "На какой срок дается {t}?", "Когда начинается {t}?"],
        "kk": ["{t} қашан?", "{t} мерзімі қандай?", "{t} қанша уақыт?", "{t} қай кезде?", "{t} кашан басталады",
               "{t} неше күн?", "{t} қанша мерзімге беріледі?"],
        "en": ["When is {t}?", "What is the deadline for {t}?", "How long is {t}?", "When does {t} start?",
               "For how long is {t} granted?"],
    },
    "amount": {
        "ru": ["Сколько стоит {t}?", "Какой размер {t}?", "Сколько платят {t}?", "Какая сумма {t}?",
               "Сколько процентов {t}?", "Размер {t}"],
        "kk": ["{t} қанша?", "{t} мөлшері қанша?", "{t} қанша тұрады?", "{t} канша акша", "{t} сомасы қанша?",
               "{t} неше пайыз?"],
        "en": ["How much is {t}?", "What is the amount of {t}?", "How much does {t} cost?", "What is the size of {t}?"],
    },
    "place": {
        "ru": ["Куда обращаться по {t}?", "Кто отвечает за {t}?", "Где оформить {t}?", "Куда подавать на {t}?",
               "В какой отдел идти насчет {t}?", "Кто принимает решение о {t}?"],
        "kk": ["{t} бойынша қайда бару керек?", "{t} үшін кімге жүгіну керек?", "{t} қайда рәсімделеді?",
               "{t} кім шешеді?", "{t} кайда барам", "{t} бойынша қай бөлімге барамын?"],
        "en": ["Where do I apply for {t}?", "Who is responsible for {t}?", "Which office handles {t}?",
               "Whom should I contact about {t}?"],
    },
    "condition": {
        "ru": ["Можно ли получить {t}?", "Кто имеет право на {t}?", "Какие условия {t}?", "Какие требования для {t}?",
               "При каких условиях дается {t}?", "Могу ли я получить {t}?", "Кому положена {t}?"],
        "kk": ["{t} алуға бола ма?", "{t} алуға кім құқылы?", "{t} шарттары қандай?", "{t} талаптары қандай?",
               "{t} кімге беріледі?", "маған {t} бола ма", "{t} алуга бола ма"],
        "en": ["Can I get {t}?", "Who is eligible for {t}?", "What are the requirements for {t}?",
               "What are the conditions for {t}?", "Am I allowed to get {t}?"],
    },
    "definition": {
        "ru": ["Что такое {t}?", "Что означает {t}?", "Что значит {t}?", "Объясни, что такое {t}"],
        "kk": ["{t} деген не?", "{t} дегеніміз не?", "{t} не?", "{t} деген не екен"],
        "en": ["What is {t}?", "What does {t} mean?", "Explain {t}", "Define {t}"],
    },
    "general": {
        "ru": ["Расскажи про {t}", "Информация о {t}", "{t}", "Что говорит положение о {t}?", "Правила {t}"],
        "kk": ["{t} туралы айтып бер", "{t} туралы ақпарат", "{t}", "{t} ережесі", "{t} туралы"],
        "en": ["Tell me about {t}", "Information about {t}", "{t}", "Rules of {t}"],
    },
}

# ---------------------------------------------------------------- cue words of each answer type
# (folded 6-letter stems): a sentence that contains them carries this kind of information
CUES: Dict[str, List[str]] = {
    "procedure": "подает подача подать оформл рассма заявл приказ после затем этап осущест провод проход "
                 "тапсыр береди карала ресимд откизи жузеге submit apply review step issued proced".split(),
    "documents": "докуме справк копия копии заявле удосто паспор фотогр медици справо отиниш куалик аныкта "
                 "кужатт кужат ксерок форма appli certif copy passpo docume form".split(),
    "deadline": "срок течени дней день недел месяц года позднее дата период мерзим кун ай апта жыл "
                "уакыт кешикп deadli days week month within period date until".split(),
    "amount": "размер сумм стоимо тенге мрп процен оплат плата молшер сомас тенге пайыз акы толем "
              "amount fee tenge percen cost paymen".split(),
    "place": "обраща офис отдел деканат департ центр комисс ответс кабине управл служба болим орталы "
             "деканат комиссия жауапт basshy office depart center commit respon dean".split(),
    "condition": "право могут может допуск услови требов наличи случае обязан кукылы шарт талап жагдай "
                 "мумкин тиис eligib may must condit requir allowe case".split(),
    "definition": "являет понима назыв опреде дегени болып табыл означа means refers define is".split(),
    "general": [],
}
_NUM_RE = re.compile(r"\d")


def cue_score(sentence: str, qtype: str) -> float:
    """0..1: how much the sentence contains the kind of information of the question type."""
    if qtype == "definition":
        # "<term> – <definition>", "... является ...", "... дегеніміз ...", "... refers to ..."
        s = sentence.strip()
        if re.match(r"^[^,;:]{3,80}?\s[–—-]\s", s) or re.search(
                r"(являетс|понимаетс|называетс|означает|представляет собой|дегенимиз|деп аталады|деп тусиниледи|"
                r"болып табылады|refers to|is defined as|\bmeans\b|\bis a\b|\bis an\b)", fold(s)):
            return 1.0
        return 0.0
    cues = CUES.get(qtype) or []
    if not cues:
        return 0.0
    st = set(stems(sentence))
    hits = sum(1 for c in cues if any(s.startswith(c) for s in st))
    score = min(1.0, hits / 2.0)
    if qtype in ("deadline", "amount") and _NUM_RE.search(sentence):
        score = min(1.0, score + 0.5)
    return score


def generate_training_data(seed: int = 11) -> Tuple[List[str], List[int]]:
    rnd = random.Random(seed)
    X, y = [], []
    for ti, t in enumerate(TYPES):
        for lang, pats in TEMPLATES[t].items():
            topics = UNI_TOPICS[lang]
            for p in pats:
                for topic in rnd.sample(topics, min(len(topics), 14)):
                    q = p.format(t=topic)
                    if rnd.random() < 0.25:
                        q = q.upper() if rnd.random() < 0.5 else fold(q)   # CAPS / typed without Kazakh letters
                    if rnd.random() < 0.3:
                        q = q.rstrip("?")
                    X.append(q); y.append(ti)
    idx = list(range(len(X)))
    rnd.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


class QuestionTypeClassifier:
    def __init__(self, l2: float = 1e-4, lr: float = 0.5, epochs: int = 40, batch_size: int = 64):
        self.l2, self.lr, self.epochs, self.batch_size = l2, lr, epochs, batch_size
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
                                        token_pattern=r"(?u)\b\w+\b")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                        max_features=40000)
        self.W = None
        self.b = None
        self.history: List[Dict] = []

    def _features(self, texts: Sequence[str], fit: bool = False):
        texts = [fold(normalize(t)) for t in texts]
        if fit:
            return sparse.hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]).tocsr()
        return sparse.hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def fit(self, texts, labels, val=None, verbose: bool = True, seed: int = 0) -> "QuestionTypeClassifier":
        X = self._features(texts, fit=True)
        y = np.asarray(labels)
        k = len(TYPES)
        Y = np.eye(k)[y]
        rng = np.random.default_rng(seed)
        self.W = np.zeros((X.shape[1], k)); self.b = np.zeros(k)
        Xv = self._features(val[0]) if val else None
        for ep in range(1, self.epochs + 1):
            order = rng.permutation(X.shape[0])
            for s in range(0, len(order), self.batch_size):
                bi = order[s:s + self.batch_size]
                P = self._softmax(X[bi] @ self.W + self.b)
                G = (P - Y[bi]) / len(bi)                      # d(cross-entropy)/d(logits)
                self.W -= self.lr * (X[bi].T @ G + self.l2 * self.W)
                self.b -= self.lr * G.sum(0)
            P = self._softmax(X @ self.W + self.b)
            loss = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
            rec = {"epoch": ep, "loss": loss, "train_acc": float((P.argmax(1) == y).mean())}
            if Xv is not None:
                rec["val_acc"] = float((self._softmax(Xv @ self.W + self.b).argmax(1) == np.asarray(val[1])).mean())
            self.history.append(rec)
            if verbose and (ep % 10 == 0 or ep == 1):
                print("    epoch {epoch:3d}  loss={loss:.4f}  train_acc={train_acc:.3f}".format(**rec)
                      + (f"  val_acc={rec['val_acc']:.3f}" if "val_acc" in rec else ""), flush=True)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self._softmax(self._features(texts) @ self.W + self.b)

    def predict(self, text: str) -> Tuple[str, float]:
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return TYPES[i], float(p[i])

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "QuestionTypeClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
