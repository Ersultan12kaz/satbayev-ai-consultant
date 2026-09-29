"""Scope classifier: is the question about the university or not?

Model: TF-IDF (word 1-2 grams + character 2-4 grams) -> logistic regression
trained by our own mini-batch gradient descent (numpy) with L2 regularisation.
Loss curves are saved during training so the training process can be shown.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import fold, normalize

# ----------------------------------------------------------------- training data
UNI_TOPICS = {
    "ru": ["академический отпуск", "стипендия", "общежитие", "перевод на другую специальность", "восстановление",
           "отчисление", "экзаменационная сессия", "апелляция", "пересдача экзамена", "летний семестр",
           "GPA", "кредиты ECTS", "профессиональная практика", "дипломная работа", "итоговая аттестация",
           "проверка на плагиат", "эдвайзер", "индивидуальный учебный план", "академическая мобильность",
           "прием на бакалавриат", "магистратура", "докторантура PhD", "грант", "скидка на обучение",
           "дистанционное обучение", "онлайн-экзамен", "электронная библиотека", "студенческое самоуправление",
           "этический кодекс студента", "Minor программа", "дуальное обучение", "приложение к диплому",
           "справка с места обучения", "оплата за обучение", "расписание занятий", "учебная система Polytech",
           "транскрипт", "рейтинг допуска", "академическая задолженность", "инклюзивное обучение",
           "иностранные студенты", "двудипломная программа", "повторное обучение", "Retake", "FX оценка"],
    "kk": ["академиялық демалыс", "шәкіртақы", "жатақхана", "басқа мамандыққа ауысу", "қайта қабылдау",
           "оқудан шығару", "емтихан сессиясы", "апелляция", "емтиханды қайта тапсыру", "жазғы семестр",
           "GPA", "ECTS кредиттері", "кәсіптік практика", "дипломдық жұмыс", "қорытынды аттестаттау",
           "плагиатқа тексеру", "эдвайзер", "жеке оқу жоспары", "академиялық ұтқырлық",
           "бакалавриатқа қабылдау", "магистратура", "PhD докторантура", "грант", "оқу ақысына жеңілдік",
           "қашықтықтан оқыту", "онлайн емтихан", "электронды кітапхана", "студенттік өзін-өзі басқару",
           "білім алушылардың этикалық кодексі", "Minor бағдарламасы", "дуальды оқыту", "дипломға қосымша",
           "оқу орнынан анықтама", "оқу ақысын төлеу", "сабақ кестесі", "транскрипт", "рейтинг",
           "академиялық берешек", "инклюзивті оқыту", "шетелдік студенттер", "қос дипломды бағдарлама"],
    "en": ["academic leave", "scholarship", "dormitory", "transfer to another program", "reinstatement",
           "expulsion", "examination session", "appeal", "exam retake", "summer semester", "GPA",
           "ECTS credits", "professional internship", "graduation thesis", "final attestation",
           "plagiarism check", "adviser", "individual study plan", "academic mobility",
           "bachelor admission", "master's program", "PhD doctoral studies", "state grant",
           "tuition discount", "distance learning", "online exam", "electronic library",
           "student self-government", "student code of ethics", "Minor program", "dual education",
           "diploma supplement", "certificate of enrollment", "tuition fee payment", "class schedule",
           "transcript", "admission rating", "academic debt", "inclusive education",
           "international students", "double degree program"],
}

UNI_PATTERNS = {
    "ru": ["Как оформить {t}?", "Что нужно для {t}?", "Какие документы нужны для {t}?", "Расскажи про {t}",
           "Какой порядок {t} в университете?", "Кто отвечает за {t}?", "Сроки {t}?", "{t} - какие правила?",
           "Можно ли получить {t}?", "Что говорит положение про {t}?", "{t}"],
    "kk": ["{t} қалай рәсімделеді?", "{t} үшін не қажет?", "{t} үшін қандай құжаттар керек?",
           "{t} туралы айтып бер", "Университетте {t} тәртібі қандай?", "{t} мерзімі қандай?",
           "{t} ережесі қандай?", "{t} алуға бола ма?", "{t}"],
    "en": ["How do I apply for {t}?", "What is needed for {t}?", "Which documents are required for {t}?",
           "Tell me about {t}", "What is the procedure of {t} at the university?", "What are the deadlines for {t}?",
           "What are the rules for {t}?", "Can I get {t}?", "{t}"],
}

OFF_TOPICS = {
    "ru": ["чемпионат мира по футболу", "погода завтра", "рецепт плова", "курс доллара", "новый фильм Marvel",
           "лучший смартфон 2026", "как похудеть", "президент США", "столица Австралии", "как выучить гитару",
           "криптовалюта биткоин", "гороскоп на неделю", "песня года", "как починить машину", "рыбалка",
           "игра Minecraft", "Олимпийские игры", "рецепт торта", "полет на Марс", "курс тенге к рублю",
           "лучший ресторан в Алматы", "как стать блогером", "сериал Игра престолов", "кто такой Илон Маск",
           "смешной анекдот", "вулканы Исландии", "поездка в Турцию", "iPhone или Samsung", "квантовая физика",
           "как приготовить борщ", "погода на выходные", "последние новости", "как заработать в интернете",
           "подарок на день рождения", "цена на золото", "лучшая игра года", "как научиться водить машину",
           "население Китая", "тренировка в зале", "смысл жизни"],
    "kk": ["футболдан әлем чемпионаты", "ертеңгі ауа райы", "палау рецепті", "доллар бағамы", "жаңа фильм",
           "ең жақсы смартфон", "қалай арықтауға болады", "АҚШ президенті", "Австралияның астанасы",
           "гитарада ойнауды үйрену", "биткоин", "апталық жұлдыз жорамал", "жылдың әні", "көлікті жөндеу",
           "балық аулау", "Minecraft ойыны", "Олимпиада ойындары", "торт рецепті", "Марсқа ұшу",
           "Алматыдағы ең жақсы мейрамхана", "блогер болу", "Илон Маск кім", "күлкілі анекдот",
           "Түркияға саяхат", "бешбармақ қалай пісіріледі", "бүгінгі жаңалықтар", "демалыс күнгі ауа райы",
           "туған күнге сыйлық", "алтын бағасы", "көлік жүргізуді үйрену", "Қытай халқының саны",
           "интернетте ақша табу", "жаттығу жоспары", "өмірдің мәні"],
    "en": ["the football world cup", "tomorrow's weather", "a pilaf recipe", "the dollar exchange rate",
           "the new Marvel movie", "the best smartphone of 2026", "how to lose weight", "the US president",
           "the capital of Australia", "learning guitar", "bitcoin price", "weekly horoscope", "song of the year",
           "fixing my car", "fishing tips", "the game Minecraft", "the Olympic games", "a cake recipe",
           "flying to Mars", "the best restaurant in Almaty", "becoming a blogger", "who Elon Musk is",
           "a funny joke", "volcanoes in Iceland", "a trip to Turkey", "quantum physics", "the weather tomorrow", "today's news",
           "the best pizza nearby", "how to fix my phone", "a good book to read", "the stock market",
           "learning to drive a car", "a birthday gift idea", "the latest iPhone", "how to cook pasta",
           "the NBA finals", "a workout plan", "the population of China", "writing a love letter",
           "the price of gold", "a video game to play", "the meaning of life", "how to make money online"],
}

OFF_PATTERNS = {
    "ru": ["Расскажи про {t}", "Что ты знаешь про {t}?", "Кто выиграл {t}?", "Подскажи {t}", "{t}?",
           "Как найти {t}?", "Посоветуй {t}"],
    "kk": ["{t} туралы айтып бер", "{t} туралы не білесің?", "{t} қандай?", "{t}?", "Маған {t} айтшы"],
    "en": ["Tell me about {t}", "What do you know about {t}?", "Who won {t}?", "Recommend {t}", "{t}?",
           "Where can I find {t}?", "What's {t}?", "How is {t}?", "Can you help me with {t}?"],
}


def generate_training_data(extra_positive: Sequence[str] = (), seed: int = 13) -> Tuple[List[str], List[int]]:
    rnd = random.Random(seed)
    X, y = [], []
    for lang in UNI_TOPICS:
        for t in UNI_TOPICS[lang]:
            for p in UNI_PATTERNS[lang]:
                X.append(p.format(t=t)); y.append(1)
        for t in OFF_TOPICS[lang]:
            for p in OFF_PATTERNS[lang]:
                X.append(p.format(t=t)); y.append(0)
    for s in extra_positive:
        X.append(s); y.append(1)
    idx = list(range(len(X)))
    rnd.shuffle(idx)
    return [X[i] for i in idx], [y[i] for i in idx]


# ----------------------------------------------------------------- model
class ScopeClassifier:
    def __init__(self, l2: float = 1e-4, lr: float = 0.5, epochs: int = 30, batch_size: int = 64):
        self.l2, self.lr, self.epochs, self.batch_size = l2, lr, epochs, batch_size
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                        token_pattern=r"(?u)\b\w+\b", max_features=30000)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                        max_features=50000)
        self.w: np.ndarray | None = None
        self.b = 0.0
        self.history: List[Dict] = []

    def _features(self, texts: Sequence[str], fit: bool = False):
        texts = [fold(normalize(t)) for t in texts]
        if fit:
            return sparse.hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]).tocsr()
        return sparse.hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, texts: Sequence[str], labels: Sequence[int], val: Tuple[Sequence[str], Sequence[int]] | None = None,
            seed: int = 0, verbose: bool = True) -> "ScopeClassifier":
        X = self._features(texts, fit=True)
        y = np.asarray(labels, dtype=np.float64)
        # class weights: positives and negatives contribute equally (data is imbalanced)
        pos = max(y.sum(), 1.0); neg = max(len(y) - y.sum(), 1.0)
        sw_all = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        Xv = yv = None
        if val is not None:
            Xv, yv = self._features(val[0]), np.asarray(val[1], dtype=np.float64)
        rng = np.random.default_rng(seed)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        # Adam optimiser
        m_w = np.zeros_like(self.w); v_w = np.zeros_like(self.w); m_b = v_b = 0.0
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        lr = self.lr * 0.02
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(X.shape[0])
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                xb, yb = X[idx], y[idx]
                p = self._sigmoid(xb @ self.w + self.b)
                g = (p - yb) * sw_all[idx]
                gw = np.asarray(xb.T @ g).ravel() / len(idx) + self.l2 * self.w
                gb = float(g.mean())
                step += 1
                m_w = b1 * m_w + (1 - b1) * gw; v_w = b2 * v_w + (1 - b2) * gw * gw
                m_b = b1 * m_b + (1 - b1) * gb; v_b = b2 * v_b + (1 - b2) * gb * gb
                mhat = m_w / (1 - b1 ** step); vhat = v_w / (1 - b2 ** step)
                self.w -= lr * mhat / (np.sqrt(vhat) + eps)
                self.b -= lr * (m_b / (1 - b1 ** step)) / (np.sqrt(v_b / (1 - b2 ** step)) + eps)
            rec = {"epoch": epoch, "train_loss": self._loss(X, y), "train_acc": self._acc(X, y)}
            if Xv is not None:
                rec.update(val_loss=self._loss(Xv, yv), val_acc=self._acc(Xv, yv))
            self.history.append(rec)
            if verbose:
                print("  epoch %2d | " % epoch + " | ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch"))
        return self

    def _loss(self, X, y):
        p = np.clip(self._sigmoid(X @ self.w + self.b), 1e-9, 1 - 1e-9)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + 0.5 * self.l2 * (self.w ** 2).sum())

    def _acc(self, X, y):
        return float(((self._sigmoid(X @ self.w + self.b) >= 0.5) == y).mean())

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self._sigmoid(self._features(texts) @ self.w + self.b)

    def is_university_question(self, text: str, threshold: float = 0.5) -> Tuple[bool, float]:
        p = float(self.predict_proba([text])[0])
        return p >= threshold, p

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ScopeClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)
