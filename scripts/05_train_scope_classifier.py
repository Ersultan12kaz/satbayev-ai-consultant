"""STEP 5 - Train the scope classifier (university-related question vs. off-topic).

Positive examples: question templates about university topics + questions built
from section headings of the real documents.  Negative examples: off-topic
questions (sport, weather, cooking, ...) in RU / KZ / EN.
Model: TF-IDF (words + char n-grams) -> logistic regression trained with our own
mini-batch Adam optimiser.  Loss / accuracy curves are saved to reports/figures.
"""
from __future__ import annotations

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import EVAL, FIGURES, MODELS, PROCESSED, REPORTS, Tee, Timer, banner, save_json

from ml.documents import read_jsonl
from ml.scope import ScopeClassifier, generate_training_data
from ml.training_data import Q_TEMPLATES, _clean_heading
from ml.text import tokenize


def main():
    Tee("05_scope")
    banner("STEP 5. Training the scope classifier")
    rnd = random.Random(1)
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    heads = list({(c["section"], c["lang"]) for c in chunks if c.get("section") and 2 <= len(tokenize(c["section"])) <= 12})
    rnd.shuffle(heads)
    extra = [rnd.choice(Q_TEMPLATES.get(l, Q_TEMPLATES["ru"])).format(h=_clean_heading(h)) for h, l in heads[:1500]]
    X, y = generate_training_data(extra_positive=extra)
    n_val = len(X) // 5
    Xtr, ytr, Xva, yva = X[n_val:], y[n_val:], X[:n_val], y[:n_val]
    print(f"  examples: train={len(Xtr)} val={len(Xva)} | positives={sum(y)} negatives={len(y) - sum(y)}")

    with Timer() as t:
        model = ScopeClassifier(epochs=25).fit(Xtr, ytr, val=(Xva, yva))
    print(f"  trained in {t.sec:.1f}s")

    results = {"history": model.history}
    ins = EVAL / "questions.jsonl"
    outs = EVAL / "out_of_scope.jsonl"
    if ins.exists() and outs.exists():
        qi = [q["question"] for q in read_jsonl(ins)]
        qo = [q["question"] for q in read_jsonl(outs)]
        pi, po = model.predict_proba(qi), model.predict_proba(qo)
        tp = int((pi >= 0.5).sum()); fn = len(qi) - tp
        tn = int((po < 0.5).sum()); fp = len(qo) - tn
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        results["eval"] = {"in_scope_recall": rec, "out_of_scope_rejection": tn / max(1, len(qo)),
                           "precision": prec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
                           "tp": tp, "fn": fn, "tn": tn, "fp": fp}
        print(f"\n  Evaluation questions: in-scope accepted {tp}/{len(qi)}, "
              f"off-topic rejected {tn}/{len(qo)}  (F1={results['eval']['f1']:.3f})")
    model.save(MODELS / "scope.pkl")
    save_json(REPORTS / "scope_metrics.json", results)

    h = model.history
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label="train")
    ax[0].plot([r["epoch"] for r in h], [r["val_loss"] for r in h], label="validation")
    ax[0].set_title("Scope classifier: log-loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot([r["epoch"] for r in h], [r["train_acc"] for r in h], label="train")
    ax[1].plot([r["epoch"] for r in h], [r["val_acc"] for r in h], label="validation")
    ax[1].set_title("Scope classifier: accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES / "scope_training.png", dpi=150)
    print("  saved: models/scope.pkl, reports/figures/scope_training.png")
    for q in ["Как оформить академический отпуск?", "Кто выиграл чемпионат мира по футболу?",
              "Жатақханаға қалай орналасамын?", "What's the weather tomorrow?"]:
        ok, p = model.is_university_question(q)
        print(f"  demo: {q!r:55s} -> university={ok} (p={p:.3f})")
    train_question_type()


def train_question_type():
    """Question-type classifier: what kind of answer is expected (steps / documents / deadline / amount ...)."""
    from ml.qtype import TYPES, QuestionTypeClassifier, generate_training_data as qtype_data
    banner("STEP 5b. Training the question-type classifier (softmax regression)")
    X, y = qtype_data()
    n_val = len(X) // 5
    print(f"  examples: train={len(X) - n_val} val={n_val}  classes={TYPES}")
    with Timer() as t:
        qt = QuestionTypeClassifier(epochs=60).fit(X[n_val:], y[n_val:], val=(X[:n_val], y[:n_val]))
    print(f"  trained in {t.sec:.1f}s")
    qt.save(MODELS / "qtype.pkl")
    h = qt.history
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot([r["epoch"] for r in h], [r["loss"] for r in h]); ax[0].set_title("Question type: loss")
    ax[1].plot([r["epoch"] for r in h], [r["train_acc"] for r in h], label="train")
    ax[1].plot([r["epoch"] for r in h], [r["val_acc"] for r in h], label="validation")
    ax[1].set_title("Question type: accuracy"); ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(FIGURES / "qtype_training.png", dpi=150)
    save_json(REPORTS / "qtype_metrics.json", {"classes": TYPES, "history": h})
    print("  saved: models/qtype.pkl, reports/figures/qtype_training.png")
    for q in ["ОБШАГАҒА ҚАНДАЙ ДОКУМЕНТТЕР КЕРЕК", "академка калай алам", "стипендия канша",
              "Когда начинается летний семестр?", "Куда обращаться по поводу перевода?", "Что такое GPA?"]:
        t_, p_ = qt.predict(q)
        print(f"  demo: {q!r:45s} -> {t_} (p={p_:.2f})")


if __name__ == "__main__":
    main()
