"""STEP 4 - Train the language identification model (RU / KZ / EN).

Training data: sentences of the chunks, label = language of the document.
Split by document (no document is both in train and test).
Also evaluated on the short questions of the evaluation set (real use-case).
"""
from __future__ import annotations

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import EVAL, FIGURES, MODELS, PROCESSED, REPORTS, Tee, Timer, banner, save_json

from ml.documents import read_jsonl
from ml.langdetect import LANGS, LanguageDetector, evaluate, heuristic_language
from ml.text import split_sentences, tokenize


def main():
    Tee("04_langdetect")
    banner("STEP 4. Training the language detector (character n-gram Naive Bayes)")
    rnd = random.Random(0)
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    docs = sorted({c["doc_id"] for c in chunks})
    rnd.shuffle(docs)
    test_docs = set(docs[: max(1, len(docs) // 5)])
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for c in chunks:
        for s in split_sentences(c["text"]):
            if 3 <= len(tokenize(s)) <= 40:
                (te_x if c["doc_id"] in test_docs else tr_x).append(s)
                (te_y if c["doc_id"] in test_docs else tr_y).append(c["lang"])
    # balance classes (max 6000 sentences per language)
    by = {l: [i for i, y in enumerate(tr_y) if y == l] for l in LANGS}
    keep = []
    for l, idx in by.items():
        rnd.shuffle(idx)
        keep += idx[:6000]
    tr_x = [tr_x[i] for i in keep]; tr_y = [tr_y[i] for i in keep]
    print(f"  train sentences: {len(tr_x)}  test sentences: {len(te_x)}")
    print("  per language (train):", {l: tr_y.count(l) for l in LANGS})

    with Timer() as t:
        model = LanguageDetector(n_max=3).fit(tr_x, tr_y)
    print(f"  trained in {t.sec:.1f}s, vocabulary of {len(model.vocab)} character n-grams")

    acc, conf = evaluate(model, te_x, te_y)
    base_acc = sum(heuristic_language(x) == y for x, y in zip(te_x, te_y)) / max(1, len(te_x))
    print(f"\n  test accuracy (sentences): model={acc:.4f} | rule-based baseline={base_acc:.4f}")
    print("  confusion matrix (true -> predicted):", conf)

    q_metrics = {}
    qfile = EVAL / "questions.jsonl"
    if qfile.exists():
        qs = read_jsonl(qfile)
        qa, qc = evaluate(model, [q["question"] for q in qs], [q["lang"] for q in qs])
        qb = sum(heuristic_language(q["question"]) == q["lang"] for q in qs) / len(qs)
        q_metrics = {"accuracy": qa, "baseline_accuracy": qb, "confusion": qc, "n": len(qs)}
        print(f"  accuracy on {len(qs)} evaluation QUESTIONS: model={qa:.4f} | baseline={qb:.4f}")

    model.save(MODELS / "langdetect.json")
    save_json(REPORTS / "langdetect_metrics.json", {"sentence_accuracy": acc, "baseline_accuracy": base_acc,
                                                    "confusion": conf, "questions": q_metrics,
                                                    "n_train": len(tr_x), "n_test": len(te_x)})
    # confusion matrix figure
    labels = list(LANGS)
    M = [[conf.get(a, {}).get(b, 0) for b in labels] for a in labels]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, M[i][j], ha="center", va="center")
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Language detector, acc={acc:.3f}")
    fig.tight_layout(); fig.savefig(FIGURES / "langdetect_confusion.png", dpi=150)
    print(f"\n  saved: models/langdetect.json, reports/figures/langdetect_confusion.png")
    for s in ["Какие документы нужны для академического отпуска?", "Жатақханаға қандай құжаттар қажет?",
              "What documents are required to apply for a dormitory?"]:
        print(f"  demo: {s!r} -> {model.predict(s)}  {model.predict_proba(s)}")


if __name__ == "__main__":
    main()
