"""STEP 8 - Train the reranker (learning to rank) and choose the "not found" threshold.

For automatically generated training questions we retrieve the Top-20 hybrid
candidates, compute features and label the candidate that contains the answer
as 1 (others 0).  Off-topic questions give examples where *nothing* is relevant,
which calibrates the probability used for the hallucination guard.
"""
from __future__ import annotations

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import FIGURES, MODELS, PROCESSED, REPORTS, Tee, Timer, banner, save_json

from ml.documents import read_jsonl
from ml.embeddings import SubwordEncoder
from ml.langdetect import LanguageDetector
from ml.lexicon import expand_query
from ml.reranker import FEATURES, Reranker, features
from ml.retriever import Retriever
from ml.scope import OFF_PATTERNS, OFF_TOPICS
from ml.training_data import make_pairs


def build(retriever, queries, lang_model, pool=20):
    X, y, groups = [], [], []
    qvecs = retriever.encoder.encode([expand_query(q) for q, _ in queries])   # one batch instead of 1 call per question
    for gi, (q, pos_idx) in enumerate(queries):
        if gi and gi % 1000 == 0:
            print(f"    features for {gi}/{len(queries)} questions", flush=True)
        lang = lang_model.predict(q) if lang_model else None
        cands = retriever.search(q, mode="hybrid", k=pool, lang=lang, qvec=qvecs[gi])
        if not cands:
            continue
        F = features(q, cands, lang)
        for c, f in zip(cands, F):
            X.append(f); y.append(1 if c.idx == pos_idx else 0); groups.append(gi)
    return np.array(X), np.array(y, dtype=np.float64), np.array(groups)


def mrr(scores, y, groups):
    rr, hit1 = [], []
    for g in np.unique(groups):
        m = groups == g
        if y[m].sum() == 0:
            continue
        order = np.argsort(-scores[m])
        rank = int(np.where(y[m][order] == 1)[0][0])
        rr.append(1 / (rank + 1)); hit1.append(rank == 0)
    return float(np.mean(rr)) if rr else 0.0, float(np.mean(hit1)) if hit1 else 0.0


def main():
    Tee("08_reranker")
    banner("STEP 8. Training the reranker")
    rnd = random.Random(5)
    enc = SubwordEncoder.load(MODELS / "embedder.pkl")
    r = Retriever.load(MODELS / "index", encoder=enc)
    lang_model = LanguageDetector.load(MODELS / "langdetect.json")
    id2idx = {c["chunk_id"]: i for i, c in enumerate(r.chunks)}
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    pairs = make_pairs(chunks, max_pairs=6000, seed=99)  # a different sample than for the embedder
    queries = [(q, id2idx.get(chunks[ci]["chunk_id"], -1)) for q, _, ci in pairs]
    rnd.shuffle(queries)
    n_test = len(queries) // 5
    off = [p.format(t=t) for lang in OFF_TOPICS for t in OFF_TOPICS[lang] for p in OFF_PATTERNS[lang]]
    rnd.shuffle(off)
    train_q = queries[n_test:] + [(q, -1) for q in off[: len(off) // 2]]
    test_q = queries[:n_test]
    print(f"  training questions: {len(train_q)}  test questions: {len(test_q)}")
    with Timer() as t:
        Xtr, ytr, gtr = build(r, train_q, lang_model)
        Xte, yte, gte = build(r, test_q, lang_model)
    print(f"  feature matrix: train {Xtr.shape}, test {Xte.shape}  ({t.sec:.1f}s)")

    model = Reranker().fit(Xtr, ytr)
    hyb_idx = FEATURES.index("hybrid")
    before = mrr(Xte[:, hyb_idx], yte, gte)
    after = mrr(model.predict_proba(Xte), yte, gte)
    print(f"\n  Test MRR / Hit@1  hybrid only : {before[0]:.4f} / {before[1]:.4f}")
    print(f"  Test MRR / Hit@1  + reranker  : {after[0]:.4f} / {after[1]:.4f}")
    print("\n  Learned feature weights (standardised):")
    for k, v in sorted(model.weights().items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:16s} {v:+.3f}")

    # threshold for "information not found": top probability for answerable vs off-topic questions
    top_ans = [model.predict_proba(Xte[gte == g]).max() for g in np.unique(gte) if yte[gte == g].sum() > 0]
    Xo, yo, go = build(r, [(q, -1) for q in off[len(off) // 2:]], lang_model)
    top_off = [model.predict_proba(Xo[go == g]).max() for g in np.unique(go)] if len(Xo) else []
    best_t, best_f = 0.3, -1
    for thr in np.arange(0.05, 0.9, 0.01):
        tp = sum(p >= thr for p in top_ans); fn = len(top_ans) - tp
        fp = sum(p >= thr for p in top_off); tn = len(top_off) - fp
        # we care more about not hallucinating: weight precision a bit higher (F0.5)
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f = 1.25 * prec * rec / max(1e-9, 0.25 * prec + rec)
        if f > best_f:
            best_f, best_t = f, float(thr)
    best_t = min(best_t, 0.5)
    print(f"\n  chosen 'not found' threshold = {best_t:.2f} (F0.5={best_f:.3f})")
    model.save(MODELS / "reranker.json")
    # guard configuration used by the RAG pipeline (see ml/rag.py): reranker probability threshold,
    # scope-classifier threshold and minimal share of question words supported by the evidence
    save_json(MODELS / "config.json", {"not_found_threshold": 0.3, "scope_threshold": 0.3, "min_overlap": 0.5,
                                       "reranker_f05_threshold": round(best_t, 3)})
    save_json(REPORTS / "reranker_metrics.json", {"mrr_hybrid": before[0], "hit1_hybrid": before[1],
                                                  "mrr_reranked": after[0], "hit1_reranked": after[1],
                                                  "weights": model.weights(), "not_found_threshold": best_t,
                                                  "history": model.history})
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].plot([h["epoch"] for h in model.history], [h["loss"] for h in model.history])
    ax[0].set_title("Reranker training loss"); ax[0].set_xlabel("epoch")
    fin = lambda v: [x for x in v if np.isfinite(x)] or [0.0]
    ax[1].hist(fin(top_ans), bins=20, alpha=.6, label="answerable")
    ax[1].hist(fin(top_off), bins=20, alpha=.6, label="off-topic")
    ax[1].axvline(best_t, color="k", ls="--", label="threshold"); ax[1].legend()
    ax[1].set_title("Top reranker probability")
    fig.tight_layout(); fig.savefig(FIGURES / "reranker.png", dpi=150)
    print("  saved: models/reranker.json, models/config.json, reports/figures/reranker.png")
    train_sentence_ranker(r, chunks)


def train_sentence_ranker(retriever, chunks):
    """Sentence ranker of the answer composer: which sentence of a fragment answers the question."""
    from ml.composer import SENT_FEATURES, SentenceRanker, make_sentence_training
    from ml.qtype import QuestionTypeClassifier
    banner("STEP 8b. Training the answer sentence ranker (logistic regression)")
    qt = QuestionTypeClassifier.load(MODELS / "qtype.pkl") if (MODELS / "qtype.pkl").exists() else None
    with Timer() as t:
        X, y, g = make_sentence_training(chunks, retriever, qt, retriever.encoder, n=3000)
    print(f"  pseudo-questions: {len(np.unique(g))}, candidate sentences: {len(y)} ({t.sec:.1f}s)")
    test = np.unique(g)[: len(np.unique(g)) // 5]
    te = np.isin(g, test)
    model = SentenceRanker().fit(X[~te], y[~te])
    base = SentenceRanker()                       # hand-made weights before training
    b = mrr(base.predict_proba(X[te]), y[te], g[te]); a = mrr(model.predict_proba(X[te]), y[te], g[te])
    print(f"\n  Test MRR / Hit@1  hand-made weights : {b[0]:.4f} / {b[1]:.4f}")
    print(f"  Test MRR / Hit@1  trained ranker    : {a[0]:.4f} / {a[1]:.4f}")
    print("  Learned feature weights (standardised):")
    for k, v in sorted(model.weights().items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:16s} {v:+.3f}")
    model = SentenceRanker().fit(X, y, verbose=False)          # final model on all data
    model.save(MODELS / "sentence_ranker.json")
    save_json(REPORTS / "sentence_ranker_metrics.json", {"features": SENT_FEATURES, "mrr_before": b[0],
                                                         "hit1_before": b[1], "mrr_after": a[0], "hit1_after": a[1],
                                                         "weights": model.weights()})
    print("  saved: models/sentence_ranker.json")


if __name__ == "__main__":
    main()
