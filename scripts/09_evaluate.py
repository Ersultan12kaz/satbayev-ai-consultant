"""STEP 9 - Evaluation and experiments.

Test set: data/eval/questions.jsonl (300 = 10 categories x 10 questions x RU/KZ/EN),
          data/eval/out_of_scope.jsonl (30), data/eval/unanswerable.jsonl (15).

Experiment 1  Embedding A (LSA only) vs Embedding B (LSA + contrastive fine-tuning)
Experiment 2  Dense vs BM25 vs Hybrid (70% dense + 30% keyword) vs Hybrid + reranker
Experiment 3  Top-3 / Top-5 / Top-10
Experiment 4  Russian vs Kazakh vs English (accuracy, recall, faithfulness, hallucination,
              citation accuracy, latency)
Metrics: Recall@K, Precision@K, MRR, answer rate, citation accuracy, faithfulness,
         hallucination rate (answers given to unanswerable / off-topic questions),
         latency avg / P50 / P95.
Outputs: reports/evaluation_results.json, reports/evaluation_report.md,
         reports/per_question.csv (with an empty column for human evaluation),
         reports/figures/*.png
"""
from __future__ import annotations

import copy
import csv
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import EVAL, FIGURES, MODELS, REPORTS, Tee, banner, save_json

from ml.documents import read_jsonl
from ml.generator import supported_ratio
from ml.metrics import latency_stats, make_title_matcher, precision_at_k, recall_at_k, reciprocal_rank
from ml.rag import RAGPipeline

LANG_NAMES = {"ru": "Русский", "kk": "Қазақша", "en": "English"}


def retrieval_scores(pipe: RAGPipeline, qs, mode: str, k: int = 10, rerank: bool = False, use_lang=True):
    rows = []
    for q in qs:
        lang = q["lang"] if use_lang else None
        cands = pipe.retriever.search(q["question"], mode=mode, k=max(20, k), lang=lang)
        if rerank:
            cands = pipe.reranker.rerank(q["question"], cands, lang=lang)
        match = make_title_matcher(q["expected_sources"])
        rel = [1 if match(c.chunk["title"]) else 0 for c in cands[:k]]
        rows.append({"R@1": recall_at_k(rel, 1), "R@3": recall_at_k(rel, 3), "R@5": recall_at_k(rel, 5),
                     "R@10": recall_at_k(rel, 10), "P@3": precision_at_k(rel, 3), "P@5": precision_at_k(rel, 5),
                     "MRR": reciprocal_rank(rel), "lang": q["lang"]})
    return rows


def mean_of(rows, key, lang=None):
    v = [r[key] for r in rows if lang is None or r["lang"] == lang]
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else 0.0


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    Tee("09_evaluate")
    banner("STEP 9. Evaluation")
    pipe = RAGPipeline.load(MODELS)
    pipe.wiki.enabled = False            # evaluation = the university documents only
    pipe.llm = None                      # the retrieval + composer pipeline (the LM is evaluated in LM-5)
    qs = read_jsonl(EVAL / "questions.jsonl")
    oos = read_jsonl(EVAL / "out_of_scope.jsonl")
    una = read_jsonl(EVAL / "unanswerable.jsonl")
    print(f"  questions: {len(qs)} | out-of-scope: {len(oos)} | unanswerable: {len(una)}")
    results, md = {}, ["# Evaluation report\n", f"Questions: {len(qs)} (RU/KZ/EN), off-topic: {len(oos)}, "
                                                 f"unanswerable: {len(una)}\n"]

    # ---------------------------------------------------------------- Experiment 1
    banner("Experiment 1: Embedding A (LSA only) vs Embedding B (fine-tuned)")
    enc = pipe.retriever.encoder
    W_trained = enc.W.copy()
    base_pipe = copy.copy(pipe)
    enc.W = np.eye(enc.dim)
    emb_before = pipe.retriever.embeddings
    import ml.retriever as R
    rA = R.Retriever(pipe.retriever.chunks, enc.encode([f"{c.get('section', '')}. {c['text']}" for c in pipe.retriever.chunks]),
                     enc, pipe.retriever.bm25)
    base_pipe.retriever = rA
    rowsA = retrieval_scores(base_pipe, qs, "dense")
    enc.W = W_trained
    pipe.retriever.embeddings = emb_before
    rowsB = retrieval_scores(pipe, qs, "dense")
    exp1 = {"Embedding A (LSA)": {k: mean_of(rowsA, k) for k in ("R@1", "R@5", "R@10", "MRR")},
            "Embedding B (LSA + InfoNCE)": {k: mean_of(rowsB, k) for k in ("R@1", "R@5", "R@10", "MRR")}}
    for name, m in exp1.items():
        print(f"  {name:28s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    results["experiment_1_embeddings"] = exp1
    md += ["## Experiment 1 - embedding models (dense retrieval)\n",
           md_table(["Model", "Recall@1", "Recall@5", "Recall@10", "MRR"],
                    [[n] + [f"{m[k]:.3f}" for k in ("R@1", "R@5", "R@10", "MRR")] for n, m in exp1.items()]), ""]

    # ---------------------------------------------------------------- Experiment 2
    banner("Experiment 2: Dense vs BM25 vs Hybrid vs Hybrid + reranker")
    configs = {"Dense": ("dense", False), "BM25 (keyword)": ("bm25", False), "Hybrid 70/30": ("hybrid", False),
               "Hybrid + reranker": ("hybrid", True)}
    exp2, exp2_rows = {}, {}
    for name, (mode, rr) in configs.items():
        t0 = time.perf_counter()
        rows = retrieval_scores(pipe, qs, mode, rerank=rr)
        exp2_rows[name] = rows
        exp2[name] = {k: mean_of(rows, k) for k in ("R@1", "R@3", "R@5", "R@10", "P@5", "MRR")}
        exp2[name]["ms_per_query"] = (time.perf_counter() - t0) * 1000 / len(qs)
        print(f"  {name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in exp2[name].items()))
    results["experiment_2_retrieval"] = exp2
    md += ["## Experiment 2 - retrieval method\n",
           md_table(["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "Precision@5", "MRR", "ms/query"],
                    [[n] + [f"{m[k]:.3f}" for k in ("R@1", "R@3", "R@5", "R@10", "P@5", "MRR")] +
                     [f"{m['ms_per_query']:.1f}"] for n, m in exp2.items()]), ""]

    # ---------------------------------------------------------------- Experiment 3
    banner("Experiment 3: number of chunks (Top-K)")
    best = exp2_rows["Hybrid + reranker"]
    exp3 = {f"Top-{k}": {"Recall": mean_of(best, f"R@{k}"),
                         "Precision": mean_of(best, f"P@{k}") if f"P@{k}" in best[0] else None}
            for k in (3, 5, 10)}
    for k, v in exp3.items():
        print(f"  {k:7s} Recall={v['Recall']:.3f}" + (f" Precision={v['Precision']:.3f}" if v["Precision"] is not None else ""))
    results["experiment_3_topk"] = exp3
    md += ["## Experiment 3 - number of retrieved chunks\n",
           md_table(["Top-K", "Recall@K", "Precision@K"],
                    [[k, f"{v['Recall']:.3f}", "-" if v["Precision"] is None else f"{v['Precision']:.3f}"]
                     for k, v in exp3.items()]), ""]

    # ---------------------------------------------------------------- Experiment 4 (full pipeline)
    banner("Experiment 4: full RAG pipeline by language")
    per_q = []
    for q in qs:
        res = pipe.answer(q["question"], mode="hybrid", top_k=5)
        match = make_title_matcher(q["expected_sources"])
        rel = [1 if match(r["title"]) else 0 for r in res["retrieved"]]
        cited = res["sources"]
        cit_acc = (sum(1 for s in cited if match(s["title"])) / len(cited)) if cited else None
        ctx = [c["text"] for c in pipe.retriever.chunks if c.get("chunk_id") in {s["chunk_id"] for s in cited}]
        faith = supported_ratio(res["text"], ctx) if res["status"] == "answered" else None
        per_q.append({"id": q["id"], "category": q["category"], "lang": q["lang"], "question": q["question"],
                      "detected_lang": res["lang"], "status": res["status"], "confidence": res["confidence"],
                      "R@1": recall_at_k(rel, 1), "R@5": recall_at_k(rel, 5), "MRR": reciprocal_rank(rel),
                      "answered": 1.0 if res["status"] == "answered" else 0.0,
                      "correct_source_answer": 1.0 if (res["status"] == "answered" and cit_acc) else 0.0,
                      "citation_accuracy": cit_acc, "faithfulness": faith,
                      "latency_ms": res["timings"]["total_ms"],
                      "top_source": cited[0]["title"] if cited else "", "answer": res["text"]})
    halluc = {}
    for name, data, expected in (("off_topic", oos, "out_of_scope"), ("unanswerable", una, "not_found")):
        rows = []
        for q in data:
            res = pipe.answer(q["question"])
            rows.append({"lang": q["lang"], "rejected": 1.0 if res["status"] != "answered" else 0.0,
                         "status": res["status"], "question": q["question"]})
        halluc[name] = rows
        print(f"  {name:13s}: correctly refused {sum(r['rejected'] for r in rows):.0f}/{len(rows)}")

    table, exp4 = [], {}
    for lang in ("ru", "kk", "en", None):
        L = [r for r in per_q if lang is None or r["lang"] == lang]
        m = {
            "Answer accuracy (answered with correct source)": mean_of(L, "correct_source_answer"),
            "Recall@1": mean_of(L, "R@1"), "Recall@5": mean_of(L, "R@5"), "MRR": mean_of(L, "MRR"),
            "Answer rate": mean_of(L, "answered"),
            "Faithfulness": mean_of([r for r in L if r["faithfulness"] is not None], "faithfulness"),
            "Citation accuracy": mean_of([r for r in L if r["citation_accuracy"] is not None], "citation_accuracy"),
            "Language detection accuracy": float(np.mean([r["detected_lang"] == r["lang"] for r in L])),
            "Hallucination rate (unanswerable+off-topic answered)": 1 - float(np.mean(
                [x["rejected"] for k in halluc for x in halluc[k] if lang is None or x["lang"] == lang])),
        }
        m.update({f"Latency {k}": v for k, v in latency_stats([r["latency_ms"] for r in L]).items()})
        exp4[lang or "all"] = m
    metrics = list(exp4["ru"].keys())
    for k in metrics:
        table.append([k] + [f"{exp4[l][k]:.3f}" if "Latency" not in k else f"{exp4[l][k]:.0f}"
                            for l in ("ru", "kk", "en", "all")])
        print(f"  {k:55s} " + "  ".join(table[-1][1:]))
    results["experiment_4_languages"] = exp4
    results["refusals"] = {k: float(np.mean([r["rejected"] for r in v])) for k, v in halluc.items()}
    by_cat = {}
    for r in per_q:
        by_cat.setdefault(r["category"], []).append(r)
    results["by_category"] = {c: {"R@5": mean_of(v, "R@5"), "MRR": mean_of(v, "MRR"),
                                  "answer_accuracy": mean_of(v, "correct_source_answer")} for c, v in by_cat.items()}
    md += ["## Experiment 4 - languages (full pipeline: hybrid + reranker, Top-5)\n",
           md_table(["Metric", "Русский", "Қазақша", "English", "All"], table), "",
           "## Refusals (hallucination guard)\n",
           md_table(["Set", "Correctly refused"], [[k, f"{v:.3f}"] for k, v in results["refusals"].items()]), "",
           "## By category\n",
           md_table(["Category", "Recall@5", "MRR", "Answer accuracy"],
                    [[c, f"{v['R@5']:.3f}", f"{v['MRR']:.3f}", f"{v['answer_accuracy']:.3f}"]
                     for c, v in results["by_category"].items()]), ""]

    save_json(REPORTS / "evaluation_results.json", results)
    (REPORTS / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")
    with open(REPORTS / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        cols = ["id", "category", "lang", "question", "status", "confidence", "R@5", "MRR", "citation_accuracy",
                "faithfulness", "latency_ms", "top_source", "answer"]
        w.writerow(cols + ["human_score_0_2"])
        for r in per_q:
            w.writerow([r[c] if not isinstance(r[c], float) else round(r[c], 4) for c in cols] + [""])

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(exp2.keys()); x = np.arange(len(names))
    for i, k in enumerate(("R@1", "R@5", "MRR")):
        ax.bar(x + (i - 1) * 0.25, [exp2[n][k] for n in names], width=0.25, label=k)
    ax.set_xticks(x, names, fontsize=8); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Retrieval methods")
    fig.tight_layout(); fig.savefig(FIGURES / "exp2_retrieval.png", dpi=150)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ks = ["Recall@5", "MRR", "Answer rate", "Citation accuracy", "Faithfulness"]
    x = np.arange(len(ks))
    for i, l in enumerate(("ru", "kk", "en")):
        ax.bar(x + (i - 1) * 0.25, [exp4[l][k] for k in ks], width=0.25, label=LANG_NAMES[l])
    ax.set_xticks(x, ks, fontsize=8); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Quality by language")
    fig.tight_layout(); fig.savefig(FIGURES / "exp4_languages.png", dpi=150)
    print("\n  saved: reports/evaluation_report.md, reports/evaluation_results.json, reports/per_question.csv")


if __name__ == "__main__":
    main()
