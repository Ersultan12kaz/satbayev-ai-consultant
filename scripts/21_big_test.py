"""STEP 21 - Big automatic test: 1 100+ questions (kk / ru / en / mixed kk-ru / without Kazakh letters / typos /
off-topic), data/eval/big_test.json.

For every question the assistant (the same pipeline as the web server) answers, and we check automatically:
  * off-topic questions are refused;
  * a university question gets an answer (not "not found");
  * the answer is in the language of the question;
  * the answer is about the topic (it contains a keyword of the topic, e.g. "жатақ|общежит|dormitor");
  * the answer has no garbage (table debris, broken mixed-script words).
Result: reports/big_test.md (tables by topic and by question style + every failed question) and big_test.json.

    python scripts/21_big_test.py [--limit 0] [--no-wiki]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from _common import EVAL, MODELS, REPORTS, Tee, banner, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.composer import natural_text  # noqa: E402
from ml.langdetect import heuristic_language  # noqa: E402
from ml.rag import RAGPipeline  # noqa: E402

MIXED = re.compile(r"[a-z]{2,}[а-яёәғқңөұүһі]{2,}|[а-яёәғқңөұүһі]{2,}[a-z]{2,}", re.I)
REFUSED = ("out_of_scope",)


def check(item, r):
    text = r.get("text", "")
    st = r.get("status")
    if item["topic"] == "off_topic":
        return {"ok": st in REFUSED or r.get("generator") == "rules", "why": "" if st in REFUSED else f"not refused ({st})"}
    probs = []
    if st != "answered":
        probs.append(f"no answer ({st})")
    else:
        body = re.sub(r"\((Құжат басқа тілде|Документ написан|The document is in|Университет құжаттарында|В документах)[^)]*\)", "", text)
        al = heuristic_language(body)
        if item["lang"] and al != item["lang"]:
            probs.append(f"language {al} instead of {item['lang']}")
        if item["kw"] and not re.search(item["kw"], text, re.I):
            probs.append("off-topic answer")
        if text.count("|") >= 2 or MIXED.search(text):
            probs.append("garbage")
    return {"ok": not probs, "why": "; ".join(probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-wiki", action="store_true")
    args = ap.parse_args()
    Tee("21_big_test")
    banner("STEP 21. Big automatic test (1 100+ questions)")
    items = json.loads((EVAL / "big_test.json").read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    pipe = RAGPipeline.load(MODELS)
    if args.no_wiki:
        pipe.wiki.enabled = False
    rows, t0 = [], time.time()
    for i, it in enumerate(items, 1):
        try:
            r = pipe.answer(it["q"])
            r["text"] = natural_text(r["text"], r["lang"])
        except Exception as e:                                   # a crash is a failed question, not a stop
            r = {"status": "error", "text": f"ERROR {e}", "generator": "", "lang": ""}
        c = check(it, r)
        rows.append({**it, "status": r.get("status"), "generator": r.get("generator"), "answer": r.get("text", "")[:400],
                     "ok": c["ok"], "why": c["why"]})
        if i % 50 == 0:
            ok = sum(x["ok"] for x in rows)
            print(f"  {i}/{len(items)}  passed {ok} ({100 * ok / i:.1f}%)  {time.time() - t0:.0f}s", flush=True)

    def table(key):
        agg = defaultdict(lambda: [0, 0])
        for x in rows:
            agg[x[key]][0] += x["ok"]; agg[x[key]][1] += 1
        return sorted(agg.items(), key=lambda kv: kv[1][0] / kv[1][1])

    ok = sum(x["ok"] for x in rows)
    reasons = defaultdict(int)
    for x in rows:
        for w in filter(None, x["why"].split("; ")):
            reasons[re.sub(r"\(.*\)", "", w).strip()] += 1
    md = [f"# Big test — {len(rows)} questions", "",
          f"**Passed: {ok}/{len(rows)} ({100 * ok / len(rows):.1f}%)**", "",
          "| Reason of failure | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    md += ["", "## By question style", "", "| Style | Passed |", "|---|---|"] + \
          [f"| {k} | {a}/{b} ({100 * a / b:.0f}%) |" for k, (a, b) in table("style")]
    md += ["", "## By topic", "", "| Topic | Passed |", "|---|---|"] + \
          [f"| {k} | {a}/{b} ({100 * a / b:.0f}%) |" for k, (a, b) in table("topic")]
    md += ["", "## Failed questions", "", "| Question | Topic | Problem | Answer (start) |", "|---|---|---|---|"]
    for x in rows:
        if not x["ok"]:
            md.append(f"| {x['q']} | {x['topic']} | {x['why']} | {x['answer'][:120].replace(chr(10), ' ').replace('|', '/')} |")
    (REPORTS / "big_test.md").write_text("\n".join(md), encoding="utf-8")
    save_json(REPORTS / "big_test.json", {"passed": ok, "total": len(rows), "rows": rows})
    banner("Summary")
    print(f"  passed {ok}/{len(rows)} ({100 * ok / len(rows):.1f}%) -> reports/big_test.md")


if __name__ == "__main__":
    main()
