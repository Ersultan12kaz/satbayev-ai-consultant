"""STEP 7b - Facts from signature blocks: who is the rector / pro-rectors (ml/facts.py).

"Ректор кім?" is not written as a sentence in the regulations, only in signature blocks
("Председатель Правления - Ректор ________ М. Бегентаев").  We extract such (role -> person)
pairs from all pages, vote over the occurrences and save models/facts.json.
"""
from __future__ import annotations

import json

from _common import MODELS, PROCESSED, RAW, Tee, banner

from ml.facts import FactBase, extract_facts, extract_lists


def main():
    Tee("07b_facts")
    banner("STEP 7b. Extracting facts (rector, pro-rectors) from the documents")
    meta = {d["doc_id"]: d for d in json.loads((PROCESSED / "documents.json").read_text(encoding="utf-8"))}

    def pages():
        with open(PROCESSED / "pages.jsonl", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                m = meta.get(p["doc_id"], {})
                p["url"] = m.get("url", p.get("url", ""))
                p["title"] = m.get("title", p.get("title", ""))
                yield p

    facts = extract_facts(pages())
    manifest = [json.loads(l) for l in (RAW / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    facts["_lists"] = extract_lists(manifest)
    for kind, by_lang in facts["_lists"].items():
        print(f"  {kind}: " + ", ".join(f"{l}={len(v)}" for l, v in by_lang.items()))
    FactBase(facts).save(MODELS / "facts.json")
    for role, v in facts.items():
        if role.startswith("_"):
            continue
        print(f"  {v['role_text'][:60]:60s} -> {v['name']}  (found {v['count']}x, e.g. {v['sources'][0]['title'][:50]})")
    print("  saved: models/facts.json")


if __name__ == "__main__":
    main()
