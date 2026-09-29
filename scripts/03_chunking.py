"""STEP 3 - Chunking + metadata.

pages.jsonl -> sentence-aware chunks (~160 words, 35 words overlap) with metadata:
{chunk_id, doc_id, title, lang, category, url, page, section, text}
Output: data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter

from _common import PROCESSED, Tee, banner, load_json, save_json

from ml.documents import Page, chunk_pages, read_jsonl, write_jsonl
from ml.langdetect import heuristic_language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=160)
    ap.add_argument("--overlap", type=int, default=35)
    args = ap.parse_args()
    Tee("03_chunking")
    banner("STEP 3. Chunking")
    pages = [Page(**p) for p in read_jsonl(PROCESSED / "pages.jsonl")]
    docs = {d["doc_id"]: d for d in load_json(PROCESSED / "documents.json")}
    # language label cleaning: the website section language vs. the real language of the text
    fixed = 0
    for pg in pages:
        real = docs.get(pg.doc_id, {}).get("heuristic_lang", pg.lang)
        if real != pg.lang:
            pg.lang = real
            fixed += 1
    print(f"  pages: {len(pages)}  (language label corrected on {fixed} pages)")

    seen, chunks, dups = set(), [], 0
    for ch in chunk_pages(pages, max_words=args.max_words, overlap_words=args.overlap):
        h = hashlib.md5(ch.text.lower().encode("utf-8")).hexdigest()
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        chunks.append(ch)
    write_jsonl(PROCESSED / "chunks.jsonl", chunks)
    lens = [c.n_words for c in chunks]
    stats = {
        "chunks": len(chunks), "duplicates_removed": dups,
        "by_language": dict(Counter(c.lang for c in chunks)),
        "by_category": dict(Counter(c.category for c in chunks)),
        "documents": len({c.doc_id for c in chunks}),
        "avg_words": round(sum(lens) / max(1, len(lens)), 1),
        "with_section": sum(1 for c in chunks if c.section),
    }
    save_json(PROCESSED / "chunk_stats.json", stats)
    banner("Summary")
    for k, v in stats.items():
        print(f"  {k:18s}: {v}")
    if chunks:
        c = chunks[len(chunks) // 2]
        print("\n  Example chunk:")
        print(f"    title  : {c.title}\n    page   : {c.page}\n    section: {c.section}\n    lang   : {c.lang}")
        print(f"    text   : {c.text[:400]}...")


if __name__ == "__main__":
    main()
