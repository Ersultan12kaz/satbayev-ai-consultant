"""STEP 2 - Text extraction and cleaning.

PDF / DOCX  ->  text of every page  ->  cleaning (hyphenation, running headers,
page numbers, repeated lines)  ->  data/processed/pages.jsonl

Scanned PDFs (no text layer) and PDFs with a broken text layer are OCR-ed (Tesseract).
Old .doc / .xls files are read with our own parser (ml/msdoc.py).

    python scripts/02_extract_text.py                # all documents
    python scripts/02_extract_text.py --incremental  # keep good documents, redo only the problem ones
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from _common import PROCESSED, RAW, Tee, banner, save_json

from ml.documents import Page, broken_kazakh, extract_document, is_scanned, read_jsonl, write_jsonl
from ml.langdetect import heuristic_language
from ml.ocr import available_langs, ocr_available


def process(rec):
    path = RAW / rec["path"]
    try:
        pages = extract_document(path, lang=rec["lang"])
    except Exception as e:
        return rec, None, f"error: {e}"
    if path.suffix.lower() != ".pdf":
        if sum(len(p.strip()) for p in pages) >= 30:        # short forms (.doc/.docx/.xls) are still documents
            return rec, [p for p in pages if p.strip()] or pages, "ok"
        return rec, None, "empty document"
    if is_scanned(pages):
        return rec, None, "scanned (no text layer)"
    return rec, pages, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true",
                    help="reuse data/processed/pages.jsonl, re-extract only skipped / broken documents")
    args = ap.parse_args()
    Tee("02_extract")
    banner("STEP 2. Text extraction + cleaning")
    manifest = [json.loads(l) for l in (RAW / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  documents in manifest: {len(manifest)}")
    kept_pages, kept_docs = [], []
    if args.incremental and (PROCESSED / "pages.jsonl").exists() and (PROCESSED / "documents.json").exists():
        old_pages = read_jsonl(PROCESSED / "pages.jsonl")
        by_doc = {}
        for p in old_pages:
            by_doc.setdefault(p["doc_id"], []).append(p)
        broken = {d for d, ps in by_doc.items() if broken_kazakh(" ".join(p["text"] for p in ps)[:20000])}
        old_docs = {d["doc_id"]: d for d in json.loads((PROCESSED / "documents.json").read_text(encoding="utf-8"))}
        stamp = (PROCESSED / "pages.jsonl").stat().st_mtime
        manifest_by_id = {r["doc_id"]: r for r in manifest}
        changed = {d for d, r in manifest_by_id.items()                 # re-downloaded after the last extraction
                   if (RAW / r["path"]).exists() and (RAW / r["path"]).stat().st_mtime > stamp}
        good = {d for d in old_docs if d in by_doc and d not in broken and d not in changed}
        kept_pages = [Page(**p) for d in sorted(good) for p in by_doc[d]]
        kept_docs = [old_docs[d] for d in sorted(good)]
        manifest = [r for r in manifest if r["doc_id"] not in good]
        print(f"  incremental: keep {len(good)} documents, re-extract {len(manifest)} "
              f"(skipped earlier or new: {len(manifest) - len(broken)}, broken Kazakh font encoding: {len(broken)})")
    if ocr_available():
        print(f"  OCR for scanned PDFs: ON (Tesseract languages: {sorted(available_langs())})")
    else:
        print("  OCR for scanned PDFs: OFF (install with 6_ocr_scanned_documents.bat) - scanned files are skipped")
    all_pages, docs, problems = list(kept_pages), list(kept_docs), []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(process, r) for r in manifest]
        for i, fut in enumerate(as_completed(futures), 1):
            rec, pages, status = fut.result()
            if pages is None:
                problems.append({"doc_id": rec["doc_id"], "title": rec["title"], "status": status})
            else:
                n_chars = 0
                for pno, text in enumerate(pages, start=1):
                    if len(text) < 30:
                        continue
                    n_chars += len(text)
                    all_pages.append(Page(doc_id=rec["doc_id"], title=rec["title"], lang=rec["lang"],
                                          category=rec["category"], url=rec["url"], page=pno, text=text))
                sample = " ".join(pages)[:3000]
                docs.append({**rec, "n_pages": len(pages), "n_chars": n_chars,
                             "heuristic_lang": heuristic_language(sample)})
            if i % 20 == 0 or i == len(futures):
                print(f"  processed {i}/{len(futures)}")
    all_pages.sort(key=lambda p: (p.doc_id, p.page))
    n = write_jsonl(PROCESSED / "pages.jsonl", all_pages)
    save_json(PROCESSED / "documents.json", sorted(docs, key=lambda d: d["doc_id"]))
    save_json(PROCESSED / "extraction_problems.json", problems)
    banner("Summary")
    print(f"  documents with text : {len(docs)}")
    print(f"  pages extracted     : {n}")
    print(f"  skipped (scanned / errors): {len(problems)}")
    by_lang = {}
    for d in docs:
        by_lang[d["lang"]] = by_lang.get(d["lang"], 0) + 1
    print(f"  by language: {by_lang}")
    mism = [d for d in docs if d["heuristic_lang"] != d["lang"]]
    if mism:
        print(f"  note: {len(mism)} documents are published in a section of another language "
              f"(e.g. a Russian file on the /kk/ page); their language label is corrected in step 3.")


if __name__ == "__main__":
    main()
