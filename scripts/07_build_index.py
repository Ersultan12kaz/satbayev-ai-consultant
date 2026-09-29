"""STEP 7 - Build the vector index (embeddings of all chunks) and the BM25 index.

Output: models/index/{chunks.json, embeddings.npy, bm25.pkl}
With --db the documents and chunks (with their vectors) are also written into the
database from DATABASE_URL (PostgreSQL + pgvector in Docker, or SQLite locally).
"""
from __future__ import annotations

import argparse

from _common import MODELS, PROCESSED, Tee, Timer, banner

from ml.documents import read_jsonl
from ml.embeddings import SubwordEncoder
from ml.retriever import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true", help="also load documents/chunks into the database")
    args = ap.parse_args()
    Tee("07_index")
    banner("STEP 7. Building the vector + keyword index")
    chunks = read_jsonl(PROCESSED / "chunks.jsonl")
    enc = SubwordEncoder.load(MODELS / "embedder.pkl")
    with Timer() as t:
        r = Retriever.build(chunks, enc)
    r.save(MODELS / "index")
    print(f"  {len(chunks)} chunks -> embeddings {r.embeddings.shape}, BM25 vocabulary {len(r.bm25.vocab)}"
          f"  ({t.sec:.1f}s)")
    for q in ["Как оформить академический отпуск?", "Жатақханада орын бөлу тәртібі", "How is the GPA calculated?"]:
        res = r.search(q, mode="hybrid", k=3)
        print(f"\n  query: {q}")
        for c in res:
            print(f"    {c.hybrid:.3f} | {c.chunk['title'][:70]} (p.{c.chunk['page']}, {c.chunk['lang']})")
    if args.db:
        from backend.app.core.database import init_db
        from backend.app.services.indexing import sync_index_to_db
        init_db()
        n = sync_index_to_db(r, replace=True)
        print(f"\n  database: {n} chunks written")
    print("\n  saved: models/index/")


if __name__ == "__main__":
    main()
