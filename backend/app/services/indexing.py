"""Keeps the vector index, the models/index files and the database in sync.
Used by scripts/07_build_index.py --db and by the admin document upload."""
import hashlib
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import delete, select

from ml.documents import Page, chunk_pages, extract_document, is_scanned
from ml.retriever import Retriever

from ..core.config import get_settings
from ..core.database import SessionLocal, use_pgvector
from ..models import Document, DocumentChunk

settings = get_settings()


def _vec(r: Retriever, i: int):
    return r.embeddings[i].astype(float).tolist()


def sync_index_to_db(r: Retriever, replace: bool = False) -> int:
    """Write documents + chunks (+ vectors) of the index to the database."""
    with SessionLocal() as db:
        if replace:
            db.execute(delete(DocumentChunk))
            db.execute(delete(Document))
            db.commit()
        existing = {d.doc_id: d for d in db.scalars(select(Document))}
        existing_chunks = set(db.scalars(select(DocumentChunk.chunk_id)))
        by_doc: Dict[str, List[int]] = {}
        for i, c in enumerate(r.chunks):
            by_doc.setdefault(c["doc_id"], []).append(i)
        n = 0
        for doc_id, idxs in by_doc.items():
            c0 = r.chunks[idxs[0]]
            doc = existing.get(doc_id)
            if doc is None:
                doc = Document(doc_id=doc_id, title=c0["title"], language=c0.get("lang", "ru"),
                               category=c0.get("category", ""), source_url=c0.get("url", ""),
                               uploaded_by=c0.get("uploaded_by", "pipeline"))
                db.add(doc)
                db.flush()
            doc.n_chunks = len(idxs)
            for i in idxs:
                c = r.chunks[i]
                if c["chunk_id"] in existing_chunks:
                    continue
                db.add(DocumentChunk(chunk_id=c["chunk_id"], document_id=doc.id, page=c["page"],
                                     section=c.get("section", ""), language=c.get("lang", "ru"), text=c["text"],
                                     embedding=_vec(r, i) if use_pgvector() else None))
                n += 1
        db.commit()
    return n


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name).strip()
    return name[:150] or "document"


def ingest_file(r: Retriever, src_path: Path, title: str, lang: Optional[str], category: str,
                uploaded_by: str, detect_lang) -> Dict:
    """Admin upload: extract -> clean -> chunk -> embed -> add to index + DB."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    pages = extract_document(src_path)
    if is_scanned(pages):
        raise ValueError("The file has no text layer (scanned document) or is empty")
    with SessionLocal() as db:
        n_uploaded = len(list(db.scalars(select(Document.id).where(Document.doc_id.like("upload_%")))))
    doc_id = f"upload_{n_uploaded + 1}_{hashlib.md5(title.encode()).hexdigest()[:6]}"
    lang = lang or detect_lang(" ".join(pages)[:3000])
    page_objs = [Page(doc_id=doc_id, title=title, lang=lang, category=category, url="", page=i, text=t)
                 for i, t in enumerate(pages, start=1) if len(t) >= 30]
    chunks = [c.__dict__ for c in chunk_pages(page_objs, min_words=5)]
    if not chunks:
        raise ValueError("No text chunks could be extracted from the file")
    for c in chunks:
        c.pop("extra", None)
        c["uploaded_by"] = uploaded_by
    r.add_chunks(chunks)
    r.save(Path(settings.MODELS_DIR) / "index")
    sync_index_to_db(r)
    shutil.copy(src_path, upload_dir / f"{doc_id}{src_path.suffix.lower()}")
    return {"doc_id": doc_id, "title": title, "language": lang, "pages": len(pages), "chunks": len(chunks)}


def delete_document(r: Retriever, doc_id: str) -> int:
    n = r.remove_document(doc_id)
    r.save(Path(settings.MODELS_DIR) / "index")
    with SessionLocal() as db:
        doc = db.scalar(select(Document).where(Document.doc_id == doc_id))
        if doc:
            db.delete(doc)
            db.commit()
    return n
