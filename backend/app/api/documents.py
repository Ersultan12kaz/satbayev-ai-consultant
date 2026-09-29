"""Documents: list / view (everyone), upload / delete (admin)."""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import require_admin
from ..models import Document, DocumentChunk, User
from ..services.indexing import delete_document, ingest_file
from ..services.rag import rag_service

router = APIRouter(prefix="/api/documents", tags=["documents"])
ALLOWED = {".pdf", ".docx", ".txt"}
MAX_BYTES = 30 * 1024 * 1024


@router.get("")
def list_documents(language: Optional[str] = None, q: Optional[str] = None, db: Session = Depends(get_db)):
    stmt = select(Document).order_by(Document.language, Document.title)
    if language:
        stmt = stmt.where(Document.language == language)
    if q:
        stmt = stmt.where(Document.title.ilike(f"%{q}%"))
    return [{"id": d.id, "doc_id": d.doc_id, "title": d.title, "language": d.language, "category": d.category,
             "source_url": d.source_url, "n_chunks": d.n_chunks, "created_at": d.created_at}
            for d in db.scalars(stmt)]


@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    d = db.scalar(select(Document).where(Document.doc_id == doc_id))
    if not d:
        raise HTTPException(404, "Document not found")
    chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == d.id)
                        .order_by(DocumentChunk.page, DocumentChunk.id)).all()
    return {"doc_id": d.doc_id, "title": d.title, "language": d.language, "category": d.category,
            "source_url": d.source_url,
            "chunks": [{"chunk_id": c.chunk_id, "page": c.page, "section": c.section, "text": c.text} for c in chunks]}


@router.post("")
async def upload_document(file: UploadFile = File(...), title: str = Form(...), language: Optional[str] = Form(None),
                          category: str = Form("admin-upload"), admin: User = Depends(require_admin)):
    if not rag_service.ready:
        raise HTTPException(503, rag_service.error or "Models not loaded")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Allowed types: {', '.join(sorted(ALLOWED))}")
    if language not in (None, "", "ru", "kk", "en"):
        raise HTTPException(400, "language must be ru, kk or en")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "File is too large (max 30 MB)")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"upload{suffix}"
        p.write_bytes(data)
        try:
            with rag_service.lock:
                info = ingest_file(rag_service.pipeline.retriever, p, title.strip()[:300], language or None,
                                   category, admin.email, rag_service.detect_language)
        except ValueError as e:
            raise HTTPException(400, str(e))
    return info


@router.delete("/{doc_id}")
def remove_document(doc_id: str, admin: User = Depends(require_admin)):
    if not rag_service.ready:
        raise HTTPException(503, rag_service.error or "Models not loaded")
    with rag_service.lock:
        n = delete_document(rag_service.pipeline.retriever, doc_id)
    if n == 0:
        raise HTTPException(404, "Document not found in the index")
    return {"deleted_chunks": n}
