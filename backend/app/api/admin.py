"""Admin dashboard: statistics, logs, reload models."""
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.security import require_admin
from ..models import Document, DocumentChunk, Message, User
from ..services.rag import rag_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
def stats(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    docs_by_lang = dict(db.execute(select(Document.language, func.count()).group_by(Document.language)).all())
    q_by_lang = dict(db.execute(select(Message.language, func.count()).group_by(Message.language)).all())
    total_q = sum(q_by_lang.values())
    by_status = dict(db.execute(select(Message.status, func.count()).group_by(Message.status)).all())
    avg_rt = db.scalar(select(func.avg(Message.response_time_ms))) or 0.0
    fb = dict(db.execute(select(Message.feedback, func.count()).where(Message.feedback.is_not(None))
                         .group_by(Message.feedback)).all())
    days = []
    for i in range(6, -1, -1):
        d0 = today - timedelta(days=i)
        n = db.scalar(select(func.count()).select_from(Message).where(Message.created_at >= d0,
                                                                      Message.created_at < d0 + timedelta(days=1)))
        days.append({"date": d0.strftime("%Y-%m-%d"), "questions": n})
    return {
        "documents": {"total": sum(docs_by_lang.values()), "by_language": docs_by_lang,
                      "chunks": db.scalar(select(func.count()).select_from(DocumentChunk))},
        "questions": {"total": total_q,
                      "today": db.scalar(select(func.count()).select_from(Message).where(Message.created_at >= today)),
                      "this_week": db.scalar(select(func.count()).select_from(Message)
                                             .where(Message.created_at >= now - timedelta(days=7))),
                      "by_language": q_by_lang,
                      "by_language_percent": {k: round(100 * v / total_q, 1) for k, v in q_by_lang.items()} if total_q else {},
                      "by_status": by_status, "per_day": days},
        "average_response_time_ms": round(float(avg_rt), 1),
        "feedback": {"positive": fb.get(1, 0), "negative": fb.get(-1, 0)},
        "users": db.scalar(select(func.count()).select_from(User)),
        "models_loaded": rag_service.ready,
        "index_chunks": len(rag_service.pipeline.retriever.chunks) if rag_service.ready else 0,
    }


@router.get("/logs")
def logs(limit: int = 100, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    rows = db.scalars(select(Message).order_by(Message.id.desc()).limit(min(limit, 500))).all()
    return [{"id": m.id, "user_id": m.user_id, "question": m.question, "language": m.language, "status": m.status,
             "confidence": m.confidence, "response_time_ms": round(m.response_time_ms, 1),
             "sources": len(m.sources or []), "retrieved_chunks": len(m.retrieved_chunks or []),
             "feedback": m.feedback, "timestamp": m.created_at} for m in rows]


@router.post("/reload")
def reload_models(admin: User = Depends(require_admin)):
    rag_service.load()
    return {"models_loaded": rag_service.ready, "error": rag_service.error}
