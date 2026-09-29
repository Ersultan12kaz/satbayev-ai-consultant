"""Chat: ask a question, history, feedback."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.database import get_db
from ml.composer import natural_text
from ..core.security import get_current_user, get_current_user_optional, chat_rate_limit
from ..models import Conversation, Message, User
from ..schemas import ChatIn, ChatOut, FeedbackIn
from ..services.rag import rag_service

router = APIRouter(prefix="/api", tags=["chat"])
settings = get_settings()


@router.post("/chat", response_model=ChatOut, dependencies=[Depends(chat_rate_limit)])
def chat(data: ChatIn, db: Session = Depends(get_db), user: Optional[User] = Depends(get_current_user_optional)):
    if user is None and not settings.ALLOW_ANONYMOUS_CHAT:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Login required")
    if not rag_service.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, rag_service.error or "Models not loaded")
    conv = None
    if data.conversation_id:
        conv = db.get(Conversation, data.conversation_id)
        if conv and user and conv.user_id not in (None, user.id):
            conv = None
    # previous questions of this conversation -> follow-ups ("қазақша нұсқада бер", "подробнее") work
    history, dialog = [], []
    if conv is not None:
        for m in conv.messages[-5:]:
            if m.status != "out_of_scope":
                history.append(m.question)
            if m.status == "answered":
                dialog.append((m.question, m.answer))          # the language model remembers the conversation
    res = rag_service.ask(data.question, mode=data.mode or settings.DEFAULT_MODE,
                          top_k=data.top_k or settings.DEFAULT_TOP_K,
                          generator=data.generator or settings.GENERATOR, lang=data.language, history=history,
                          dialog=dialog[-2:])
    res["text"] = natural_text(res["text"], res["lang"])
    if conv is None:
        conv = Conversation(user_id=user.id if user else None, title=data.question[:120])
        db.add(conv)
        db.flush()
    msg = Message(conversation_id=conv.id, user_id=user.id if user else None, question=res["question"],
                  answer=res["text"], language=res["lang"], status=res["status"], confidence=res["confidence"],
                  confidence_score=res["confidence_score"], mode=res["mode"], sources=res["sources"],
                  retrieved_chunks=res["retrieved"], response_time_ms=res["timings"]["total_ms"])
    db.add(msg)
    db.commit()
    return ChatOut(message_id=msg.id, conversation_id=conv.id, question=res["question"], answer=res["text"],
                   language=res["lang"], status=res["status"], confidence=res["confidence"],
                   confidence_score=res["confidence_score"], sources=res["sources"],
                   response_time_ms=res["timings"]["total_ms"], timings=res["timings"], retrieved=res["retrieved"],
                   generator=res.get("generator", ""), question_type=res.get("question_type"),
                   concepts=res.get("concepts", []), scope_probability=res.get("scope_probability"),
                   evidence_overlap=res.get("evidence_overlap"))


@router.post("/messages/{message_id}/feedback")
def feedback(message_id: int, data: FeedbackIn, db: Session = Depends(get_db)):
    msg = db.get(Message, message_id)
    if not msg:
        raise HTTPException(404, "Message not found")
    msg.feedback = data.value
    db.commit()
    return {"ok": True}


@router.get("/conversations")
def conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    convs = db.scalars(select(Conversation).where(Conversation.user_id == user.id)
                       .order_by(Conversation.id.desc()).limit(50)).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at, "messages": len(c.messages)} for c in convs]


@router.get("/conversations/{conv_id}")
def conversation(conv_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = db.get(Conversation, conv_id)
    if not c or c.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    return {"id": c.id, "title": c.title,
            "messages": [{"id": m.id, "question": m.question, "answer": m.answer, "language": m.language,
                          "status": m.status, "confidence": m.confidence, "sources": m.sources,
                          "response_time_ms": m.response_time_ms, "created_at": m.created_at} for m in c.messages]}
