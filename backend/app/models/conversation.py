from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan",
                            order_by="Message.id")


class Message(Base):
    """Every question/answer is logged: user, question, language, response, sources, timings."""
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(5), index=True)
    status: Mapped[str] = mapped_column(String(20), default="answered")
    confidence: Mapped[str] = mapped_column(String(10), default="low")
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    mode: Mapped[str] = mapped_column(String(10), default="hybrid")
    sources: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_chunks: Mapped[list] = mapped_column(JSON, default=list)
    response_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # +1 / -1
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    conversation = relationship("Conversation", back_populates="messages")
