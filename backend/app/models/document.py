from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(5), index=True)
    category: Mapped[str] = mapped_column(String(120), default="")
    source_url: Mapped[str] = mapped_column(Text, default="")
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[str] = mapped_column(String(255), default="pipeline")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
