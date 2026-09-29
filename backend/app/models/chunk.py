from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base, use_pgvector

EMBEDDING_DIM = 256

if use_pgvector():
    from pgvector.sqlalchemy import Vector
    _EmbeddingType = Vector(EMBEDDING_DIM)
else:
    from sqlalchemy import JSON as _JSON
    _EmbeddingType = _JSON


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer, default=1)
    section: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(5), index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(_EmbeddingType, nullable=True)
    document = relationship("Document", back_populates="chunks")
