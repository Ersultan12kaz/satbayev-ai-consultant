"""SQLAlchemy engine / session. SQLite locally, PostgreSQL (+pgvector) in Docker."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import get_settings

settings = get_settings()
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
if _is_sqlite:
    from pathlib import Path
    Path(settings.DATABASE_URL.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if _is_sqlite else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def use_pgvector() -> bool:
    return settings.VECTOR_BACKEND == "pgvector" and settings.DATABASE_URL.startswith("postgresql")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from .. import models  # noqa: F401  (register tables)
    if use_pgvector():
        with engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    if use_pgvector():
        from sqlalchemy import text
        with engine.begin() as conn:  # approximate nearest-neighbour index for cosine distance
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw ON document_chunks "
                              "USING hnsw (embedding vector_cosine_ops)"))
