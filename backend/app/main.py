"""FastAPI application: Satbayev University multilingual RAG consultant.

Run (from the project root):
    uvicorn backend.app.main:app --reload
Then open http://127.0.0.1:8000  (chat)  and  http://127.0.0.1:8000/admin.html  (admin panel).
API docs: http://127.0.0.1:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from .api import admin, auth, chat, documents  # noqa: E402
from .core.config import get_settings  # noqa: E402
from .core.database import SessionLocal, init_db  # noqa: E402
from .core.security import hash_password  # noqa: E402
from .models import DocumentChunk, User  # noqa: E402
from .services.rag import rag_service  # noqa: E402

settings = get_settings()


def ensure_admin():
    with SessionLocal() as db:
        if not db.scalar(select(User).where(User.email == settings.ADMIN_EMAIL.lower())):
            db.add(User(email=settings.ADMIN_EMAIL.lower(), full_name="Administrator",
                        password_hash=hash_password(settings.ADMIN_PASSWORD), role="admin"))
            db.commit()


def ensure_db_index():
    """First start: copy documents/chunks of the trained index into the database."""
    if not rag_service.ready:
        return
    with SessionLocal() as db:
        n = db.scalar(select(func.count()).select_from(DocumentChunk))
    if n == 0:
        from .services.indexing import sync_index_to_db
        sync_index_to_db(rag_service.pipeline.retriever)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    ensure_admin()
    rag_service.load()
    ensure_db_index()
    if rag_service.ready:
        print(f"[startup] models loaded, {len(rag_service.pipeline.retriever.chunks)} chunks in the index")
    else:
        print(f"[startup] WARNING: {rag_service.error}")
    yield


app = FastAPI(title=settings.APP_NAME, version="1.0.0", lifespan=lifespan,
              description="Multilingual (RU/KZ/EN) RAG virtual assistant for Satbayev University students")
@app.middleware("http")
async def no_cache_pages(request, call_next):
    """The page, scripts and styles are always re-validated, so a new version is shown without Ctrl+F5."""
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(admin.router)


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "models_loaded": rag_service.ready, "error": rag_service.error,
            "chunks": len(rag_service.pipeline.retriever.chunks) if rag_service.ready else 0,
            "documents": len({c["doc_id"] for c in rag_service.pipeline.retriever.chunks}) if rag_service.ready else 0}


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
