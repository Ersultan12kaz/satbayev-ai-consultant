"""RAG service: loads the trained pipeline once and answers questions."""
import threading
from pathlib import Path
from typing import Dict, Optional

from ml.rag import RAGPipeline

from ..core.config import get_settings
from .retrieval import make_retriever

settings = get_settings()


class RAGService:
    def __init__(self):
        self.pipeline: Optional[RAGPipeline] = None
        self.error: Optional[str] = None
        self.lock = threading.RLock()

    def load(self) -> None:
        d = Path(settings.MODELS_DIR)
        try:
            p = RAGPipeline.load(d)
            p.retriever = make_retriever(p.retriever, settings.VECTOR_BACKEND)
            p.extractive.encoder = p.composer.encoder = p.retriever.encoder
            self.pipeline, self.error = p, None
        except FileNotFoundError as e:
            self.pipeline, self.error = None, f"Models are not trained yet ({e.filename}). Run scripts/run_all.py"
        except Exception as e:  # pragma: no cover
            self.pipeline, self.error = None, f"Could not load models: {e}"

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def ask(self, question: str, mode: str, top_k: int, generator: str, lang: Optional[str] = None,
            history: Optional[list] = None, dialog: Optional[list] = None) -> Dict:
        with self.lock:
            return self.pipeline.answer(question, mode=mode, top_k=top_k, generator=generator, lang=lang,
                                        history=history, dialog=dialog)

    def detect_language(self, text: str) -> str:
        return self.pipeline.detect_language(text) if self.pipeline else "ru"


rag_service = RAGService()
