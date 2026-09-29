"""Pydantic schemas (request / response validation)."""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RegisterIn(BaseModel):
    email: str = Field(min_length=5, max_length=255, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=6, max_length=128)
    full_name: str = Field(default="", max_length=255)


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    conversation_id: Optional[int] = None
    mode: Optional[str] = Field(default=None, pattern="^(dense|bm25|hybrid)$")
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    language: Optional[str] = Field(default=None, pattern="^(ru|kk|en)$")
    generator: Optional[str] = Field(default=None, pattern="^(composer|extractive)$")


class Source(BaseModel):
    n: int
    title: str
    page: int
    url: str = ""
    lang: str = ""
    section: str = ""
    doc_id: str = ""
    chunk_id: Optional[str] = None
    score: float = 0.0


class ChatOut(BaseModel):
    message_id: int
    conversation_id: int
    question: str
    answer: str
    language: str
    status: str
    confidence: str
    confidence_score: float
    sources: List[Source]
    response_time_ms: float
    timings: Dict[str, float] = {}
    retrieved: List[Dict] = []
    generator: str = ""
    question_type: Optional[Dict[str, float]] = None
    concepts: List[str] = []
    scope_probability: Optional[float] = None
    evidence_overlap: Optional[float] = None


class FeedbackIn(BaseModel):
    value: int = Field(ge=-1, le=1)


class DocumentOut(BaseModel):
    id: int
    doc_id: str
    title: str
    language: str
    category: str
    source_url: str
    n_chunks: int
    created_at: datetime
