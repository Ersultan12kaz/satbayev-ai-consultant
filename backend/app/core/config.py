"""Application settings (read from environment variables or the .env file)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Satbayev University AI Consultant"
    # sqlite for local run (PyCharm), postgresql+psycopg2://... for Docker (pgvector)
    DATABASE_URL: str = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
    VECTOR_BACKEND: str = "numpy"          # numpy | pgvector
    MODELS_DIR: str = str(ROOT / "models")
    UPLOAD_DIR: str = str(ROOT / "data" / "uploads")

    SECRET_KEY: str = "change-me-in-.env-0f3a9c1e7b5d4a2c8e6f1b3d5a7c9e2f"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_MINUTES: int = 60 * 24

    ADMIN_EMAIL: str = "admin@satbayev.university"
    ADMIN_PASSWORD: str = "Admin12345!"

    CORS_ORIGINS: str = "*"
    RATE_LIMIT_PER_MINUTE: int = 30         # login / register attempts per minute
    CHAT_RATE_LIMIT_PER_MINUTE: int = 0     # questions per minute, 0 = no limit
    DEFAULT_MODE: str = "hybrid"           # dense | bm25 | hybrid
    DEFAULT_TOP_K: int = 5
    GENERATOR: str = "composer"            # composer (our answer composer) | extractive (baseline)
    ALLOW_ANONYMOUS_CHAT: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
