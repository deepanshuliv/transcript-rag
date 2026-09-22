"""Application configuration for the local backend.

Phase 0 defines the environment contract. Model loading remains lazy, while
OpenRouter clients are created only when a query-planning call is requested.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHROMA_DIR = DATA_DIR / "chroma"
INDEXES_DIR = DATA_DIR / "indexes"
DEFAULT_COLLECTION_NAME = "transcript_chunks"


class Settings(BaseModel):
    """Environment-backed settings shared by later backend phases."""

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    query_model: str = "openai/gpt-4.1-nano"
    answer_model: str = "openai/gpt-5-nano"
    rerank_model: str = "cohere/rerank-v3.5:free"
    embedding_provider: str = "openrouter"
    embedding_model: str = "baai/bge-m3"
    embedding_dimension: int = Field(default=1024, ge=1)

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> "Settings":
        """Load the Phase 0 environment contract without requiring an API key."""

        load_dotenv(dotenv_path or BACKEND_DIR / ".env")
        dimension = os.getenv("EMBEDDING_DIMENSION", "1024")
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            query_model=os.getenv("QUERY_MODEL", "openai/gpt-4.1-nano"),
            answer_model=os.getenv("ANSWER_MODEL", "openai/gpt-5-nano"),
            rerank_model=os.getenv(
                "RERANK_MODEL", "cohere/rerank-v3.5:free"
            ),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openrouter"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "baai/bge-m3"),
            embedding_dimension=int(dimension),
        )


settings = Settings.from_env()
