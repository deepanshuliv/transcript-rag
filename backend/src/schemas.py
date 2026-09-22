"""Pydantic data contracts defined by IMPLEMENTATION_PLAN.md."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptChunk(BaseModel):
    """A deterministic, timestamped interviewer-question/expert-answer pair.

    ``chunk_id`` is the canonical stable evidence ID for Phase 1. The
    read-only ``evidence_id`` property makes that relationship explicit for
    downstream retrieval and validation code without changing the plan's
    serialized schema.
    """

    chunk_id: str
    source_file: str
    source_hash: str
    country: str
    expert: str
    speaker: str
    question_text: str
    answer_text: str
    retrieval_text: str
    question_timestamp: str
    answer_timestamp: str
    char_start: int | None = None
    char_end: int | None = None
    parent_chunk_id: str | None = None

    @property
    def evidence_id(self) -> str:
        """Return the stable evidence identifier used by later phases."""

        return self.chunk_id


class QueryPlan(BaseModel):
    """Strict query-planning contract reserved for Phase 3."""

    model_config = ConfigDict(extra="forbid")

    original_query: str = Field(min_length=1)
    normalized_query: str = Field(min_length=1)
    search_queries: list[str] = Field(min_length=1, max_length=5)
    countries: list[str] = Field(default_factory=list)
    experts: list[str] = Field(default_factory=list)
    intent: Literal[
        "fact_lookup",
        "comparison",
        "summary",
        "timeline",
        "multi_part",
        "unknown",
    ]
    requires_all_countries: bool = False


class EvidenceItem(BaseModel):
    """Retrieved evidence contract reserved for later retrieval phases."""

    evidence_id: str
    text: str
    country: str
    expert: str
    source_file: str
    timestamp: str
    retrieval_score: float | None = None
    rerank_score: float | None = None


class EvidenceBundle(BaseModel):
    """Evidence bundle contract reserved for later answer phases."""

    request_id: str
    query: str
    items: dict[str, EvidenceItem]
    ordered_ids: list[str]


class AnswerClaim(BaseModel):
    """Answer-claim contract reserved for later validation phases."""

    claim: str
    evidence_ids: list[str]


class LLMAnswer(BaseModel):
    """Strict model-answer contract reserved for later answer phases."""

    answer: str
    claims: list[AnswerClaim]
    abstain: bool
    abstain_reason: str | None = None


class IndexManifest(BaseModel):
    """Manifest tying all Phase 2 persistence artifacts to one index build."""

    schema_version: int = 1
    embedding_model: str
    embedding_dimension: int = Field(ge=1)
    chroma_collection: str
    chunks_path: str
    bm25_index_path: str
    source_hashes: dict[str, str]
    chunk_ids: list[str]
    chunk_count: int = Field(ge=0)
