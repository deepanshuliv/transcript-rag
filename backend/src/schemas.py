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

    @classmethod
    def from_chunk(
        cls,
        chunk: TranscriptChunk,
        *,
        retrieval_score: float | None = None,
        rerank_score: float | None = None,
    ) -> "EvidenceItem":
        """Create retrieval evidence while retaining citation metadata."""

        return cls(
            evidence_id=chunk.chunk_id,
            text=chunk.retrieval_text,
            country=chunk.country,
            expert=chunk.expert,
            source_file=chunk.source_file,
            timestamp=chunk.answer_timestamp,
            retrieval_score=retrieval_score,
            rerank_score=rerank_score,
        )


class EvidenceBundle(BaseModel):
    """Evidence bundle contract reserved for later answer phases."""

    request_id: str
    query: str
    items: dict[str, EvidenceItem]
    ordered_ids: list[str]


class AnswerClaim(BaseModel):
    """Answer-claim contract reserved for later validation phases."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_ids: list[str]


class LLMAnswer(BaseModel):
    """Strict model-answer contract reserved for later answer phases."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    claims: list[AnswerClaim]
    abstain: bool
    abstain_reason: str | None = None


GuideQuestionId = Literal[
    "adoption",
    "barriers",
    "economics",
    "training_outcomes",
    "outlook",
    "timeline",
]


class LLMGuideItem(BaseModel):
    """One answer in the structured six-question expert guide response."""

    model_config = ConfigDict(extra="forbid")

    question_id: GuideQuestionId
    claims: list[AnswerClaim] = Field(default_factory=list)
    abstain: bool = False
    abstain_reason: str | None = None


class LLMGuideAnswer(BaseModel):
    """One model response containing answers to all guide questions."""

    model_config = ConfigDict(extra="forbid")

    answers: list[LLMGuideItem] = Field(min_length=6, max_length=6)


class Citation(BaseModel):
    """Application-generated citation reconstructed from EvidenceBundle data."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    quote: str
    country: str
    expert: str
    source_file: str
    timestamp: str


class VerifiedClaim(BaseModel):
    """A validated answer claim and its application-generated citations."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_ids: list[str]
    citations: list[Citation]
    disagreement: bool = False


class VerifiedAnswer(BaseModel):
    """Final Phase 7 answer contract returned by the answer orchestrator."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    answer: str
    claims: list[VerifiedClaim]
    citations: list[Citation]
    abstain: bool
    abstain_reason: str | None = None


class GuideQuestionResult(BaseModel):
    question_id: GuideQuestionId
    question: str
    result: VerifiedAnswer


class ExpertGuideResult(BaseModel):
    country: str
    expert: str
    source_file: str
    answers: list[GuideQuestionResult]


class GuideResponse(BaseModel):
    countries: list[str]
    experts: list[ExpertGuideResult]


class IndexManifest(BaseModel):
    """Manifest tying all Phase 2 persistence artifacts to one index build."""

    schema_version: int = 1
    embedding_provider: str = "local"
    embedding_model: str
    embedding_dimension: int = Field(ge=1)
    chroma_collection: str
    chunks_path: str
    bm25_index_path: str
    source_hashes: dict[str, str]
    chunk_ids: list[str]
    chunk_count: int = Field(ge=0)
