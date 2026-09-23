"""FastAPI routes for the local transcript Q&A application."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from .answer import AnswerGenerationError, answer_market_comparison, answer_question
from .config import DATA_DIR, INDEXES_DIR, settings
from .guide import GuideGenerationError, answer_interview_guide
from .ingest import LocalIndex, build_index, load_index
from .llm import OpenRouterConfigurationError
from .schemas import EvidenceItem, GuideResponse, VerifiedAnswer
from .scope import available_countries, canonicalize_countries


RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = INDEXES_DIR / "manifest.json"
logger = logging.getLogger("uvicorn.error")

TRACE_LABELS = {
    "query_planning": "Planning the search",
    "dense_retrieval": "Searching semantic matches",
    "bm25_retrieval": "Checking keyword matches",
    "fusion": "Combining source matches",
    "reranking": "Reranking the evidence",
    "evidence_gate": "Checking evidence coverage",
    "citation_bundle": "Preparing citations",
    "answer_generation_1": "Generating the answer",
    "answer_generation_2": "Revising the answer",
    "answer_validation_1": "Validating the answer",
    "answer_validation_2": "Rechecking the citations",
    "total": "Answer preparation total",
    "answer_stream": "Streaming the answer",
}
ANSWER_CACHE_VERSION = "verified-answer-v1"
ANSWER_CACHE_CAPACITY = 128
_answer_cache: OrderedDict[str, VerifiedAnswer] = OrderedDict()
_answer_cache_lock = Lock()


def _answer_cache_key(
    question: str,
    index: LocalIndex,
    options: dict[str, object],
) -> str:
    """Key exact verified-answer reuse to query, scope, model, and corpus."""

    payload = {
        "version": ANSWER_CACHE_VERSION,
        "question": " ".join(question.split()),
        "scope": options,
        "answer_model": settings.answer_model,
        "query_model": settings.query_model,
        "source_hashes": sorted(index.manifest.source_hashes.items()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_answer(key: str) -> VerifiedAnswer | None:
    with _answer_cache_lock:
        answer = _answer_cache.get(key)
        if answer is None:
            return None
        _answer_cache.move_to_end(key)
        return answer.model_copy(deep=True)


def _remember_answer(key: str, answer: VerifiedAnswer) -> None:
    if not answer.valid or answer.abstain:
        return
    with _answer_cache_lock:
        _answer_cache[key] = answer.model_copy(deep=True)
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > ANSWER_CACHE_CAPACITY:
            _answer_cache.popitem(last=False)


class AskRequest(BaseModel):
    """Request body for a transcript question."""

    question: str = Field(min_length=1)
    country_scope: list[str] | None = Field(default=None, min_length=1)
    require_all_countries: bool = False


class IngestRequest(BaseModel):
    """Optional raw transcript filenames; omitted means all raw transcripts."""

    source_files: list[str] | None = None


class IndexStatus(BaseModel):
    """Current process-local index status exposed to the frontend."""

    ready: bool
    chunk_count: int
    source_files: list[str]
    last_indexed_at: str | None = None


class IngestResponse(IndexStatus):
    """Status returned after a successful ingest operation."""

    indexed: bool = True


class BackendState:
    """Process-local index holder shared by the FastAPI routes."""

    def __init__(self) -> None:
        self.index: LocalIndex | None = None
        self._lock = Lock()

    def load_existing(self) -> None:
        """Load a previously persisted index when the API process starts."""

        if not MANIFEST_PATH.exists():
            return
        try:
            index = load_index(data_dir=DATA_DIR)
            if index.manifest.embedding_provider != settings.embedding_provider:
                logger.warning(
                    "Existing index uses %s embeddings; configured provider is %s. "
                    "Rebuild it with POST /api/ingest before asking questions.",
                    index.manifest.embedding_provider,
                    settings.embedding_provider,
                )
                self.index = None
                return
            if index.manifest.embedding_model != settings.embedding_model:
                logger.warning(
                    "Existing index uses embedding model %s; configured model is %s. "
                    "Rebuild it with POST /api/ingest before asking questions.",
                    index.manifest.embedding_model,
                    settings.embedding_model,
                )
                self.index = None
                return
            self.index = index
        except (FileNotFoundError, ValueError):
            self.index = None

    def replace(self, index: LocalIndex) -> None:
        """Atomically replace the index after a completed ingest."""

        with self._lock:
            self.index = index


state = BackendState()


def _last_indexed_at() -> str | None:
    if not MANIFEST_PATH.exists():
        return None
    timestamp = MANIFEST_PATH.stat().st_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _status() -> IndexStatus:
    index = state.index
    if index is None:
        return IndexStatus(
            ready=False,
            chunk_count=0,
            source_files=[],
            last_indexed_at=None,
        )
    return IndexStatus(
        ready=True,
        chunk_count=index.manifest.chunk_count,
        source_files=sorted({chunk.source_file for chunk in index.chunks}),
        last_indexed_at=_last_indexed_at(),
    )


def _source_paths(source_files: list[str] | None) -> list[Path]:
    """Resolve only transcript filenames directly inside the raw data folder."""

    if not source_files:
        paths = sorted(RAW_DIR.glob("*.txt"))
    else:
        raw_root = RAW_DIR.resolve()
        paths = []
        for source_file in source_files:
            candidate = (RAW_DIR / source_file).resolve()
            if candidate.parent != raw_root or candidate.suffix.lower() != ".txt":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "source_files must contain transcript filenames from "
                        "backend/data/raw"
                    ),
                )
            paths.append(candidate)
    if not paths or any(not path.is_file() for path in paths):
        raise HTTPException(
            status_code=400,
            detail="No readable transcript files were found in backend/data/raw",
        )
    return paths


def _answer_options(request: AskRequest, index: LocalIndex) -> dict[str, object]:
    """Validate an optional UI market scope against the loaded index."""

    options: dict[str, object] = {}
    if request.country_scope is not None:
        available = available_countries(index)
        if any(
            not canonicalize_countries([country], available)
            for country in request.country_scope
        ):
            raise HTTPException(
                status_code=400,
                detail="country_scope must contain markets present in the index",
            )
        options["country_scope"] = canonicalize_countries(
            request.country_scope,
            available,
        )
    if request.require_all_countries:
        options["require_all_countries"] = True
    return options


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.load_existing()
    yield


app = FastAPI(
    title="Transcript Intelligence RAG",
    version="0.8.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status", response_model=IndexStatus)
def get_status() -> IndexStatus:
    return _status()


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest | None = None) -> IngestResponse:
    paths = _source_paths(request.source_files if request else None)
    try:
        index = build_index(paths, data_dir=DATA_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc
    state.replace(index)
    status = _status()
    return IngestResponse(**status.model_dump())


@app.post("/api/ask", response_model=VerifiedAnswer)
def ask(request: AskRequest) -> VerifiedAnswer:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if state.index is None:
        raise HTTPException(
            status_code=503,
            detail="No index is loaded. Run POST /api/ingest first.",
        )
    try:
        return answer_question(
            question,
            state.index,
            **_answer_options(request, state.index),
        )
    except OpenRouterConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AnswerGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer provider returned an invalid response after retrying.",
        ) from exc


@app.post("/api/guide", response_model=GuideResponse)
def interview_guide() -> GuideResponse:
    """Return all six evidence-grounded guide answers for every expert."""

    if state.index is None:
        raise HTTPException(
            status_code=503,
            detail="No index is loaded. Run POST /api/ingest first.",
        )
    try:
        return answer_interview_guide(state.index)
    except OpenRouterConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GuideGenerationError as exc:
        raise HTTPException(
            status_code=502,
            detail="The guide provider did not return six valid answers for every expert.",
        ) from exc


@app.post("/api/compare", response_model=VerifiedAnswer)
def compare_markets() -> VerifiedAnswer:
    """Compare common themes and differences across all indexed markets."""

    if state.index is None:
        raise HTTPException(
            status_code=503,
            detail="No index is loaded. Run POST /api/ingest first.",
        )
    try:
        return answer_market_comparison(state.index)
    except OpenRouterConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _sse(event: dict[str, object]) -> str:
    """Serialize one server-sent event without buffering application state."""

    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream grounded progress, source metadata, and the verified answer."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if state.index is None:
        raise HTTPException(
            status_code=503,
            detail="No index is loaded. Run POST /api/ingest first.",
        )

    index = state.index
    answer_options = _answer_options(request, index)
    cache_key = _answer_cache_key(question, index, answer_options)
    cached_answer = _cached_answer(cache_key)
    request_id = uuid.uuid4().hex[:8]

    async def event_stream():
        loop = asyncio.get_running_loop()
        trace_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        timings: list[dict[str, object]] = []
        request_started = time.perf_counter()
        first_delta_at: float | None = None
        delta_event_count = 0
        logger.info("[trace %s] REQUEST started", request_id)

        def enqueue_answer_delta(text: str) -> None:
            nonlocal first_delta_at, delta_event_count
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            delta_event_count += 1
            trace_queue.put_nowait({"type": "delta", "text": text})

        def answer_delta(text: str) -> None:
            loop.call_soon_threadsafe(enqueue_answer_delta, text)

        def answer_reset() -> None:
            loop.call_soon_threadsafe(
                trace_queue.put_nowait,
                {"type": "reset"},
            )

        def trace(stage: str, elapsed_ms: float, details: dict[str, object]) -> None:
            phase = str(details.get("phase", "finished"))
            event = {
                "type": "timing",
                "stage": stage,
                "label": TRACE_LABELS.get(stage, stage),
                "elapsed_ms": elapsed_ms,
                "phase": phase,
                "details": details,
            }
            timings.append(event)
            if phase == "started":
                logger.info("[trace %s] START %s", request_id, TRACE_LABELS.get(stage, stage))
            else:
                logger.info(
                    "[trace %s] %s %s — %.2f ms %s",
                    request_id,
                    phase.upper(),
                    TRACE_LABELS.get(stage, stage),
                    elapsed_ms,
                    {key: value for key, value in details.items() if key != "phase"},
                )
            loop.call_soon_threadsafe(trace_queue.put_nowait, event)

        answer_task: asyncio.Task[VerifiedAnswer] | None = None
        if cached_answer is None:
            answer_task = asyncio.create_task(
                asyncio.to_thread(
                    answer_question,
                    question,
                    index,
                    request_id=request_id,
                    trace=trace,
                    on_answer_delta=answer_delta,
                    on_answer_reset=answer_reset,
                    **answer_options,
                )
            )
        else:
            logger.info("[trace %s] VERIFIED ANSWER CACHE hit", request_id)

        try:
            yield _sse(
                {
                    "type": "status",
                    "stage": "cache_hit" if cached_answer is not None else "started",
                    "label": (
                        "Using the saved verified answer"
                        if cached_answer is not None
                        else "Reading your question"
                    ),
                    "detail": (
                        "This exact question and transcript version were answered before."
                        if cached_answer is not None
                        else "Following the request as each step completes"
                    ),
                }
            )

            while answer_task is not None and not answer_task.done():
                try:
                    yield _sse(await asyncio.wait_for(trace_queue.get(), timeout=0.2))
                except asyncio.TimeoutError:
                    continue

            answer = (
                cached_answer
                if cached_answer is not None
                else await answer_task
            )
            _remember_answer(cache_key, answer)
            await asyncio.sleep(0)
            while not trace_queue.empty():
                yield _sse(trace_queue.get_nowait())

            yield _sse(
                {
                    "type": "status",
                    "stage": "finalizing",
                    "label": "Finalizing the source trail",
                    "detail": "The answer is verified and ready to stream",
                }
            )
            if cached_answer is not None and answer.answer:
                first_delta_at = time.perf_counter()
                delta_event_count = 1
                yield _sse({"type": "delta", "text": answer.answer})
            yield _sse(
                {
                    "type": "citations",
                    "citations": [citation.model_dump() for citation in answer.citations],
                }
            )

            stream_event = {
                "type": "timing",
                "stage": "answer_stream",
                "label": TRACE_LABELS["answer_stream"],
                "elapsed_ms": (
                    round((time.perf_counter() - first_delta_at) * 1000, 2)
                    if first_delta_at is not None
                    else 0.0
                ),
                "phase": "finished",
                "details": {
                    "model_delta_events": delta_event_count,
                    "first_delta_ms": (
                        round((first_delta_at - request_started) * 1000, 2)
                        if first_delta_at is not None
                        else None
                    ),
                },
            }
            timings.append(stream_event)
            logger.info(
                "[trace %s] FINISHED %s — %.2f ms %s",
                request_id,
                TRACE_LABELS["answer_stream"],
                stream_event["elapsed_ms"],
                stream_event["details"],
            )
            logger.info(
                "[trace %s] REQUEST complete — %.2f ms",
                request_id,
                (time.perf_counter() - request_started) * 1000,
            )
            yield _sse(stream_event)

            yield _sse(
                {
                    "type": "done",
                    "answer": answer.model_dump(),
                    "timings": timings,
                }
            )
        except Exception:
            logger.exception("[trace %s] REQUEST failed", request_id)
            yield _sse(
                {
                    "type": "error",
                    "message": "The request could not be completed. Please try again.",
                }
            )
        finally:
            if answer_task is not None and not answer_task.done():
                answer_task.cancel()
                logger.warning("[trace %s] REQUEST cancelled", request_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/evidence/{evidence_id}", response_model=EvidenceItem)
def get_evidence(evidence_id: str) -> EvidenceItem:
    if state.index is None:
        raise HTTPException(
            status_code=503,
            detail="No index is loaded. Run POST /api/ingest first.",
        )
    chunk = state.index.get_chunk(evidence_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="Evidence ID not found")
    return EvidenceItem.from_chunk(chunk)
