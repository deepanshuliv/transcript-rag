"""FastAPI routes for the local transcript Q&A application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .answer import answer_question
from .config import DATA_DIR, INDEXES_DIR
from .ingest import LocalIndex, build_index, load_index
from .llm import OpenRouterConfigurationError
from .schemas import EvidenceItem, VerifiedAnswer


RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = INDEXES_DIR / "manifest.json"


class AskRequest(BaseModel):
    """Request body for a transcript question."""

    question: str = Field(min_length=1)


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
            self.index = load_index(data_dir=DATA_DIR)
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
        return answer_question(question, state.index)
    except OpenRouterConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
