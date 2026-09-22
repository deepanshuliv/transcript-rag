"""FastAPI application shell for later API phases.

Phase 0/1 intentionally exposes no retrieval or answer endpoints yet.
"""

from fastapi import FastAPI


app = FastAPI(title="Transcript Intelligence RAG", version="0.1.0")
