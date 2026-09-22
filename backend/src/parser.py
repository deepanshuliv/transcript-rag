"""Deterministic parser for the supplied expert-call transcript format."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .schemas import TranscriptChunk


TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{2}:\d{2})[ \t]*\r?$")
EXPERT_RE = re.compile(r"^Expert\s+\d+\s*[–-]\s*(?P<expert>.+?)\s*$")
FIELD_RE = re.compile(r"^(?P<field>Role|Market)\s*:\s*(?P<value>.+?)\s*$")
SPEAKER_RE = re.compile(
    r"^(?P<speaker>[^:\n]+?)\s*:\s*(?P<content>.*)$"
)


@dataclass(frozen=True)
class _Utterance:
    timestamp: str
    speaker: str
    text: str
    block_start: int
    block_end: int


def _slug(value: str) -> str:
    """Create a stable ASCII slug for evidence IDs."""

    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")
    return slug or "transcript"


def _line_end(text: str, start: int) -> int:
    """Return the end offset of a line, excluding its line break."""

    newline = text.find("\n", start)
    return len(text) if newline == -1 else newline


def _parse_header(text: str, first_timestamp_start: int) -> tuple[str, str, str]:
    """Extract the full expert name, role, and market from the header."""

    header = text[:first_timestamp_start]
    expert = ""
    country = ""
    role = ""
    for raw_line in header.splitlines():
        line = raw_line.strip()
        expert_match = EXPERT_RE.match(line)
        if expert_match:
            expert = expert_match.group("expert").strip()
            continue
        field_match = FIELD_RE.match(line)
        if not field_match:
            continue
        field = field_match.group("field")
        value = field_match.group("value").strip()
        if field == "Market":
            country = value
        elif field == "Role":
            role = value

    if not expert:
        raise ValueError("Transcript header is missing an 'Expert N – Name' line")
    if not country:
        raise ValueError("Transcript header is missing a 'Market: Country' line")
    return expert, country, role


def _parse_utterances(text: str) -> tuple[list[_Utterance], int]:
    """Parse timestamp blocks and speaker-prefixed utterances."""

    timestamp_matches = list(
        re.finditer(r"(?m)^(?P<timestamp>\d{2}:\d{2})[ \t]*\r?$", text)
    )
    if not timestamp_matches:
        raise ValueError("Transcript does not contain any timestamp lines")

    utterances: list[_Utterance] = []
    first_timestamp_start = timestamp_matches[0].start()
    for index, timestamp_match in enumerate(timestamp_matches):
        block_start = timestamp_match.start()
        block_end = (
            timestamp_matches[index + 1].start()
            if index + 1 < len(timestamp_matches)
            else len(text)
        )
        content_start = _line_end(text, timestamp_match.end())
        if content_start < len(text):
            content_start += 1
        block = text[content_start:block_end]

        # The regular format puts the speaker prefix on the first non-empty
        # line. Continuation lines are retained as part of the utterance.
        leading_match = re.match(r"[ \t]*(?:\r?\n)?", block)
        speaker_start = content_start + (leading_match.end() if leading_match else 0)
        speaker_line_end = _line_end(text, speaker_start)
        speaker_line = text[speaker_start:speaker_line_end]
        speaker_match = SPEAKER_RE.match(speaker_line)
        if not speaker_match:
            raise ValueError(
                f"Timestamp {timestamp_match.group('timestamp')} is not followed "
                "by a speaker prefix"
            )

        speaker = speaker_match.group("speaker").strip()
        first_content_start = speaker_start + speaker_match.start("content")
        raw_text = text[first_content_start:block_end]
        utterance_text = raw_text.strip()
        if not utterance_text:
            raise ValueError(
                f"Speaker '{speaker}' at {timestamp_match.group('timestamp')} "
                "has no utterance text"
            )

        utterances.append(
            _Utterance(
                timestamp=timestamp_match.group("timestamp"),
                speaker=speaker,
                text=utterance_text,
                block_start=block_start,
                block_end=block_end,
            )
        )

    return utterances, first_timestamp_start


def _make_chunk(
    *,
    question: _Utterance,
    answer: _Utterance,
    expert: str,
    country: str,
    source_file: str,
    source_hash: str,
    occurrence: int,
) -> TranscriptChunk:
    """Build one schema-valid chunk from a question/answer pair."""

    base_id = f"{_slug(country)}_{answer.timestamp.replace(':', '_')}"
    chunk_id = base_id if occurrence == 1 else f"{base_id}_{occurrence}"
    retrieval_text = f"Question: {question.text}\nAnswer: {answer.text}"
    return TranscriptChunk(
        chunk_id=chunk_id,
        source_file=source_file,
        source_hash=source_hash,
        country=country,
        expert=expert,
        speaker=answer.speaker,
        question_text=question.text,
        answer_text=answer.text,
        retrieval_text=retrieval_text,
        question_timestamp=question.timestamp,
        answer_timestamp=answer.timestamp,
        char_start=question.block_start,
        char_end=answer.block_end,
    )


def parse_transcript(path: str | Path) -> list[TranscriptChunk]:
    """Parse a transcript into deterministic timestamped Q&A chunks.

    The source bytes are hashed before decoding, and all text used in the
    returned chunks comes from the original decoded file. The parser pairs
    each interviewer utterance with the next non-interviewer utterance. It does
    not call an LLM or perform semantic interpretation.
    """

    transcript_path = Path(path)
    raw_bytes = transcript_path.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8")

    utterances, first_timestamp_start = _parse_utterances(text)
    expert, country, _role = _parse_header(text, first_timestamp_start)
    source_file = transcript_path.name

    chunks: list[TranscriptChunk] = []
    pending_question: _Utterance | None = None
    id_occurrences: dict[str, int] = {}
    for utterance in utterances:
        if utterance.speaker.casefold() == "interviewer":
            pending_question = utterance
            continue
        if pending_question is None:
            raise ValueError(
                f"Expert utterance at {utterance.timestamp} has no preceding question"
            )

        base_id = f"{_slug(country)}_{utterance.timestamp.replace(':', '_')}"
        occurrence = id_occurrences.get(base_id, 0) + 1
        id_occurrences[base_id] = occurrence
        chunks.append(
            _make_chunk(
                question=pending_question,
                answer=utterance,
                expert=expert,
                country=country,
                source_file=source_file,
                source_hash=source_hash,
                occurrence=occurrence,
            )
        )
        pending_question = None

    if pending_question is not None:
        raise ValueError(
            f"Interviewer question at {pending_question.timestamp} has no answer"
        )
    return chunks


def parse_transcripts(paths: list[str | Path]) -> list[TranscriptChunk]:
    """Parse multiple transcript paths in the supplied order."""

    chunks: list[TranscriptChunk] = []
    for path in paths:
        chunks.extend(parse_transcript(path))
    return chunks
