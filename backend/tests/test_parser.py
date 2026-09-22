"""Acceptance tests for the deterministic Phase 1 transcript parser."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from src.parser import parse_transcript, parse_transcripts


ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = [
    ROOT / "Transcript_1_France.txt",
    ROOT / "Transcript_2_Germany.txt",
    ROOT / "Transcript_3_UK.txt",
]


@pytest.mark.parametrize("path", TRANSCRIPTS)
def test_every_expert_answer_is_a_timestamped_qa_chunk(path: Path) -> None:
    chunks = parse_transcript(path)

    assert len(chunks) == 7
    assert all(chunk.country for chunk in chunks)
    assert all(chunk.expert for chunk in chunks)
    assert all(chunk.speaker for chunk in chunks)
    assert all(chunk.question_text for chunk in chunks)
    assert all(chunk.answer_text for chunk in chunks)
    assert all(chunk.answer_timestamp for chunk in chunks)
    assert all(chunk.question_timestamp for chunk in chunks)
    assert all(chunk.evidence_id == chunk.chunk_id for chunk in chunks)


def test_parser_preserves_expected_metadata_and_pairing() -> None:
    chunks = parse_transcript(TRANSCRIPTS[0])

    first = chunks[0]
    assert first.chunk_id == "france_00_18"
    assert first.source_file == "Transcript_1_France.txt"
    assert first.country == "France"
    assert first.expert == "Dr. Jean Martin"
    assert first.speaker == "Dr. Martin"
    assert first.question_timestamp == "00:00"
    assert first.answer_timestamp == "00:18"
    assert first.question_text == (
        "Thanks for joining. To begin, how would you describe robotic surgery "
        "adoption in France today?"
    )
    assert first.answer_text == (
        "Adoption is growing, but it is still concentrated in larger academic "
        "hospitals and private centres with stronger capital budgets. Smaller "
        "regional hospitals are much slower."
    )
    assert first.retrieval_text == (
        f"Question: {first.question_text}\nAnswer: {first.answer_text}"
    )


def test_answer_text_is_exactly_findable_in_original_source() -> None:
    for path in TRANSCRIPTS:
        # Decode the original bytes directly so offsets remain aligned with
        # the parser's no-newline-normalization contract.
        source = path.read_bytes().decode("utf-8")
        chunks = parse_transcript(path)
        source_hash = sha256(path.read_bytes()).hexdigest()
        assert all(chunk.source_hash == source_hash for chunk in chunks)
        for chunk in chunks:
            assert chunk.answer_text in source
            assert chunk.char_start is not None
            assert chunk.char_end is not None
            paired_source = source[chunk.char_start : chunk.char_end]
            assert chunk.question_text in paired_source
            assert chunk.answer_text in paired_source


def test_all_three_transcripts_parse_into_unique_evidence_ids() -> None:
    chunks = parse_transcripts(TRANSCRIPTS)

    assert len(chunks) == 21
    assert len({chunk.evidence_id for chunk in chunks}) == 21
    assert {chunk.country for chunk in chunks} == {"France", "Germany", "United Kingdom"}


def test_parser_rejects_an_unanswered_question(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(
        "Expert 1 – Test Expert\nRole: Test\nMarket: Testland\n\n"
        "00:00\nInterviewer: A question?\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no answer"):
        parse_transcript(path)
