"""Unit tests for CleaningService and ChunkingService."""

from __future__ import annotations

import pytest

from cortex.core.config import Settings
from cortex.services.chunking import ChunkingService
from cortex.services.cleaning import CleaningService


@pytest.fixture
def cleaning_service() -> CleaningService:
    """CleaningService instance for unit tests."""
    return CleaningService()


@pytest.fixture
def chunking_service(test_settings: Settings) -> ChunkingService:
    """ChunkingService configured with test settings."""
    return ChunkingService(
        test_settings.model_copy(
            update={
                "ingestion_chunk_size_words": 20,
                "ingestion_chunk_overlap_words": 5,
            }
        )
    )


def test_cleaning_normalizes_whitespace(cleaning_service: CleaningService) -> None:
    """CleaningService collapses irregular spacing and blank lines."""
    raw = "Hello    world\r\n\r\n\r\n\nSecond   paragraph"
    cleaned = cleaning_service.clean(raw)
    assert cleaned == "Hello world\n\nSecond paragraph"


def test_cleaning_removes_non_printable_characters(
    cleaning_service: CleaningService,
) -> None:
    """Non-printable characters are stripped from extracted text."""
    raw = "Hello\x00world\x07"
    cleaned = cleaning_service.clean(raw)
    assert cleaned == "Helloworld"


def test_chunking_splits_into_multiple_chunks(
    chunking_service: ChunkingService,
) -> None:
    """Long text is split into multiple chunks with indices."""
    paragraph_one = " ".join(f"word{i}" for i in range(15))
    paragraph_two = " ".join(f"term{i}" for i in range(15))
    text = f"{paragraph_one}\n\n{paragraph_two}"

    chunks = chunking_service.chunk(text)
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(chunk.token_count > 0 for chunk in chunks)


def test_chunking_preserves_paragraph_boundaries_when_possible(
    chunking_service: ChunkingService,
) -> None:
    """Small paragraphs remain intact inside a chunk when they fit."""
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = chunking_service.chunk(text)
    assert len(chunks) == 1
    assert "Short paragraph one." in chunks[0].text
    assert "Short paragraph two." in chunks[0].text


def test_chunking_applies_overlap(chunking_service: ChunkingService) -> None:
    """Consecutive chunks share overlapping words."""
    words = [f"token{i}" for i in range(40)]
    text = " ".join(words)
    chunks = chunking_service.chunk(text)
    assert len(chunks) >= 2

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap = first_words[-5:]
    assert second_words[:5] == overlap
