"""Semantic text chunking service."""

from __future__ import annotations

from dataclasses import dataclass

from cortex.core.config import Settings


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk of text produced by the chunking pipeline."""

    chunk_index: int
    text: str
    token_count: int


class ChunkingService:
    """Splits cleaned text into overlapping semantic chunks.

    Chunks are built from whole paragraphs whenever possible. Word counts are
    used as a lightweight token estimate until a tokenizer is introduced in a
    later phase.
    """

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.ingestion_chunk_size_words
        self._overlap = settings.ingestion_chunk_overlap_words
        if self._overlap >= self._chunk_size:
            msg = "Chunk overlap must be smaller than chunk size"
            raise ValueError(msg)

    def chunk(self, text: str) -> list[TextChunk]:
        """Split ``text`` into ordered chunks with configured size and overlap."""
        paragraphs = [
            paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
        ]
        if not paragraphs:
            return []

        paragraph_words = [paragraph.split() for paragraph in paragraphs]
        chunks: list[TextChunk] = []
        current_words: list[str] = []
        paragraph_index = 0

        while paragraph_index < len(paragraph_words):
            words = paragraph_words[paragraph_index]
            paragraph_index += 1

            if not current_words or len(current_words) + len(words) <= self._chunk_size:
                current_words.extend(words)
            else:
                current_words = self._emit_chunks(chunks, current_words)
                current_words.extend(words)

            while len(current_words) >= self._chunk_size:
                current_words = self._emit_chunks(chunks, current_words)

        if current_words:
            self._emit_chunk(chunks, current_words)

        return chunks

    def _emit_chunks(self, existing: list[TextChunk], words: list[str]) -> list[str]:
        """Emit full-sized chunks and return the remaining word window."""
        while len(words) >= self._chunk_size:
            self._emit_chunk(existing, words)
            words = self._carry_overlap(words)
        return words

    def _emit_chunk(self, existing: list[TextChunk], words: list[str]) -> None:
        chunk_words = words[: self._chunk_size]
        chunk_text = " ".join(chunk_words)
        existing.append(
            TextChunk(
                chunk_index=len(existing),
                text=chunk_text,
                token_count=self.count_tokens(chunk_text),
            )
        )

    def _carry_overlap(self, words: list[str]) -> list[str]:
        """Return overlap tail plus any words beyond the emitted chunk."""
        emitted = words[: self._chunk_size]
        remainder = words[self._chunk_size :]
        if self._overlap <= 0:
            return remainder
        overlap = emitted[-self._overlap :]
        return overlap + remainder

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count using whitespace-delimited words."""
        if not text.strip():
            return 0
        return len(text.split())
