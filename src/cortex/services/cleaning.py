"""Text normalization service for extracted document content."""

from __future__ import annotations

import re
import unicodedata


class CleaningService:
    """Normalizes raw extracted text while preserving paragraph structure."""

    _NON_PRINTABLE_PATTERN = re.compile(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]")
    _MULTI_SPACE_PATTERN = re.compile(r"[^\S\n]+")
    _EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        """Normalize whitespace and remove non-printable characters."""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = self._NON_PRINTABLE_PATTERN.sub("", normalized)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = self._MULTI_SPACE_PATTERN.sub(" ", normalized)

        paragraphs: list[str] = []
        for paragraph in normalized.split("\n\n"):
            collapsed = " ".join(paragraph.split())
            if collapsed:
                paragraphs.append(collapsed)

        cleaned = "\n\n".join(paragraphs)
        cleaned = self._EXCESS_NEWLINES_PATTERN.sub("\n\n", cleaned)
        return cleaned.strip()
