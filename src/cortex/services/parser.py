"""PDF text extraction service."""

from __future__ import annotations

import asyncio
import io
import logging

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from cortex.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class ParserService:
    """Extracts ordered plain text from PDF binaries.

    Uses ``pypdf`` for lightweight, dependency-minimal text extraction. Images
    and non-text content are ignored by the underlying library.
    """

    async def parse_pdf(self, content: bytes) -> str:
        """Extract text from ``content`` preserving page order."""
        try:
            return await asyncio.to_thread(self._extract_text, content)
        except DocumentProcessingError:
            raise
        except Exception as exc:
            logger.exception("Unexpected PDF parsing failure")
            raise DocumentProcessingError(
                "Failed to read PDF document",
                details={"reason": str(exc)},
            ) from exc

    @staticmethod
    def _extract_text(content: bytes) -> str:
        if not content:
            raise DocumentProcessingError(
                "PDF file is empty",
                details={"field": "file"},
            )

        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
        except PdfReadError as exc:
            raise DocumentProcessingError(
                "PDF document is unreadable or corrupted",
                details={"reason": str(exc)},
            ) from exc

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise DocumentProcessingError(
                        "PDF document is password-protected",
                        details={"encrypted": True},
                    )
            except DocumentProcessingError:
                raise
            except Exception as exc:
                raise DocumentProcessingError(
                    "PDF document is password-protected",
                    details={"encrypted": True, "reason": str(exc)},
                ) from exc

        page_texts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted = page.extract_text() or ""
            except Exception as exc:
                raise DocumentProcessingError(
                    "Failed to extract text from PDF page",
                    details={"page_number": page_number, "reason": str(exc)},
                ) from exc
            if extracted.strip():
                page_texts.append(extracted.strip())

        if not page_texts:
            raise DocumentProcessingError(
                "PDF document contains no extractable text",
                details={"page_count": len(reader.pages)},
            )

        return "\n\n".join(page_texts)
