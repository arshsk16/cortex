"""Unit tests for ParserService."""

from __future__ import annotations

import pytest

from cortex.core.exceptions import DocumentProcessingError
from cortex.services.parser import ParserService


@pytest.fixture
def parser_service() -> ParserService:
    """ParserService instance for unit tests."""
    return ParserService()


@pytest.mark.asyncio
async def test_parse_pdf_extracts_text_in_page_order(
    parser_service: ParserService,
) -> None:
    """ParserService preserves page order when extracting text."""
    from io import BytesIO

    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 800, "Page one content")
    pdf.showPage()
    pdf.drawString(72, 800, "Page two content")
    pdf.save()
    text = await parser_service.parse_pdf(buffer.getvalue())
    assert "Page one content" in text
    assert "Page two content" in text
    assert text.index("Page one content") < text.index("Page two content")


@pytest.mark.asyncio
async def test_parse_pdf_rejects_empty_content(
    parser_service: ParserService,
) -> None:
    """Empty PDF payloads raise DocumentProcessingError."""
    with pytest.raises(DocumentProcessingError) as exc_info:
        await parser_service.parse_pdf(b"")
    assert exc_info.value.code == "document_processing_error"


@pytest.mark.asyncio
async def test_parse_pdf_rejects_invalid_bytes(
    parser_service: ParserService,
) -> None:
    """Corrupted PDF bytes raise DocumentProcessingError."""
    with pytest.raises(DocumentProcessingError):
        await parser_service.parse_pdf(b"%PDF-1.4\nnot-a-valid-pdf")


@pytest.mark.asyncio
async def test_parse_pdf_rejects_pdf_without_text(
    parser_service: ParserService,
) -> None:
    """PDFs with no extractable text raise DocumentProcessingError."""
    from io import BytesIO

    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    canvas.Canvas(buffer).save()
    with pytest.raises(DocumentProcessingError) as exc_info:
        await parser_service.parse_pdf(buffer.getvalue())
    assert "no extractable text" in exc_info.value.message.lower()
