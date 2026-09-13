"""PDF helpers for tests."""

from __future__ import annotations

from io import BytesIO

from reportlab.pdfgen import canvas


def build_pdf_with_text(*lines: str) -> bytes:
    """Build a minimal PDF containing the provided lines of text."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    y_position = 800
    for line in lines:
        pdf.drawString(72, y_position, line)
        y_position -= 24
    pdf.save()
    return buffer.getvalue()
