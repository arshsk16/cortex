"""Filesystem storage adapter for uploaded documents."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from cortex.core.config import Settings

logger = logging.getLogger(__name__)

PDF_EXTENSION = ".pdf"


class StorageService:
    """Low-level file persistence for document binaries.

    This class performs only filesystem operations. Validation, ownership, and
    database persistence belong in ``DocumentService``.
    """

    def __init__(self, settings: Settings) -> None:
        self._base_path = Path(settings.document_storage_path)

    @property
    def base_path(self) -> Path:
        """Return the configured storage root directory."""
        return self._base_path

    def generate_storage_filename(self) -> str:
        """Return a unique filename preserving the PDF extension."""
        return f"{uuid4()}{PDF_EXTENSION}"

    def build_storage_path(self, storage_filename: str) -> str:
        """Return the relative storage path for a generated filename."""
        return str(self._base_path / storage_filename).replace("\\", "/")

    async def save(self, content: bytes, storage_filename: str) -> str:
        """Write ``content`` to disk and return the relative storage path."""
        destination = self._base_path / storage_filename
        await asyncio.to_thread(self._write_bytes, destination, content)
        storage_path = self.build_storage_path(storage_filename)
        logger.debug("Saved file to %s (%d bytes)", storage_path, len(content))
        return storage_path

    async def delete(self, storage_path: str) -> None:
        """Delete a file at ``storage_path`` if it exists."""
        file_path = Path(storage_path)
        if not file_path.is_file():
            logger.warning("Storage delete skipped; file not found: %s", storage_path)
            return
        await asyncio.to_thread(file_path.unlink)
        logger.debug("Deleted file at %s", storage_path)

    async def read(self, storage_path: str) -> bytes:
        """Read and return the raw bytes for a stored file."""
        file_path = Path(storage_path)
        return await asyncio.to_thread(file_path.read_bytes)

    @staticmethod
    def _write_bytes(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
