"""Unit tests for StorageService."""

from __future__ import annotations

from pathlib import Path

import pytest

from cortex.core.config import Settings
from cortex.services.storage import PDF_EXTENSION, StorageService


@pytest.fixture
def storage_service(test_settings: Settings) -> StorageService:
    """StorageService bound to the pytest temporary directory."""
    return StorageService(test_settings)


@pytest.mark.asyncio
async def test_generate_storage_filename_uses_pdf_extension(
    storage_service: StorageService,
) -> None:
    """Generated filenames always end with the PDF extension."""
    filename = storage_service.generate_storage_filename()
    assert filename.endswith(PDF_EXTENSION)
    assert len(filename) > len(PDF_EXTENSION)


@pytest.mark.asyncio
async def test_save_and_read_round_trip(
    storage_service: StorageService,
    minimal_pdf_bytes: bytes,
) -> None:
    """Saved files can be read back from disk."""
    storage_filename = storage_service.generate_storage_filename()
    storage_path = await storage_service.save(minimal_pdf_bytes, storage_filename)

    assert Path(storage_path).is_file()
    content = await storage_service.read(storage_path)
    assert content == minimal_pdf_bytes


@pytest.mark.asyncio
async def test_delete_removes_file(
    storage_service: StorageService,
    minimal_pdf_bytes: bytes,
) -> None:
    """Delete removes an existing stored file."""
    storage_filename = storage_service.generate_storage_filename()
    storage_path = await storage_service.save(minimal_pdf_bytes, storage_filename)

    await storage_service.delete(storage_path)
    assert not Path(storage_path).exists()


@pytest.mark.asyncio
async def test_delete_missing_file_is_noop(storage_service: StorageService) -> None:
    """Deleting a missing file does not raise."""
    await storage_service.delete("storage/documents/does-not-exist.pdf")


def test_build_storage_path_uses_configured_root(
    storage_service: StorageService,
    test_settings: Settings,
) -> None:
    """Storage paths are rooted under the configured directory."""
    from pathlib import Path

    path = storage_service.build_storage_path("abc.pdf")
    expected = Path(test_settings.document_storage_path) / "abc.pdf"
    assert Path(path).as_posix() == expected.as_posix()
