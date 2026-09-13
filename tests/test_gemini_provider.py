"""Unit tests for GeminiProvider (all external calls mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cortex.core.config import Settings
from cortex.core.exceptions import ServiceUnavailableError
from cortex.llm.gemini import GeminiProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "database_url": "postgresql+asyncpg://u:p@localhost/db",
        "jwt_secret_key": "test-secret-key-that-is-at-least-32-chars",
        "gemini_api_key": "fake-api-key",
        "gemini_model_name": "gemini-2.0-flash",
        "gemini_temperature": 0.2,
        "gemini_max_output_tokens": 512,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _make_provider(settings: Settings | None = None) -> GeminiProvider:
    return GeminiProvider(settings or _make_settings())


def _make_response(text: str) -> MagicMock:
    """Return a mock object resembling a Gemini GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_model_text() -> None:
    """generate() returns the stripped text from the Gemini response."""
    provider = _make_provider()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_response(
        "  The capital is Paris.  "
    )
    provider._client = mock_client

    result = await provider.generate("What is the capital of France?")

    assert result == "The capital is Paris."
    mock_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_passes_model_name_and_prompt() -> None:
    """generate() forwards the correct model name and prompt to the SDK."""
    provider = _make_provider()

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_response("answer")
    provider._client = mock_client

    await provider.generate("My prompt text")

    call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs.kwargs["contents"] == "My prompt text"


@pytest.mark.asyncio
async def test_generate_wraps_sdk_exception_as_service_unavailable() -> None:
    """SDK errors are wrapped in ServiceUnavailableError."""
    provider = _make_provider()

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    provider._client = mock_client

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await provider.generate("Some prompt")

    assert "LLM generation failed" in str(exc_info.value)
    assert exc_info.value.code == "service_unavailable"


@pytest.mark.asyncio
async def test_generate_lazy_initialises_client() -> None:
    """The google.genai client is created on first call, not in __init__."""
    settings = _make_settings()
    provider = GeminiProvider(settings)

    # Client should not exist yet
    assert provider._client is None

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _make_response("hi")

    target = "cortex.llm.gemini.GeminiProvider._get_client"
    with patch(target, return_value=fake_client):
        await provider.generate("hello")

    # Client creation is delegated; we just confirm generate called without error
    fake_client.models.generate_content.assert_called_once()


@pytest.mark.asyncio
async def test_generate_uses_configured_temperature_and_tokens() -> None:
    """generate() passes temperature and max_output_tokens from settings."""

    settings = _make_settings(gemini_temperature=0.7, gemini_max_output_tokens=256)
    provider = _make_provider(settings)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _make_response("ok")
    provider._client = mock_client

    await provider.generate("prompt")

    config_arg = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config_arg.temperature == pytest.approx(0.7)
    assert config_arg.max_output_tokens == 256


# ---------------------------------------------------------------------------
# generate_stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_yields_chunks() -> None:
    """generate_stream() yields text parts from the SDK streaming response."""
    provider = _make_provider()

    chunk_a = MagicMock()
    chunk_a.text = "Hello, "
    chunk_b = MagicMock()
    chunk_b.text = "world!"

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.return_value = iter([chunk_a, chunk_b])
    provider._client = mock_client

    stream = await provider.generate_stream("Stream prompt")
    parts = [part async for part in stream]

    assert parts == ["Hello, ", "world!"]


@pytest.mark.asyncio
async def test_generate_stream_wraps_sdk_exception() -> None:
    """Streaming SDK errors are wrapped in ServiceUnavailableError."""
    provider = _make_provider()

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = ConnectionError("timeout")
    provider._client = mock_client

    with pytest.raises(ServiceUnavailableError):
        stream = await provider.generate_stream("prompt")
        _ = [part async for part in stream]
