"""Tests for the retry policy in MemoryExtractorService.extract_and_store.

Covers the four scenarios required by the Phase 19 audit:
1. Successful extraction on first attempt
2. Transient failure followed by success on a later attempt
3. All retry attempts exhausted (permanent transient failure)
4. Non-transient error is NOT retried
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.services.memory_extractor import (
    ExtractedFacts,
    FactEvaluation,
    MemoryExtractorService,
    _is_transient,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user() -> MagicMock:
    u = MagicMock()
    u.id = "user-retry-test"
    return u


@pytest.fixture
def llm() -> AsyncMock:
    return AsyncMock()


def _make_extractor(llm: AsyncMock, **kwargs: object) -> MemoryExtractorService:
    """Return a MemoryExtractorService with accelerated retry timing."""
    memory_service = AsyncMock()
    memory_service.search.return_value = []
    return MemoryExtractorService(
        llm_provider=llm,
        memory_service=memory_service,
        # Use zero delay so tests run instantly.
        base_delay_s=0.0,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# _is_transient unit tests
# ---------------------------------------------------------------------------


class TestIsTransient:
    def test_service_unavailable_is_transient(self) -> None:
        assert _is_transient(ServiceUnavailableError("LLM down"))

    def test_oserror_is_transient(self) -> None:
        assert _is_transient(OSError("connection reset"))

    def test_timeout_is_transient(self) -> None:
        assert _is_transient(TimeoutError())

    def test_bad_request_is_not_transient(self) -> None:
        assert not _is_transient(BadRequestError("bad input"))

    def test_generic_cortex_error_is_not_transient(self) -> None:
        from cortex.core.exceptions import NotFoundError

        assert not _is_transient(NotFoundError())

    def test_pydantic_validation_error_is_not_transient(self) -> None:
        import pydantic

        try:
            pydantic.TypeAdapter(int).validate_python("not-an-int")
        except pydantic.ValidationError as exc:
            assert not _is_transient(exc)

    def test_unknown_exception_defaults_to_transient(self) -> None:
        """Unknown errors get one more chance by default."""
        assert _is_transient(RuntimeError("mystery"))


# ---------------------------------------------------------------------------
# 1. Successful extraction on first attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_extraction_on_first_attempt(
    llm: AsyncMock, user: MagicMock
) -> None:
    """Nominal path: extraction succeeds immediately with no retries."""
    llm.generate_structured.side_effect = [
        ExtractedFacts(facts=["User likes async Python"]),
        FactEvaluation(action="create", target_memory_id=None),
    ]
    extractor = _make_extractor(llm)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await extractor.extract_and_store(
            user=user,
            question="I like async Python",
            answer="Noted.",
            memory_hits=[],
        )

    # No retry delay should have been invoked.
    mock_sleep.assert_not_called()
    # LLM was called once for extraction; no candidates → no evaluation call.
    assert llm.generate_structured.call_count == 1
    # Memory was created.
    extractor._memory.create.assert_called_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 2. Transient failure followed by success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_failure_then_success(
    llm: AsyncMock, user: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """First attempt raises ServiceUnavailableError; second succeeds."""
    llm.generate_structured.side_effect = [
        # Attempt 1: transient failure
        ServiceUnavailableError("LLM timeout"),
        # Attempt 2: success path (no candidates → direct create, one LLM call)
        ExtractedFacts(facts=["User prefers dark mode"]),
    ]
    extractor = _make_extractor(llm, max_attempts=3)

    with (
        caplog.at_level(logging.INFO, logger="cortex.services.memory_extractor"),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await extractor.extract_and_store(
            user=user,
            question="I prefer dark mode",
            answer="Okay.",
            memory_hits=[],
        )

    # Exactly one retry delay for the first failure.
    mock_sleep.assert_awaited_once()
    # Logged the transient-retry warning (always captured at WARNING).
    assert "transient failure" in caplog.text.lower()
    # Logged success on retry (INFO — captured because of caplog.at_level above).
    assert "succeeded on attempt 2" in caplog.text
    # Memory was ultimately created.
    extractor._memory.create.assert_called_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. All retry attempts exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_attempts_exhausted(
    llm: AsyncMock, user: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """All attempts fail with transient errors; final warning is logged."""
    llm.generate_structured.side_effect = ServiceUnavailableError("LLM down")
    extractor = _make_extractor(llm, max_attempts=3)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # Must not raise — background tasks must be non-fatal.
        await extractor.extract_and_store(
            user=user,
            question="anything",
            answer="anything",
            memory_hits=[],
        )

    # Two sleep calls: after attempt 1 and after attempt 2.
    assert mock_sleep.await_count == 2
    # Final exhausted warning was logged.
    assert "failed after 3 attempts" in caplog.text
    # Memory operations were never reached.
    extractor._memory.create.assert_not_called()  # type: ignore[attr-defined]
    extractor._memory.update.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 4. Non-transient error is NOT retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_transient_error_not_retried(
    llm: AsyncMock, user: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """BadRequestError (non-transient) aborts immediately without any retry."""
    llm.generate_structured.side_effect = BadRequestError("invalid schema")
    extractor = _make_extractor(llm, max_attempts=3)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await extractor.extract_and_store(
            user=user,
            question="anything",
            answer="anything",
            memory_hits=[],
        )

    # No sleep/retry should have occurred.
    mock_sleep.assert_not_called()
    # LLM was called exactly once — no retry.
    llm.generate_structured.assert_called_once()
    # Logged the non-transient abort.
    assert "non-transient" in caplog.text.lower()
    # Memory operations were never reached.
    extractor._memory.create.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 5. OsError is retried (additional transient case)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oserror_is_retried(
    llm: AsyncMock, user: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """OSError (e.g. DB connection reset) is treated as transient."""
    llm.generate_structured.side_effect = [
        OSError("connection reset by peer"),
        ExtractedFacts(facts=[]),  # second attempt: no facts, clean exit
    ]
    extractor = _make_extractor(llm, max_attempts=3)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await extractor.extract_and_store(
            user=user,
            question="q",
            answer="a",
            memory_hits=[],
        )

    mock_sleep.assert_awaited_once()
    assert "transient failure" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 6. Pydantic ValidationError is not retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pydantic_validation_error_not_retried(
    llm: AsyncMock, user: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """pydantic.ValidationError is a programming/data error — never retried."""
    import pydantic

    try:
        pydantic.TypeAdapter(int).validate_python("not-an-int")
        validation_exc: pydantic.ValidationError | None = None
    except pydantic.ValidationError as exc:
        validation_exc = exc

    assert validation_exc is not None
    llm.generate_structured.side_effect = validation_exc
    extractor = _make_extractor(llm, max_attempts=3)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await extractor.extract_and_store(
            user=user,
            question="q",
            answer="a",
            memory_hits=[],
        )

    mock_sleep.assert_not_called()
    llm.generate_structured.assert_called_once()
    assert "non-transient" in caplog.text.lower()


# ---------------------------------------------------------------------------
# 7. extract_and_store remains non-fatal (never raises)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_and_store_never_raises(
    llm: AsyncMock, user: MagicMock
) -> None:
    """Even with all attempts failing, the coroutine must not propagate."""
    llm.generate_structured.side_effect = RuntimeError("catastrophic")
    extractor = _make_extractor(llm, max_attempts=2)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Must complete without raising.
        result = await extractor.extract_and_store(
            user=user,
            question="q",
            answer="a",
            memory_hits=[],
        )

    assert result is None  # always returns None
