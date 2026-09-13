"""Agent streaming endpoint — POST /api/v1/agent/run/stream.

Streams the agent execution lifecycle as Server-Sent Events:

* ``tool_call``   — LLM selected a tool (before execution)
* ``tool_result`` — tool returned its observation (preview only)
* ``token``       — one chunk of the final LLM answer
* ``done``        — run complete with full metadata
* ``error``       — unrecoverable failure; stream ends

This endpoint is additive: the existing non-streaming POST /api/v1/agent/run
is completely unchanged.

SSE wire format::

    data: {"type": "...", ...}\\n\\n

A client using ``EventSource`` in the browser or ``httpx`` with streaming
can consume events as they arrive.

Error handling:
- ``BadRequestError`` (empty question) → ``error`` event, stream closes.
- ``ForbiddenError`` / ``NotFoundError`` (ownership) → ``error`` event, stream closes.
- ``ServiceUnavailableError`` (LLM down) → ``error`` event, stream closes.
- Client disconnect: Starlette closes the async generator via ``GeneratorExit``
  (not ``asyncio.CancelledError``).  If the generator is cancelled via an
  explicit ``Task.cancel()``, ``CancelledError`` is logged and **re-raised**
  so the event loop can propagate cancellation to the parent task.
  In both cases the ``finally`` block commits any pending DB writes.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from cortex.agent.events import ErrorEvent, format_sse
from cortex.api.deps import AgentServiceDep, CurrentActiveUserDep, SessionDep
from cortex.schemas.agent import AgentRunRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/run/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream agent execution via Server-Sent Events",
    description=(
        "Executes the same agent + tool-calling loop as POST /agent/run but "
        "streams events incrementally using Server-Sent Events (SSE).  "
        "Connect with an EventSource-compatible client.  "
        "Events: tool_call, tool_result, token, done, error."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"description": "SSE stream of agent execution events"},
        400: {"description": "Empty question"},
        401: {"description": "Not authenticated"},
        403: {"description": "Conversation not owned by user"},
        404: {"description": "Conversation not found"},
        422: {"description": "Validation error"},
    },
)
async def agent_run_stream(
    payload: AgentRunRequest,
    current_user: CurrentActiveUserDep,
    agent_service: AgentServiceDep,
    session: SessionDep,
) -> StreamingResponse:
    """Stream agent execution events for the given question."""

    async def _event_generator():
        try:
            async for event in agent_service.stream(
                question=payload.message,
                user=current_user,
                conversation_id=payload.conversation_id,
            ):
                yield format_sse(event)
                await asyncio.sleep(0)  # yield control to event loop
        except asyncio.CancelledError:
            logger.info(
                "Agent stream cancelled by client: user_id=%s", current_user.id
            )
            raise  # must propagate so the event loop can cancel the parent task
        except Exception as exc:
            logger.exception(
                "Unexpected error in agent stream: user_id=%s", current_user.id
            )
            yield format_sse(ErrorEvent(message=f"Internal error: {exc}"))
            return
        finally:
            # Commit any pending DB writes (user message, assistant message)
            try:
                await session.commit()
            except Exception:
                logger.exception("Failed to commit session after agent stream")

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
