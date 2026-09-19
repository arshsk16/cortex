"""Agent endpoint — POST /api/v1/agent/run."""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, status

from cortex.api.deps import AgentServiceDep, CurrentActiveUserDep, SessionDep
from cortex.schemas.agent import (
    AgentCitationRead,
    AgentResponse,
    AgentRunRequest,
    AgentToolCallRead,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/run",
    response_model=AgentResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the Cortex agent on a user question",
    description=(
        "Executes a single-agent + tool-calling loop.  The agent autonomously "
        "decides which tools to call (RAG search, document list, calculator), "
        "accumulates retrieval results, then produces a grounded final answer "
        "using the existing RAG pipeline.  "
        "Optionally linked to a conversation for history persistence — "
        "conversation ownership is enforced identically to the /chat endpoint."
    ),
    responses={
        200: {"description": "Grounded answer with tool trace and citations"},
        400: {"description": "Empty question"},
        401: {"description": "Not authenticated"},
        403: {"description": "Conversation not owned by user"},
        404: {"description": "Conversation not found"},
        503: {"description": "LLM generation failed"},
        422: {"description": "Validation error"},
    },
)
async def agent_run(
    payload: AgentRunRequest,
    current_user: CurrentActiveUserDep,
    agent_service: AgentServiceDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
) -> AgentResponse:
    """Run the agent and return the grounded answer with tool trace."""
    result = await agent_service.run(
        question=payload.message,
        user=current_user,
        conversation_id=payload.conversation_id,
        background_tasks=background_tasks,
    )

    await session.commit()

    tool_calls = [
        AgentToolCallRead(
            tool_name=tc.tool_name,
            args=tc.args,
            observation=tc.observation,
        )
        for tc in result.tool_calls
    ]

    citations = [
        AgentCitationRead(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
        )
        for chunk in result.retrieved_chunks
    ]

    return AgentResponse(
        answer=result.answer,
        tool_calls_made=tool_calls,
        citations=citations,
        conversation_id=payload.conversation_id,
    )
