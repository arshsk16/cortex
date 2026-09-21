"""Agent orchestration service - single-agent + tool-calling loop.

Architecture (Phase 14)
-----------------------
AgentService.run() has four phases, all sharing a single AgentState object:

1. Build state - security gate, load bounded conversation history,
   persist user message, retrieve long-term memories.

2. Tool-calling loop - dispatched to one of two strategies:
   * Native path: used when LLMProvider implements SupportsToolCalling.
   * Prompt-based fallback: optionally runs AgentPlanner, parses JSON decisions.

   Both strategies apply Phase 14 improvements:
   - Deduplication: duplicate tool calls are skipped with a synthetic observation.
   - Status tracking: each ToolCallRecord records status and duration_ms.
   - Budget tracking: budget_exhausted set when max_tool_calls is hit.

3. Grounded final answer - PromptBuilder assembles a structured prompt.
   When budget_exhausted, a note instructs best-effort synthesis.
   A compact tool_call_summary() is appended to improve synthesis quality.

4. Persistence - assistant message, token usage, ephemeral state cleanup,
   and memory extraction background task (Phase 13C).

Streaming (stream()) mirrors run() but emits SSE events and streams
the final answer token-by-token.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastapi import BackgroundTasks

from cortex.agent.events import AgentEvent
from cortex.agent.prompt import AgentPromptBuilder
from cortex.agent.registry import ToolRegistry
from cortex.agent.result import (
    TOOL_STATUS_ERROR,
    TOOL_STATUS_OK,
    TOOL_STATUS_SKIPPED,
    AgentResult,
    ToolCallRecord,
)
from cortex.agent.state import AgentState
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import AgentMessage, ToolCallRequest, ToolResult
from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.services.memory_extractor import MemoryExtractorService
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.agent.planner import AgentPlanner
    from cortex.db.models.user import User
    from cortex.llm.base import LLMProvider
    from cortex.services.conversation import ConversationService
    from cortex.services.memory import MemoryService
    from cortex.state_store.base import StateStore

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_CALLS = 5
_DEFAULT_HISTORY_LIMIT = 10
_DEFAULT_STATE_TTL = 1800  # 30 minutes
_DEFAULT_MEMORY_LIMIT = 5  # max long-term memory hits per run

# Injected when the LLM tries to call the same tool with the same args again.
_DUPLICATE_TOOL_MSG = (
    "Tool already called with these arguments in this run. "
    "Use the observation already available above."
)

# Max chars injected per tool observation into the prompt (Phase 14).
_MAX_OBSERVATION_CHARS = 4000


def _freeze_args(args: dict[str, Any]) -> str:
    """Return a stable string key for a tool-args dict (deduplication)."""
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(sorted(args.items()))


class AgentService:
    """Orchestrate a single-agent + tool-calling loop with conversation memory.

    Parameters
    ----------
    llm_provider:
        The application-scoped LLM provider.
    tool_registry:
        Registry pre-populated with the available tools.
    prompt_builder:
        The existing PromptBuilder used for the final grounded-answer step.
    conversation_service:
        Optional conversation service for history and persistence.
    max_tool_calls:
        Maximum number of tool invocations per agent run.
    conversation_history_limit:
        Maximum number of previous messages to load.
    state_store:
        Optional ephemeral state store (Redis or Null).
    state_ttl_seconds:
        TTL for Redis state snapshots.
    memory_service:
        Optional MemoryService for long-term memory retrieval (Phase 13B).
    memory_retrieval_limit:
        Maximum number of memory entries to retrieve per run.
    memory_extractor:
        Optional Phase 13C MemoryExtractorService.
    planner:
        Optional Phase 14 AgentPlanner. When None (default), planning is
        disabled. Only used on the prompt-based path; skipped for native
        tool-calling providers.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        state_store: StateStore | None = None,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
        conversation_history_limit: int = _DEFAULT_HISTORY_LIMIT,
        state_ttl_seconds: int = _DEFAULT_STATE_TTL,
        memory_service: MemoryService | None = None,
        memory_retrieval_limit: int = _DEFAULT_MEMORY_LIMIT,
        memory_extractor: MemoryExtractorService | None = None,
        planner: AgentPlanner | None = None,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._state_store = state_store
        self._max_tool_calls = max_tool_calls
        self._history_limit = conversation_history_limit
        self._state_ttl = state_ttl_seconds
        self._memory_service = memory_service
        self._memory_limit = memory_retrieval_limit
        self._memory_extractor = memory_extractor
        self._planner = planner
        self._agent_prompt_builder = AgentPromptBuilder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None = None,
        background_tasks: BackgroundTasks | None = None,
    ) -> AgentResult:
        """Run the agent on a user question and return the grounded answer."""
        question = question.strip()
        if not question:
            raise BadRequestError(
                "Agent question must not be empty",
                details={"field": "question"},
            )

        state = await self._build_state(
            question=question,
            user=user,
            conversation_id=conversation_id,
        )

        try:
            from cortex.llm.base import SupportsToolCalling

            if isinstance(self._llm, SupportsToolCalling):
                logger.debug(
                    "Agent using native tool-calling path (provider=%s)",
                    type(self._llm).__name__,
                )
                await self._native_tool_loop(state=state)
            else:
                logger.debug(
                    "Agent using prompt-based tool-calling path (provider=%s)",
                    type(self._llm).__name__,
                )
                await self._prompt_tool_loop(state=state)

            # Phase 14: augment question with tool-call summary for synthesis
            question_with_summary = self._augment_question_with_summary(question, state)
            grounded_prompt = self._prompt_builder.build(
                question=question_with_summary,
                retrieved_chunks=state.retrieved_chunks,
                history=state.history if state.has_history else None,
                memory_hits=state.memory_hits if state.has_memory_hits else None,
            )
            if state.budget_exhausted:
                grounded_prompt += (
                    "\n\n[NOTE: The tool-call budget was exhausted. "
                    "Synthesise a best-effort answer from the above context. "
                    "Acknowledge any gaps honestly.]"
                )

            try:
                final_answer = await self._llm.generate(grounded_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM final-answer generation failed",
                    details={"reason": str(exc)},
                ) from exc

            logger.info(
                "Agent run complete: user_id=%s conversation_id=%s "
                "tool_calls=%d chunks=%d history=%d "
                "budget_exhausted=%s elapsed_ms=%.1f",
                user.id,
                conversation_id,
                state.total_tool_calls,
                len(state.retrieved_chunks),
                len(state.history),
                state.budget_exhausted,
                state.elapsed_ms,
            )

            if conversation_id and self._conv_service:
                citation_dicts = [
                    {
                        "document_id": c.document_id,
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                    }
                    for c in state.retrieved_chunks
                ]
                await self._conv_service.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                    citations=citation_dicts or None,
                )
                prompt_tokens = len(grounded_prompt) // 4
                completion_tokens = len(final_answer) // 4
                await self._conv_service.record_token_usage(
                    conversation_id=conversation_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

            await self._delete_state(state)

            if background_tasks is not None and self._memory_extractor is not None:
                background_tasks.add_task(
                    self._memory_extractor.extract_and_store,
                    user=user,
                    question=question,
                    answer=final_answer,
                    memory_hits=state.memory_hits,
                )

            return AgentResult(
                answer=final_answer,
                tool_calls=state.tool_calls,
                retrieved_chunks=state.retrieved_chunks,
            )
        except Exception as exc:
            await self._save_state(state, status="failed", error=str(exc))
            raise

    async def stream(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None = None,
        background_tasks: BackgroundTasks | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent and yield SSE events incrementally."""
        from cortex.agent.events import DoneEvent, ErrorEvent, TokenEvent
        from cortex.llm.base import SupportsToolCalling

        question = question.strip()
        if not question:
            yield ErrorEvent(message="Agent question must not be empty")
            return

        try:
            state = await self._build_state(
                question=question,
                user=user,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            yield ErrorEvent(message=str(exc))
            return

        try:
            if isinstance(self._llm, SupportsToolCalling):
                async for event in self._native_tool_loop_stream(state=state):
                    yield event
            else:
                async for event in self._prompt_tool_loop_stream(state=state):
                    yield event

            question_with_summary = self._augment_question_with_summary(question, state)
            grounded_prompt = self._prompt_builder.build(
                question=question_with_summary,
                retrieved_chunks=state.retrieved_chunks,
                history=state.history if state.has_history else None,
                memory_hits=state.memory_hits if state.has_memory_hits else None,
            )
            if state.budget_exhausted:
                grounded_prompt += (
                    "\n\n[NOTE: The tool-call budget was exhausted. "
                    "Synthesise a best-effort answer from the above context. "
                    "Acknowledge any gaps honestly.]"
                )

            final_parts: list[str] = []
            try:
                token_stream = await self._llm.generate_stream(grounded_prompt)
                async for token in token_stream:
                    if token:
                        final_parts.append(token)
                        yield TokenEvent(text=token)
            except ServiceUnavailableError as exc:
                await self._save_state(state, status="failed", error=str(exc))
                yield ErrorEvent(message=f"LLM streaming failed: {exc}")
                return
            except asyncio.CancelledError:
                await self._save_state(
                    state, status="cancelled", error="Client disconnected"
                )
                raise
            except Exception as exc:
                await self._save_state(state, status="failed", error=str(exc))
                yield ErrorEvent(message=f"LLM streaming failed: {exc}")
                return

            final_answer = "".join(final_parts)

            logger.info(
                "Agent stream complete: user_id=%s conversation_id=%s "
                "tool_calls=%d chunks=%d history=%d "
                "budget_exhausted=%s elapsed_ms=%.1f",
                user.id,
                conversation_id,
                state.total_tool_calls,
                len(state.retrieved_chunks),
                len(state.history),
                state.budget_exhausted,
                state.elapsed_ms,
            )

            if conversation_id and self._conv_service:
                citation_dicts = [
                    {
                        "document_id": c.document_id,
                        "chunk_id": c.chunk_id,
                        "chunk_index": c.chunk_index,
                    }
                    for c in state.retrieved_chunks
                ]
                await self._conv_service.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                    citations=citation_dicts or None,
                )
                prompt_tokens = len(grounded_prompt) // 4
                completion_tokens = len(final_answer) // 4
                await self._conv_service.record_token_usage(
                    conversation_id=conversation_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

            await self._delete_state(state)

            tool_calls_made = [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "observation": tc.observation,
                }
                for tc in state.tool_calls
            ]
            citations = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in state.retrieved_chunks
            ]
            if background_tasks is not None and self._memory_extractor is not None:
                background_tasks.add_task(
                    self._memory_extractor.extract_and_store,
                    user=user,
                    question=question,
                    answer=final_answer,
                    memory_hits=state.memory_hits,
                )

            yield DoneEvent(
                answer=final_answer,
                tool_calls_made=tool_calls_made,
                citations=citations,
                conversation_id=conversation_id,
            )
        except asyncio.CancelledError:
            await self._save_state(
                state, status="cancelled", error="Client disconnected"
            )
            raise
        except Exception as exc:
            await self._save_state(state, status="failed", error=str(exc))
            yield ErrorEvent(message=f"Tool loop failed: {exc}")
            return

    # ------------------------------------------------------------------
    # Private - State construction
    # ------------------------------------------------------------------

    async def _build_state(
        self,
        *,
        question: str,
        user: User,
        conversation_id: str | None,
    ) -> AgentState:
        """Build the initial AgentState for this run.

        Performs three operations in order:

        1. Load bounded conversation history (ownership enforced by
           ConversationService.get_history).
        2. Persist the incoming user message to the conversation (if one
           was provided).
        3. Retrieve long-term semantic memory hits for the current user
           and question via MemoryService (Phase 13B).  Failure is non-fatal.
        """
        history = []

        if conversation_id and self._conv_service:
            history = await self._conv_service.get_history(
                conversation_id=conversation_id,
                user_id=user.id,
                limit=self._history_limit,
            )
            logger.debug(
                "Loaded conversation history: conversation_id=%s messages=%d",
                conversation_id,
                len(history),
            )
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            await self._conv_service.set_auto_title_if_needed(
                conversation_id=conversation_id,
                first_message=question,
            )

        memory_hits = []
        if self._memory_service is not None and self._memory_limit > 0:
            try:
                memory_hits = await self._memory_service.search(
                    user=user,
                    query=question,
                    limit=self._memory_limit,
                )
                logger.debug(
                    "Retrieved %d long-term memory hits for user_id=%s",
                    len(memory_hits),
                    user.id,
                )
            except Exception:
                logger.warning(
                    "AgentService: memory retrieval failed; continuing without "
                    "long-term memory (user_id=%s)",
                    user.id,
                    exc_info=True,
                )

        state = AgentState(
            question=question,
            user=user,
            conversation_id=conversation_id,
            history=history,
            memory_hits=memory_hits,
        )
        await self._save_state(state, status="in_progress")
        return state

    # ------------------------------------------------------------------
    # Private - Phase 14 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_question_with_summary(question: str, state: AgentState) -> str:
        """Append compact tool-call summary to the question for grounded answer."""
        summary = state.tool_call_summary()
        if not summary:
            return question
        return (
            f"{question}\n\n"
            f"=== TOOL CALLS MADE ===\n{summary}\n=== END OF TOOL CALLS ==="
        )

    # ------------------------------------------------------------------
    # Private - Native tool-calling loop (Phase 9, updated Phase 14)
    # ------------------------------------------------------------------

    async def _native_tool_loop(self, *, state: AgentState) -> None:
        """Native tool-calling loop with Phase 14 deduplication and tracking."""
        from cortex.llm.base import SupportsToolCalling

        assert isinstance(self._llm, SupportsToolCalling)
        messages: list[AgentMessage] = [AgentMessage(role="user", text=state.question)]
        seen_calls: set[tuple[str, str]] = set()

        for iteration in range(self._max_tool_calls):
            logger.debug(
                "Agent native loop iteration=%d user_id=%s",
                iteration + 1,
                state.user.id,
            )

            try:
                result = await self._llm.generate_with_tools(
                    messages=messages,
                    tool_schemas=self._registry.tool_schemas,
                )
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generate_with_tools failed during agent loop",
                    details={"reason": str(exc)},
                ) from exc

            if result.is_text:
                logger.info(
                    "Agent native loop: text response after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            tc: ToolCallRequest = result.tool_call  # type: ignore[assignment]
            tool_name = tc.tool_name
            frozen = (tool_name, _freeze_args(tc.args))
            messages.append(AgentMessage(role="model", tool_call=tc))

            if frozen in seen_calls:
                logger.info(
                    "Agent native loop: skipping duplicate call to %r (iteration=%d)",
                    tool_name,
                    iteration + 1,
                )
                observation = _DUPLICATE_TOOL_MSG
                state.tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tc.args,
                        observation=observation,
                        status=TOOL_STATUS_SKIPPED,
                        duration_ms=0.0,
                    )
                )
                messages.append(
                    AgentMessage(
                        role="tool",
                        tool_result=ToolResult(
                            tool_name=tool_name,
                            output=observation,
                            call_id=tc.call_id,
                        ),
                    )
                )
                await self._save_state(state, status="in_progress")
                continue

            seen_calls.add(frozen)

            t0 = time.perf_counter()
            tc_status = TOOL_STATUS_OK
            try:
                observation = await self._registry.dispatch(
                    name=tool_name,
                    args=tc.args,
                    user=state.user,
                )
            except BadRequestError as exc:
                observation = f"Tool error: {exc.message}"
                tc_status = TOOL_STATUS_ERROR
                logger.warning(
                    "Agent native loop: tool %r raised BadRequestError: %s",
                    tool_name,
                    exc.message,
                )
            duration_ms = (time.perf_counter() - t0) * 1000

            tool_obj = None
            with contextlib.suppress(BadRequestError):
                tool_obj = self._registry.get_tool(tool_name)
            if isinstance(tool_obj, RAGSearchTool):
                state.retrieved_chunks.extend(tool_obj.last_results)

            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    args=tc.args,
                    observation=observation,
                    status=tc_status,
                    duration_ms=duration_ms,
                )
            )
            await self._save_state(state, status="in_progress")

            messages.append(
                AgentMessage(
                    role="tool",
                    tool_result=ToolResult(
                        tool_name=tool_name,
                        output=observation,
                        call_id=tc.call_id,
                    ),
                )
            )
        else:
            logger.warning(
                "Agent native loop: reached max_tool_calls=%d without text response",
                self._max_tool_calls,
            )
            state.budget_exhausted = True

    # ------------------------------------------------------------------
    # Private - Prompt-based tool-calling loop (Phase 8 fallback, Phase 14)
    # ------------------------------------------------------------------

    async def _prompt_tool_loop(self, *, state: AgentState) -> None:
        """Prompt-based tool-calling loop with Phase 14 improvements."""
        if self._planner is not None:
            try:
                task_plan = await self._planner.plan(
                    question=state.question,
                    tool_schemas=self._registry.tool_schemas,
                    has_memory=state.has_memory_hits,
                    has_history=state.has_history,
                )
                state.plan = task_plan.steps
                logger.debug(
                    "AgentPlanner: steps=%r rationale=%r confidence=%.2f",
                    task_plan.steps,
                    task_plan.rationale,
                    task_plan.confidence,
                )
            except Exception:
                logger.warning(
                    "AgentService: planner error; continuing without plan",
                    exc_info=True,
                )

        context_flags = {
            "has_history": state.has_history,
            "has_memory": state.has_memory_hits,
        }
        current_prompt = self._agent_prompt_builder.build_initial(
            question=state.question,
            tool_schemas=self._registry.tool_schemas,
            context_flags=context_flags,
        )

        seen_calls: set[tuple[str, str]] = set()
        loop_count = 0

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent prompt loop iteration=%d user_id=%s",
                loop_count,
                state.user.id,
            )

            try:
                raw_response = await self._llm.generate(current_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generation failed during agent loop",
                    details={"reason": str(exc)},
                ) from exc

            decision = self._parse_decision(raw_response)
            action = decision.get("action", "")

            if action == "final_answer":
                logger.info(
                    "Agent prompt loop: final_answer after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})
                frozen = (tool_name, _freeze_args(tool_args))

                if frozen in seen_calls:
                    logger.info(
                        "Agent prompt loop: skipping duplicate"
                        " call to %r (iteration=%d)",
                        tool_name,
                        loop_count,
                    )
                    observation = _DUPLICATE_TOOL_MSG
                    state.tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            args=tool_args,
                            observation=observation,
                            status=TOOL_STATUS_SKIPPED,
                            duration_ms=0.0,
                        )
                    )
                    await self._save_state(state, status="in_progress")
                    current_prompt = self._agent_prompt_builder.build_observation_turn(
                        previous_prompt=current_prompt,
                        llm_decision=raw_response,
                        tool_name=tool_name,
                        observation=observation,
                        max_observation_chars=_MAX_OBSERVATION_CHARS,
                    )
                    continue

                seen_calls.add(frozen)

                t0 = time.perf_counter()
                tc_status = TOOL_STATUS_OK
                try:
                    observation = await self._registry.dispatch(
                        name=tool_name,
                        args=tool_args,
                        user=state.user,
                    )
                except BadRequestError as exc:
                    observation = f"Tool error: {exc.message}"
                    tc_status = TOOL_STATUS_ERROR
                    tool_name = tool_name or "unknown"
                duration_ms = (time.perf_counter() - t0) * 1000

                tool_obj = None
                with contextlib.suppress(BadRequestError):
                    tool_obj = self._registry.get_tool(tool_name)
                if isinstance(tool_obj, RAGSearchTool):
                    state.retrieved_chunks.extend(tool_obj.last_results)

                state.tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tool_args,
                        observation=observation,
                        status=tc_status,
                        duration_ms=duration_ms,
                    )
                )
                await self._save_state(state, status="in_progress")

                current_prompt = self._agent_prompt_builder.build_observation_turn(
                    previous_prompt=current_prompt,
                    llm_decision=raw_response,
                    tool_name=tool_name,
                    observation=observation,
                    max_observation_chars=_MAX_OBSERVATION_CHARS,
                )
            else:
                logger.warning(
                    "Agent prompt loop: unexpected action=%r; treating as final answer",
                    action,
                )
                break
        else:
            logger.warning(
                "Agent prompt loop: reached max_tool_calls=%d without final_answer",
                self._max_tool_calls,
            )
            state.budget_exhausted = True

    # ------------------------------------------------------------------
    # Private - Streaming tool-calling loops (Phase 11, updated Phase 14)
    # ------------------------------------------------------------------

    async def _native_tool_loop_stream(
        self, *, state: AgentState
    ) -> AsyncGenerator[AgentEvent, None]:
        """Native streaming tool loop with Phase 14 deduplication and tracking."""
        from cortex.agent.events import ToolCallEvent, ToolResultEvent
        from cortex.llm.base import SupportsToolCalling

        assert isinstance(self._llm, SupportsToolCalling)
        messages: list[AgentMessage] = [AgentMessage(role="user", text=state.question)]
        seen_calls: set[tuple[str, str]] = set()

        for iteration in range(self._max_tool_calls):
            logger.debug(
                "Agent native stream loop iteration=%d user_id=%s",
                iteration + 1,
                state.user.id,
            )

            try:
                result = await self._llm.generate_with_tools(
                    messages=messages,
                    tool_schemas=self._registry.tool_schemas,
                )
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generate_with_tools failed during agent stream loop",
                    details={"reason": str(exc)},
                ) from exc

            if result.is_text:
                logger.info(
                    "Agent native stream loop: text after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            tc: ToolCallRequest = result.tool_call  # type: ignore[assignment]
            tool_name = tc.tool_name
            frozen = (tool_name, _freeze_args(tc.args))
            messages.append(AgentMessage(role="model", tool_call=tc))

            if frozen in seen_calls:
                logger.info(
                    "Agent native stream loop: skipping duplicate call to %r",
                    tool_name,
                )
                observation = _DUPLICATE_TOOL_MSG
                state.tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tc.args,
                        observation=observation,
                        status=TOOL_STATUS_SKIPPED,
                        duration_ms=0.0,
                    )
                )
                messages.append(
                    AgentMessage(
                        role="tool",
                        tool_result=ToolResult(
                            tool_name=tool_name,
                            output=observation,
                            call_id=tc.call_id,
                        ),
                    )
                )
                await self._save_state(state, status="in_progress")
                continue

            seen_calls.add(frozen)
            yield ToolCallEvent(tool_name=tool_name, args=tc.args)

            t0 = time.perf_counter()
            tc_status = TOOL_STATUS_OK
            try:
                observation = await self._registry.dispatch(
                    name=tool_name,
                    args=tc.args,
                    user=state.user,
                )
            except BadRequestError as exc:
                observation = f"Tool error: {exc.message}"
                tc_status = TOOL_STATUS_ERROR
                logger.warning(
                    "Agent native stream loop: tool %r raised BadRequestError: %s",
                    tool_name,
                    exc.message,
                )
            duration_ms = (time.perf_counter() - t0) * 1000

            tool_obj = None
            with contextlib.suppress(BadRequestError):
                tool_obj = self._registry.get_tool(tool_name)
            if isinstance(tool_obj, RAGSearchTool):
                state.retrieved_chunks.extend(tool_obj.last_results)

            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    args=tc.args,
                    observation=observation,
                    status=tc_status,
                    duration_ms=duration_ms,
                )
            )
            await self._save_state(state, status="in_progress")

            messages.append(
                AgentMessage(
                    role="tool",
                    tool_result=ToolResult(
                        tool_name=tool_name,
                        output=observation,
                        call_id=tc.call_id,
                    ),
                )
            )
            yield ToolResultEvent.from_observation(tool_name, observation)
        else:
            logger.warning(
                "Agent native stream loop: reached max_tool_calls=%d",
                self._max_tool_calls,
            )
            state.budget_exhausted = True

    async def _prompt_tool_loop_stream(
        self, *, state: AgentState
    ) -> AsyncGenerator[AgentEvent, None]:
        """Prompt-based streaming tool loop with Phase 14 improvements."""
        from cortex.agent.events import ToolCallEvent, ToolResultEvent

        if self._planner is not None:
            try:
                task_plan = await self._planner.plan(
                    question=state.question,
                    tool_schemas=self._registry.tool_schemas,
                    has_memory=state.has_memory_hits,
                    has_history=state.has_history,
                )
                state.plan = task_plan.steps
            except Exception:
                logger.warning(
                    "AgentService stream: planner error; continuing without plan",
                    exc_info=True,
                )

        context_flags = {
            "has_history": state.has_history,
            "has_memory": state.has_memory_hits,
        }
        current_prompt = self._agent_prompt_builder.build_initial(
            question=state.question,
            tool_schemas=self._registry.tool_schemas,
            context_flags=context_flags,
        )

        seen_calls: set[tuple[str, str]] = set()
        loop_count = 0

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent prompt stream loop iteration=%d user_id=%s",
                loop_count,
                state.user.id,
            )

            try:
                raw_response = await self._llm.generate(current_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generation failed during agent stream loop",
                    details={"reason": str(exc)},
                ) from exc

            decision = self._parse_decision(raw_response)
            action = decision.get("action", "")

            if action == "final_answer":
                logger.info(
                    "Agent prompt stream loop: final_answer after %d tool call(s)",
                    state.total_tool_calls,
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})
                frozen = (tool_name, _freeze_args(tool_args))

                if frozen in seen_calls:
                    logger.info(
                        "Agent prompt stream loop: skipping duplicate call to %r",
                        tool_name,
                    )
                    observation = _DUPLICATE_TOOL_MSG
                    state.tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            args=tool_args,
                            observation=observation,
                            status=TOOL_STATUS_SKIPPED,
                            duration_ms=0.0,
                        )
                    )
                    await self._save_state(state, status="in_progress")
                    current_prompt = self._agent_prompt_builder.build_observation_turn(
                        previous_prompt=current_prompt,
                        llm_decision=raw_response,
                        tool_name=tool_name,
                        observation=observation,
                        max_observation_chars=_MAX_OBSERVATION_CHARS,
                    )
                    continue

                seen_calls.add(frozen)
                yield ToolCallEvent(tool_name=tool_name or "unknown", args=tool_args)

                t0 = time.perf_counter()
                tc_status = TOOL_STATUS_OK
                try:
                    observation = await self._registry.dispatch(
                        name=tool_name,
                        args=tool_args,
                        user=state.user,
                    )
                except BadRequestError as exc:
                    observation = f"Tool error: {exc.message}"
                    tc_status = TOOL_STATUS_ERROR
                    tool_name = tool_name or "unknown"
                duration_ms = (time.perf_counter() - t0) * 1000

                tool_obj = None
                with contextlib.suppress(BadRequestError):
                    tool_obj = self._registry.get_tool(tool_name)
                if isinstance(tool_obj, RAGSearchTool):
                    state.retrieved_chunks.extend(tool_obj.last_results)

                state.tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tool_args,
                        observation=observation,
                        status=tc_status,
                        duration_ms=duration_ms,
                    )
                )
                await self._save_state(state, status="in_progress")
                yield ToolResultEvent.from_observation(tool_name, observation)

                current_prompt = self._agent_prompt_builder.build_observation_turn(
                    previous_prompt=current_prompt,
                    llm_decision=raw_response,
                    tool_name=tool_name,
                    observation=observation,
                    max_observation_chars=_MAX_OBSERVATION_CHARS,
                )
            else:
                logger.warning(
                    "Agent prompt stream loop: unexpected action=%r",
                    action,
                )
                break
        else:
            logger.warning(
                "Agent prompt stream loop: reached max_tool_calls=%d",
                self._max_tool_calls,
            )
            state.budget_exhausted = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(raw: str) -> dict:
        """Extract a JSON decision dict from the LLM's raw text response."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner_lines = (
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            text = "\n".join(inner_lines).strip()
        try:
            decision = json.loads(text)
            if not isinstance(decision, dict):
                raise ValueError("Expected a JSON object")  # noqa: TRY301
            return decision
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Agent could not parse LLM decision as JSON: %s (raw=%r)",
                exc,
                raw[:300],
            )
            return {"action": "final_answer", "answer": raw.strip()}

    # ------------------------------------------------------------------
    # Ephemeral state-store helpers (Phase 12)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_key(
        user_id: str,
        conversation_id: str | None,
        run_id: str,
    ) -> str:
        """Construct namespaced Redis key for an agent run."""
        conv = conversation_id or "_stateless"
        return f"cortex:agent:run:{user_id}:{conv}:{run_id}"

    @staticmethod
    def _snapshot(
        state: AgentState,
        *,
        status: str = "in_progress",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Serialise AgentState to an ephemeral, JSON-safe dictionary.

        Phase 14: includes plan, budget_exhausted, and per-tool
        status/duration_ms for richer state inspection.
        """
        return {
            "run_id": state.run_id,
            "user_id": str(state.user.id),
            "conversation_id": state.conversation_id,
            "question": state.question,
            "started_at_iso": getattr(state, "started_at_iso", ""),
            "plan": state.plan,
            "budget_exhausted": state.budget_exhausted,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "args": tc.args,
                    "observation": tc.observation,
                    "status": tc.status,
                    "duration_ms": tc.duration_ms,
                }
                for tc in state.tool_calls
            ],
            "retrieved_chunk_ids": [c.chunk_id for c in state.retrieved_chunks],
            "memory_hits_count": len(state.memory_hits),
            "status": status,
            "error": error,
        }

    async def _save_state(
        self,
        state: AgentState,
        *,
        status: str = "in_progress",
        error: str | None = None,
    ) -> None:
        """Persist state snapshot to StateStore if configured; never raises."""
        if self._state_store is None:
            return
        try:
            key = self._state_key(
                user_id=str(state.user.id),
                conversation_id=state.conversation_id,
                run_id=state.run_id,
            )
            snapshot = self._snapshot(state, status=status, error=error)
            await self._state_store.save(key, snapshot, ttl_seconds=self._state_ttl)
        except Exception:
            logger.warning(
                "AgentService: failed to save state snapshot (run_id=%s)",
                state.run_id,
                exc_info=True,
            )

    async def _delete_state(self, state: AgentState) -> None:
        """Delete ephemeral state on run completion; never raises."""
        if self._state_store is None:
            return
        try:
            key = self._state_key(
                user_id=str(state.user.id),
                conversation_id=state.conversation_id,
                run_id=state.run_id,
            )
            await self._state_store.delete(key)
        except Exception:
            logger.warning(
                "AgentService: failed to delete state snapshot (run_id=%s)",
                state.run_id,
                exc_info=True,
            )

    async def load_state(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Load an ephemeral agent state snapshot from the state store.

        Returns None if no state store is configured or the key is not found.
        """
        if self._state_store is None:
            return None
        try:
            key = self._state_key(
                user_id=user_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )
            return await self._state_store.load(key)
        except Exception:
            logger.warning(
                "AgentService: failed to load state snapshot (run_id=%s)",
                run_id,
                exc_info=True,
            )
            return None
