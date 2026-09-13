"""Agent orchestration service — single-agent + tool-calling loop.

Architecture
------------
1. Build the initial reasoning prompt (system + tool schemas + question).
2. Ask the LLM to decide: call a tool, or produce a final answer.
3. If the LLM picks a tool, dispatch it via
   :class:`~cortex.agent.registry.ToolRegistry`,
   append the observation, and loop (up to ``max_tool_calls`` iterations).
4. After the loop, build a grounded final-answer prompt using the
   **existing** :class:`~cortex.services.prompt_builder.PromptBuilder` and
   call :class:`~cortex.llm.base.LLMProvider` — **identical path to RAGService**.
5. Optionally persist the turn to a conversation via
   :class:`~cortex.services.conversation.ConversationService`.

Provider agnosticism
--------------------
``LLMProvider.generate(prompt)`` takes a plain string.  Tool schemas are
embedded in the system prompt; the LLM's tool-call decision arrives as a
JSON string that is parsed with ``json.loads``.  No Gemini SDK specific
tool-calling features are used here.

Conversation security
---------------------
When ``conversation_id`` is supplied, the service calls
:meth:`~cortex.services.conversation.ConversationService.get_conversation`
before any processing to enforce ownership (raises ``ForbiddenError`` /
``NotFoundError`` on violations — same behaviour as the existing chat
endpoint).
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING

from cortex.agent.prompt import AgentPromptBuilder
from cortex.agent.registry import ToolRegistry
from cortex.agent.result import AgentResult, ToolCallRecord
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from cortex.db.models.user import User
    from cortex.llm.base import LLMProvider
    from cortex.services.conversation import ConversationService

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOL_CALLS = 5


class AgentService:
    """Orchestrate a single-agent + tool-calling loop.

    Parameters
    ----------
    llm_provider:
        The application-scoped LLM provider (e.g. GeminiProvider).
    tool_registry:
        Registry pre-populated with the available tools.
    prompt_builder:
        The existing PromptBuilder used for the final grounded-answer step.
    conversation_service:
        Optional — when supplied, the agent enforces conversation ownership
        and persists the user/assistant message pair.
    max_tool_calls:
        Maximum number of tool invocations per agent run.  Prevents infinite
        loops in case the LLM keeps requesting tools.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        prompt_builder: PromptBuilder,
        conversation_service: ConversationService | None = None,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self._llm = llm_provider
        self._registry = tool_registry
        self._prompt_builder = prompt_builder
        self._conv_service = conversation_service
        self._max_tool_calls = max_tool_calls
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
    ) -> AgentResult:
        """Run the agent on a user question and return the grounded answer.

        Parameters
        ----------
        question:
            The user's natural-language question.
        user:
            Authenticated owner — passed through to every tool for ownership
            enforcement.
        conversation_id:
            When set, ownership is verified against this conversation before
            any processing, and the user/assistant messages are persisted on
            completion.

        Returns
        -------
        AgentResult
            Contains the final answer, tool-call trace, and accumulated
            retrieval results for citation construction.
        """
        question = question.strip()
        if not question:
            raise BadRequestError(
                "Agent question must not be empty",
                details={"field": "question"},
            )

        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 0: Conversation ownership check (security gate)
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            await self._conv_service.get_conversation(
                conversation_id=conversation_id,
                user_id=user.id,
            )

        # ------------------------------------------------------------------
        # Step 1: Persist user message
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            await self._conv_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            await self._conv_service.set_auto_title_if_needed(
                conversation_id=conversation_id,
                first_message=question,
            )

        # ------------------------------------------------------------------
        # Step 2: Tool-calling reasoning loop
        # ------------------------------------------------------------------
        tool_calls: list[ToolCallRecord] = []
        all_retrieved_chunks: list[RetrievalResult] = []

        current_prompt = self._agent_prompt_builder.build_initial(
            question=question,
            tool_schemas=self._registry.tool_schemas,
        )

        loop_count = 0
        early_final_answer: str | None = None

        while loop_count < self._max_tool_calls:
            loop_count += 1
            logger.debug(
                "Agent loop iteration=%d user_id=%s conversation_id=%s",
                loop_count,
                user.id,
                conversation_id,
            )

            # Ask the LLM for the next action
            try:
                raw_response = await self._llm.generate(current_prompt)
            except ServiceUnavailableError:
                raise
            except Exception as exc:
                raise ServiceUnavailableError(
                    "LLM generation failed during agent loop",
                    details={"reason": str(exc)},
                ) from exc

            # Parse JSON decision
            decision = self._parse_decision(raw_response)
            action = decision.get("action", "")

            if action == "final_answer":
                early_final_answer = str(decision.get("answer", "")).strip()
                logger.info(
                    "Agent reached final_answer after %d tool calls",
                    len(tool_calls),
                )
                break

            if action == "tool_call":
                tool_name = str(decision.get("tool", "")).strip()
                tool_args: dict = decision.get("args", {})

                # Dispatch the tool
                try:
                    observation = await self._registry.dispatch(
                        name=tool_name,
                        args=tool_args,
                        user=user,
                    )
                except BadRequestError as exc:
                    # Unknown tool or bad args — feed the error back as an
                    # observation so the LLM can self-correct
                    observation = f"Tool error: {exc.message}"
                    tool_name = tool_name or "unknown"

                # Accumulate structured RAG results for citation
                tool_obj = None
                with contextlib.suppress(BadRequestError):
                    tool_obj = self._registry.get_tool(tool_name)
                if isinstance(tool_obj, RAGSearchTool):
                    all_retrieved_chunks.extend(tool_obj.last_results)

                tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=tool_args,
                        observation=observation,
                    )
                )

                # Extend prompt with this turn's decision + observation
                current_prompt = self._agent_prompt_builder.build_observation_turn(
                    previous_prompt=current_prompt,
                    llm_decision=raw_response,
                    tool_name=tool_name,
                    observation=observation,
                )
            else:
                # Unexpected action — treat raw text as the final answer
                logger.warning(
                    "Agent returned unexpected action=%r; treating as final answer",
                    action,
                )
                early_final_answer = raw_response.strip()
                break

        # ------------------------------------------------------------------
        # Step 3: Generate grounded final answer via existing RAG path
        #
        # If the agent already declared a final_answer, we still run
        # PromptBuilder + LLM on the accumulated chunks to produce a
        # properly grounded, cited response — exactly as RAGService does.
        # If no chunks were retrieved, PromptBuilder will include the
        # "no context" note per its existing behaviour.
        # ------------------------------------------------------------------
        grounded_prompt = self._prompt_builder.build(
            question=question,
            retrieved_chunks=all_retrieved_chunks,
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

        # If the agent declared an early final answer but we have no retrieved
        # chunks to ground it, use the agent's declared answer directly to
        # avoid the "I cannot answer" PromptBuilder disclaimer.
        if not all_retrieved_chunks and early_final_answer:
            final_answer = early_final_answer

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Agent run complete: user_id=%s conversation_id=%s "
            "tool_calls=%d chunks=%d total_ms=%.1f answer_len=%d",
            user.id,
            conversation_id,
            len(tool_calls),
            len(all_retrieved_chunks),
            total_ms,
            len(final_answer),
        )

        # ------------------------------------------------------------------
        # Step 4: Persist assistant message + token usage (optional)
        # ------------------------------------------------------------------
        if conversation_id and self._conv_service:
            citation_dicts = [
                {
                    "document_id": c.document_id,
                    "chunk_id": c.chunk_id,
                    "chunk_index": c.chunk_index,
                }
                for c in all_retrieved_chunks
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

        return AgentResult(
            answer=final_answer,
            tool_calls=tool_calls,
            retrieved_chunks=all_retrieved_chunks,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_decision(raw: str) -> dict:
        """Extract a JSON decision dict from the LLM's raw text response.

        Strips markdown code fences if present.  Falls back to an empty
        dict on parse failure so the loop can exit gracefully.
        """
        text = raw.strip()

        # Strip markdown code fences (```json ... ``` or ``` ... ```)
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first (```json or ```) and last (```) lines
            inner_lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
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
            # Return a synthetic final-answer so the loop exits cleanly
            return {"action": "final_answer", "answer": raw.strip()}
