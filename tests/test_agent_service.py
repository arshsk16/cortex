"""Unit tests for AgentService orchestration logic.

All external I/O (LLM, retriever, document service, conversation service)
is mocked so these tests run without any network or database access.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.agent.registry import ToolRegistry
from cortex.agent.result import AgentResult
from cortex.agent.service import AgentService
from cortex.agent.tools.calculator import CalculatorTool
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.core.exceptions import BadRequestError, ServiceUnavailableError
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_retriever():
    """Retriever that returns one fake chunk."""
    retriever = MagicMock()
    chunk = RetrievalResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        text="Paris is the capital of France.",
        score=0.95,
    )
    retriever.retrieve = AsyncMock(return_value=[chunk])
    return retriever


@pytest.fixture
def mock_retriever_empty():
    """Retriever that returns no results."""
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def mock_doc_service():
    """Minimal DocumentService mock."""
    svc = MagicMock()
    svc.list = AsyncMock(
        return_value=MagicMock(
            items=[
                MagicMock(
                    id="doc-1",
                    title="France Report",
                    status="ready",
                    created_at=None,
                )
            ],
            total=1,
        )
    )
    return svc


@pytest.fixture
def mock_conv_service():
    """ConversationService mock that passes ownership checks."""
    svc = MagicMock()
    svc.get_history = AsyncMock(return_value=[])  # Phase 10: history gate
    svc.get_conversation = AsyncMock(return_value=MagicMock(id="conv-1"))
    svc.add_message = AsyncMock()
    svc.set_auto_title_if_needed = AsyncMock()
    svc.record_token_usage = AsyncMock()
    return svc


def _make_registry(retriever, doc_service=None) -> ToolRegistry:
    """Build a ToolRegistry with RAGSearchTool and CalculatorTool."""
    registry = ToolRegistry()
    registry.register(RAGSearchTool(retriever))
    registry.register(CalculatorTool())
    return registry


def _make_service(
    llm_provider,
    registry: ToolRegistry,
    conv_service=None,
    max_tool_calls: int = 5,
) -> AgentService:
    return AgentService(
        llm_provider=llm_provider,
        tool_registry=registry,
        prompt_builder=PromptBuilder(),
        conversation_service=conv_service,
        max_tool_calls=max_tool_calls,
    )


# ---------------------------------------------------------------------------
# Test: direct final answer (no tool calls)
# ---------------------------------------------------------------------------


class TestAgentDirectAnswer:
    async def test_agent_returns_final_answer_directly(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """When LLM replies with final_answer on turn 0, no tools are called."""
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                # Turn 0: immediate final answer
                json.dumps({"action": "final_answer", "answer": "The answer is 42."}),
                # Grounded answer call
                "The answer is 42.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        result = await service.run(question="What is the answer?", user=sample_user)

        assert isinstance(result, AgentResult)
        assert result.tool_calls == []
        assert result.retrieved_chunks == []
        # LLM called twice: once for loop, once for grounded answer
        assert mock_llm_provider.generate.call_count == 2


# ---------------------------------------------------------------------------
# Test: RAG tool call then final answer
# ---------------------------------------------------------------------------


class TestAgentWithRAGTool:
    async def test_agent_calls_rag_search_and_produces_grounded_answer(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """Agent calls rag_search, observes results, then gives grounded answer."""
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                # Turn 0: call rag_search
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "rag_search",
                        "args": {"query": "capital of France"},
                    }
                ),
                # Turn 1: final answer
                json.dumps(
                    {"action": "final_answer", "answer": "Paris is the capital."}
                ),
                # Grounded answer
                "Paris is the capital of France.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        result = await service.run(
            question="What is the capital of France?", user=sample_user
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "rag_search"
        assert len(result.retrieved_chunks) == 1
        assert result.retrieved_chunks[0].document_id == "doc-1"
        mock_retriever.retrieve.assert_called_once()
        # Retriever was called with the authenticated user
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs["user"] is sample_user

    async def test_rag_tool_observation_appended_to_prompt(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """The retrieval observation should appear in the second LLM call's prompt."""
        captured_prompts: list[str] = []

        async def capture_generate(prompt: str) -> str:
            captured_prompts.append(prompt)
            if len(captured_prompts) == 1:
                return json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "rag_search",
                        "args": {"query": "France capital"},
                    }
                )
            if len(captured_prompts) == 2:
                return json.dumps({"action": "final_answer", "answer": "Paris"})
            return "Paris"

        mock_llm_provider.generate = capture_generate
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        await service.run(question="Capital of France?", user=sample_user)

        # Second call prompt must include the retrieval observation
        assert len(captured_prompts) >= 2
        assert "Paris is the capital of France." in captured_prompts[1]


# ---------------------------------------------------------------------------
# Test: Calculator tool call
# ---------------------------------------------------------------------------


class TestAgentWithCalculatorTool:
    async def test_agent_calls_calculator(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "calculator",
                        "args": {"expression": "2 ** 10"},
                    }
                ),
                json.dumps({"action": "final_answer", "answer": "2^10 = 1024"}),
                "2 raised to the power of 10 is 1024.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        result = await service.run(
            question="What is 2 to the power of 10?", user=sample_user
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "calculator"
        assert "1024" in result.tool_calls[0].observation


# ---------------------------------------------------------------------------
# Test: Tool call cap enforcement
# ---------------------------------------------------------------------------


class TestAgentToolCallCap:
    async def test_loop_stops_at_max_tool_calls(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """Agent must stop after max_tool_calls and proceed to grounded answer."""
        # LLM always wants to call rag_search — never reaches final_answer naturally
        always_tool = json.dumps(
            {
                "action": "tool_call",
                "tool": "rag_search",
                "args": {"query": "test"},
            }
        )
        mock_llm_provider.generate = AsyncMock(
            side_effect=[always_tool] * 10 + ["Final grounded answer."]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry, max_tool_calls=3)

        result = await service.run(question="Tell me something.", user=sample_user)

        # Tool calls must be capped at 3
        assert len(result.tool_calls) == 3
        # Retriever called 3 times
        assert mock_retriever.retrieve.call_count == 3


# ---------------------------------------------------------------------------
# Test: Unknown tool — graceful degradation
# ---------------------------------------------------------------------------


class TestAgentUnknownTool:
    async def test_unknown_tool_observation_fed_back_to_llm(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """Unknown tool name returns error observation; agent can self-correct."""
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "action": "tool_call",
                        "tool": "nonexistent_tool",
                        "args": {},
                    }
                ),
                json.dumps(
                    {"action": "final_answer", "answer": "Sorry, I used a wrong tool."}
                ),
                "I cannot determine the answer.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        result = await service.run(question="Do something.", user=sample_user)

        # Should have one tool call record with the error observation
        assert len(result.tool_calls) == 1
        assert "error" in result.tool_calls[0].observation.lower()
        assert result.retrieved_chunks == []


# ---------------------------------------------------------------------------
# Test: Malformed JSON from LLM — graceful fallback
# ---------------------------------------------------------------------------


class TestAgentMalformedLLMResponse:
    async def test_non_json_response_treated_as_final_answer(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """If the LLM returns non-JSON, agent exits loop and uses the text as-is."""
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                "I'll answer directly: The sky is blue.",  # not JSON
                "The sky is blue.",  # grounded answer call
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        result = await service.run(question="What colour is the sky?", user=sample_user)

        # No tools called; loop exited immediately
        assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Test: Empty question rejected
# ---------------------------------------------------------------------------


class TestAgentValidation:
    async def test_empty_question_raises_bad_request(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        with pytest.raises(BadRequestError):
            await service.run(question="   ", user=sample_user)

        mock_llm_provider.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Conversation ownership enforced
# ---------------------------------------------------------------------------


class TestAgentConversationSecurity:
    async def test_ownership_check_called_before_any_processing(
        self, mock_llm_provider, mock_retriever, mock_conv_service, sample_user
    ) -> None:
        """get_history must be called first so ownership errors abort early."""
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps({"action": "final_answer", "answer": "Hello."}),
                "Hello.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(
            mock_llm_provider, registry, conv_service=mock_conv_service
        )

        await service.run(
            question="Hello",
            user=sample_user,
            conversation_id="conv-1",
        )

        # Phase 10: ownership is enforced via get_history (which calls get_conversation
        # internally in the real ConversationService)
        mock_conv_service.get_history.assert_called_once_with(
            conversation_id="conv-1",
            user_id=sample_user.id,
            limit=10,
        )

    async def test_forbidden_conversation_blocks_processing(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        """A ForbiddenError from get_history must propagate immediately."""
        from cortex.core.exceptions import ForbiddenError

        conv_service = MagicMock()
        conv_service.get_history = AsyncMock(
            side_effect=ForbiddenError("You do not own this conversation")
        )

        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry, conv_service=conv_service)

        with pytest.raises(ForbiddenError):
            await service.run(
                question="Hello",
                user=sample_user,
                conversation_id="someone-elses-conv",
            )

        # LLM must never be called if ownership check fails
        mock_llm_provider.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Test: LLM failure propagates
# ---------------------------------------------------------------------------


class TestAgentLLMFailure:
    async def test_llm_unavailable_raises_service_error(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        mock_llm_provider.generate = AsyncMock(
            side_effect=ServiceUnavailableError("LLM down")
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(mock_llm_provider, registry)

        with pytest.raises(ServiceUnavailableError):
            await service.run(question="Will this fail?", user=sample_user)


# ---------------------------------------------------------------------------
# Test: Conversation persistence
# ---------------------------------------------------------------------------


class TestAgentConversationPersistence:
    async def test_messages_persisted_when_conversation_id_given(
        self, mock_llm_provider, mock_retriever, mock_conv_service, sample_user
    ) -> None:
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps({"action": "final_answer", "answer": "Persisted answer."}),
                "Persisted answer.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(
            mock_llm_provider, registry, conv_service=mock_conv_service
        )

        await service.run(
            question="Persist this please",
            user=sample_user,
            conversation_id="conv-1",
        )

        # User message persisted
        calls = [c.kwargs for c in mock_conv_service.add_message.call_args_list]
        roles = [c["role"] for c in calls]
        assert "user" in roles
        assert "assistant" in roles

        # Token usage recorded
        mock_conv_service.record_token_usage.assert_called_once()

    async def test_no_persistence_without_conversation_id(
        self, mock_llm_provider, mock_retriever, mock_conv_service, sample_user
    ) -> None:
        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps({"action": "final_answer", "answer": "Ephemeral answer."}),
                "Ephemeral answer.",
            ]
        )
        registry = _make_registry(mock_retriever)
        service = _make_service(
            mock_llm_provider, registry, conv_service=mock_conv_service
        )

        # No conversation_id supplied
        await service.run(question="No history please", user=sample_user)

        mock_conv_service.add_message.assert_not_called()
        mock_conv_service.record_token_usage.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Memory extraction background task scheduling
# ---------------------------------------------------------------------------


class TestAgentMemoryExtractionBackgroundTasks:
    async def test_run_schedules_memory_extraction_task(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        from fastapi import BackgroundTasks

        mock_llm_provider.generate = AsyncMock(
            side_effect=[
                json.dumps({"action": "final_answer", "answer": "I remember you."}),
                "I remember you.",
            ]
        )
        registry = _make_registry(mock_retriever)
        mock_extractor = MagicMock()
        service = AgentService(
            llm_provider=mock_llm_provider,
            tool_registry=registry,
            prompt_builder=PromptBuilder(),
            memory_extractor=mock_extractor,
        )

        bg_tasks = MagicMock(spec=BackgroundTasks)
        result = await service.run(
            question="My name is Bob.",
            user=sample_user,
            background_tasks=bg_tasks,
        )

        assert result.answer == "I remember you."
        bg_tasks.add_task.assert_called_once_with(
            mock_extractor.extract_and_store,
            user=sample_user,
            question="My name is Bob.",
            answer="I remember you.",
            memory_hits=[],
        )

    async def test_stream_schedules_memory_extraction_task(
        self, mock_llm_provider, mock_retriever, sample_user
    ) -> None:
        from fastapi import BackgroundTasks

        async def _gen():
            yield "I "
            yield "remember."

        mock_llm_provider.generate = AsyncMock(
            return_value=json.dumps(
                {"action": "final_answer", "answer": "I remember."}
            )
        )
        mock_llm_provider.generate_stream = AsyncMock(return_value=_gen())
        registry = _make_registry(mock_retriever)
        mock_extractor = MagicMock()
        service = AgentService(
            llm_provider=mock_llm_provider,
            tool_registry=registry,
            prompt_builder=PromptBuilder(),
            memory_extractor=mock_extractor,
        )

        bg_tasks = MagicMock(spec=BackgroundTasks)
        events = []
        async for ev in service.stream(
            question="My name is Bob.",
            user=sample_user,
            background_tasks=bg_tasks,
        ):
            events.append(ev)

        bg_tasks.add_task.assert_called_once_with(
            mock_extractor.extract_and_store,
            user=sample_user,
            question="My name is Bob.",
            answer="I remember.",
            memory_hits=[],
        )
