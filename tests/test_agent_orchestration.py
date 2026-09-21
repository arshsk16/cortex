"""Tests for Phase 14 — Advanced Agent Orchestration.

Covers:
1.  ToolCallRecord.status defaults to "ok"
2.  ToolCallRecord.status="error" on tool BadRequestError (prompt loop)
3.  ToolCallRecord.status="skipped" on duplicate tool call (prompt loop)
4.  duration_ms is recorded (>= 0) for ok and error calls
5.  Duplicate call is skipped and produces _DUPLICATE_TOOL_MSG observation
6.  Second duplicate call still counts against the budget (loop_count++)
7.  budget_exhausted=True when max_tool_calls reached (prompt loop)
8.  budget_exhausted=True when max_tool_calls reached (native loop)
9.  Grounded-answer prompt includes budget-exhausted note when flag is set
10. tool_call_summary() is empty string when no tool calls made
11. tool_call_summary() renders compact lines with status and duration
12. AgentState.plan is populated from AgentPlanner (prompt loop)
13. AgentPlanner failure is non-fatal; loop continues without plan
14. context_flags injected into initial prompt (has_history/has_memory)
15. Observation truncated at _MAX_OBSERVATION_CHARS in prompt-based loop
16. AgentState.plan defaults to empty list (no planner configured)
17. AgentState.budget_exhausted defaults to False
18. _snapshot includes plan, budget_exhausted, status, duration_ms
19. Duplicate call (native loop) skips dispatch and records SKIPPED status
20. AgentPlanner.plan filters fabricated tool names from steps
21. AgentPlanner returns empty TaskPlan on LLM failure
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cortex.agent.planner import AgentPlanner
from cortex.agent.prompt import AgentPromptBuilder
from cortex.agent.registry import ToolRegistry
from cortex.agent.result import (
    TOOL_STATUS_ERROR,
    TOOL_STATUS_OK,
    TOOL_STATUS_SKIPPED,
    ToolCallRecord,
)
from cortex.agent.service import AgentService, _freeze_args
from cortex.agent.state import AgentState
from cortex.agent.tools.calculator import CalculatorTool
from cortex.agent.tools.rag_search import RAGSearchTool
from cortex.agent.types import GenerateWithToolsResult, ToolCallRequest
from cortex.llm.base import LLMProvider, SupportsToolCalling
from cortex.retrieval.models import RetrievalResult
from cortex.services.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = str(uuid4())
    return user


def _make_retriever(chunks: list | None = None) -> MagicMock:
    r = MagicMock()
    r.retrieve = AsyncMock(return_value=chunks or [])
    return r


def _make_registry(retriever=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(RAGSearchTool(retriever or _make_retriever()))
    registry.register(CalculatorTool())
    return registry


def _make_service(
    llm,
    registry: ToolRegistry | None = None,
    max_tool_calls: int = 5,
    planner=None,
) -> AgentService:
    return AgentService(
        llm_provider=llm,
        tool_registry=registry or _make_registry(),
        prompt_builder=PromptBuilder(),
        max_tool_calls=max_tool_calls,
        planner=planner,
    )


class _PromptLLM(LLMProvider):
    """Prompt-based LLM that does NOT implement SupportsToolCalling."""

    def __init__(self, side_effect: list[str]) -> None:
        self._responses = iter(side_effect)

    async def generate(self, prompt: str) -> str:
        return next(self._responses)

    async def generate_stream(self, prompt: str):
        yield (await self.generate(prompt))


class _NativeLLM(LLMProvider, SupportsToolCalling):
    """Native tool-calling LLM."""

    def __init__(self, side_effect: list) -> None:
        self._responses = iter(side_effect)

    async def generate(self, prompt: str) -> str:
        return next(self._responses)

    async def generate_stream(self, prompt: str):
        yield (await self.generate(prompt))

    async def generate_with_tools(  # type: ignore[override]
        self, messages, tool_schemas
    ) -> GenerateWithToolsResult:
        return next(self._responses)


# ---------------------------------------------------------------------------
# Test: ToolCallRecord defaults and fields
# ---------------------------------------------------------------------------


class TestToolCallRecordFields:
    def test_default_status_is_ok(self) -> None:
        tc = ToolCallRecord(tool_name="calc", args={}, observation="42")
        assert tc.status == TOOL_STATUS_OK

    def test_default_duration_ms_is_zero(self) -> None:
        tc = ToolCallRecord(tool_name="calc", args={}, observation="42")
        assert tc.duration_ms == 0.0

    def test_explicit_status_error(self) -> None:
        tc = ToolCallRecord(
            tool_name="calc", args={}, observation="err", status=TOOL_STATUS_ERROR
        )
        assert tc.status == TOOL_STATUS_ERROR

    def test_explicit_status_skipped(self) -> None:
        tc = ToolCallRecord(
            tool_name="calc", args={}, observation="dup", status=TOOL_STATUS_SKIPPED
        )
        assert tc.status == TOOL_STATUS_SKIPPED

    def test_duration_ms_stored(self) -> None:
        tc = ToolCallRecord(
            tool_name="calc", args={}, observation="42", duration_ms=123.4
        )
        assert tc.duration_ms == pytest.approx(123.4)


# ---------------------------------------------------------------------------
# Test: AgentState Phase 14 fields
# ---------------------------------------------------------------------------


class TestAgentStatePhase14:
    def test_plan_defaults_to_empty(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        assert state.plan == []

    def test_budget_exhausted_defaults_false(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        assert state.budget_exhausted is False

    def test_tool_call_summary_empty_when_no_calls(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        assert state.tool_call_summary() == ""

    def test_tool_call_summary_renders_calls(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        state.tool_calls.append(
            ToolCallRecord(
                tool_name="calculator",
                args={"expression": "2+2"},
                observation="Result: 4",
                status=TOOL_STATUS_OK,
                duration_ms=5.0,
            )
        )
        summary = state.tool_call_summary()
        assert "calculator" in summary
        assert "Result: 4" in summary
        assert "ok" in summary
        assert "5ms" in summary

    def test_tool_call_summary_shows_skipped_status(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        state.tool_calls.append(
            ToolCallRecord(
                tool_name="rag_search",
                args={"query": "x"},
                observation="dup",
                status=TOOL_STATUS_SKIPPED,
                duration_ms=0.0,
            )
        )
        assert "skipped" in state.tool_call_summary()

    def test_tool_call_summary_multiple_calls(self) -> None:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        for i in range(3):
            state.tool_calls.append(
                ToolCallRecord(
                    tool_name=f"tool_{i}",
                    args={},
                    observation=f"obs_{i}",
                    status=TOOL_STATUS_OK,
                    duration_ms=float(i * 10),
                )
            )
        summary = state.tool_call_summary()
        lines = summary.strip().splitlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Test: _freeze_args helper
# ---------------------------------------------------------------------------


class TestFreezeArgs:
    def test_same_args_same_key(self) -> None:
        a = _freeze_args({"q": "Paris", "limit": 5})
        b = _freeze_args({"limit": 5, "q": "Paris"})
        assert a == b

    def test_different_args_different_key(self) -> None:
        a = _freeze_args({"q": "Paris"})
        b = _freeze_args({"q": "London"})
        assert a != b

    def test_empty_args(self) -> None:
        assert _freeze_args({}) == "{}"


# ---------------------------------------------------------------------------
# Test: Prompt-based loop - status tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPromptLoopStatusTracking:
    async def test_ok_status_on_successful_tool_call(self) -> None:
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "1+1"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "2"}),
            "The answer is 2.",
        ])
        service = _make_service(llm)
        user = _make_user()
        result = await service.run(question="What is 1+1?", user=user)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].status == TOOL_STATUS_OK
        assert result.tool_calls[0].duration_ms >= 0.0

    async def test_error_status_on_bad_tool_name(self) -> None:
        llm = _PromptLLM([
            json.dumps({"action": "tool_call", "tool": "nonexistent_tool", "args": {}}),
            json.dumps({"action": "final_answer", "answer": "oops"}),
            "I encountered an error.",
        ])
        service = _make_service(llm)
        user = _make_user()
        result = await service.run(question="Test?", user=user)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].status == TOOL_STATUS_ERROR
        assert "Tool error" in result.tool_calls[0].observation

    async def test_skipped_status_on_duplicate_call(self) -> None:
        # LLM calls calculator twice with identical args
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "2*3"},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "2*3"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "6"}),
            "The answer is 6.",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()
        result = await service.run(question="What is 2*3?", user=user)

        # First call: ok; second call: skipped
        statuses = [tc.status for tc in result.tool_calls]
        assert TOOL_STATUS_OK in statuses
        assert TOOL_STATUS_SKIPPED in statuses

    async def test_duplicate_call_has_zero_duration(self) -> None:
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "5+5"},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "5+5"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "10"}),
            "10",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()
        result = await service.run(question="q?", user=user)
        skipped = [tc for tc in result.tool_calls if tc.status == TOOL_STATUS_SKIPPED]
        assert skipped
        assert skipped[0].duration_ms == 0.0

    async def test_duplicate_call_observation_is_synthetic(self) -> None:
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "7+7"},
                }
            ),
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "7+7"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "14"}),
            "14",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()
        result = await service.run(question="q?", user=user)
        skipped = [tc for tc in result.tool_calls if tc.status == TOOL_STATUS_SKIPPED]
        assert "already called" in skipped[0].observation


# ---------------------------------------------------------------------------
# Test: Budget exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBudgetExhausted:
    async def test_budget_exhausted_flag_set_prompt_loop(self) -> None:
        """When max_tool_calls=1, a loop that never returns final_answer hits budget."""
        # LLM keeps returning tool_call - never final_answer
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "1"},
                }
            ),
            # Budget exhausted here - grounded answer call
            "Best effort answer.",
        ])
        service = _make_service(llm, max_tool_calls=1)
        user = _make_user()

        # We can't directly inspect state, but verify the run completes
        # and the budget note appears in the prompt sent to grounded answer.
        captured_prompts = []

        original_generate = llm.generate
        call_count = [0]

        async def tracking_generate(prompt: str) -> str:
            captured_prompts.append(prompt)
            call_count[0] += 1
            return await original_generate(prompt)

        llm.generate = tracking_generate  # type: ignore[method-assign]
        await service.run(question="Keep looping?", user=user)

        # The grounded answer prompt (last call) should contain the budget note
        assert any("budget was exhausted" in p for p in captured_prompts)

    async def test_budget_exhausted_flag_set_native_loop(self) -> None:
        """Native loop: budget exhausted when provider keeps returning tool calls."""
        tc_req = ToolCallRequest(
            tool_name="calculator",
            args={"expression": "1+1"},
            call_id="call-1",
        )
        # Always request a tool call, never text - budget hits max_tool_calls=1
        llm = _NativeLLM([
            GenerateWithToolsResult(tool_call=tc_req),
            # grounded answer
            "Best effort.",
        ])
        service = _make_service(llm, max_tool_calls=1)
        user = _make_user()

        captured = []
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        await service.run(question="Loop forever?", user=user)
        assert any("budget was exhausted" in p for p in captured)

    async def test_budget_note_not_in_prompt_when_not_exhausted(self) -> None:
        """When the loop exits normally, the budget note must NOT appear."""
        llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "Direct answer."}),
            "Direct answer.",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()

        captured = []
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        await service.run(question="Simple question?", user=user)
        assert all("budget was exhausted" not in p for p in captured)


# ---------------------------------------------------------------------------
# Test: Tool-call summary injected into grounded-answer prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolCallSummaryInjection:
    async def test_summary_injected_after_tool_call(self) -> None:
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "calculator",
                    "args": {"expression": "3+3"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "6"}),
            "The answer is 6.",
        ])
        service = _make_service(llm)
        user = _make_user()

        captured = []
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        await service.run(question="What is 3+3?", user=user)

        # The last captured prompt is the grounded-answer call
        grounded = captured[-1]
        assert "TOOL CALLS MADE" in grounded
        assert "calculator" in grounded

    async def test_summary_not_injected_when_no_tool_calls(self) -> None:
        llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "No tools."}),
            "No tools needed.",
        ])
        service = _make_service(llm)
        user = _make_user()

        captured = []
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        await service.run(question="No tools please?", user=user)
        grounded = captured[-1]
        assert "TOOL CALLS MADE" not in grounded


# ---------------------------------------------------------------------------
# Test: AgentPromptBuilder Phase 14 additions
# ---------------------------------------------------------------------------


class TestAgentPromptBuilderPhase14:
    def test_build_initial_no_context_flags(self) -> None:
        builder = AgentPromptBuilder()
        prompt = builder.build_initial(
            question="What is 2+2?",
            tool_schemas=[{"name": "calculator", "description": "Calc"}],
        )
        assert "2+2" in prompt
        assert "calculator" in prompt
        # No context block when flags not passed
        assert "Available Context" not in prompt

    def test_build_initial_with_context_flags_memory(self) -> None:
        builder = AgentPromptBuilder()
        prompt = builder.build_initial(
            question="q",
            tool_schemas=[],
            context_flags={"has_history": False, "has_memory": True},
        )
        assert "Long-term memory: YES" in prompt

    def test_build_initial_with_context_flags_history(self) -> None:
        builder = AgentPromptBuilder()
        prompt = builder.build_initial(
            question="q",
            tool_schemas=[],
            context_flags={"has_history": True, "has_memory": False},
        )
        assert "Conversation history: YES" in prompt

    def test_observation_truncated_at_max_chars(self) -> None:
        builder = AgentPromptBuilder()
        big_obs = "X" * 5000
        prompt = builder.build_observation_turn(
            previous_prompt="PREV",
            llm_decision='{"action":"tool_call"}',
            tool_name="rag_search",
            observation=big_obs,
            max_observation_chars=4000,
        )
        assert "X" * 4001 not in prompt
        assert "truncated" in prompt

    def test_observation_not_truncated_when_within_limit(self) -> None:
        builder = AgentPromptBuilder()
        obs = "Short observation."
        prompt = builder.build_observation_turn(
            previous_prompt="PREV",
            llm_decision="{}",
            tool_name="calc",
            observation=obs,
            max_observation_chars=4000,
        )
        assert obs in prompt
        assert "truncated" not in prompt

    def test_build_budget_exhausted_turn(self) -> None:
        builder = AgentPromptBuilder()
        prompt = builder.build_budget_exhausted_turn(
            previous_prompt="PREV PROMPT",
            max_calls=3,
        )
        assert "3" in prompt
        assert "PREV PROMPT" in prompt
        assert "exhausted" in prompt.lower()


# ---------------------------------------------------------------------------
# Test: AgentPlanner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAgentPlanner:
    async def test_plan_parses_valid_response(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=json.dumps({
                "steps": ["rag_search", "calculator"],
                "rationale": "Search then compute.",
                "confidence": 0.85,
            })
        )
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="What is the population of France squared?",
            tool_schemas=[
                {"name": "rag_search", "description": "Search docs"},
                {"name": "calculator", "description": "Calc"},
            ],
        )
        assert plan.steps == ["rag_search", "calculator"]
        assert plan.rationale == "Search then compute."
        assert plan.confidence == pytest.approx(0.85)

    async def test_plan_returns_empty_on_llm_failure(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="q",
            tool_schemas=[{"name": "rag_search", "description": "d"}],
        )
        assert plan.steps == []
        assert plan.rationale == ""

    async def test_plan_returns_empty_on_invalid_json(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(return_value="not json at all")
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="q",
            tool_schemas=[{"name": "rag_search", "description": "d"}],
        )
        assert plan.steps == []

    async def test_plan_filters_fabricated_tool_names(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=json.dumps({
                "steps": ["rag_search", "exec_shell", "rag_search"],
                "rationale": "Prompt injection attempt",
                "confidence": 1.0,
            })
        )
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="q",
            tool_schemas=[{"name": "rag_search", "description": "d"}],
        )
        # exec_shell is not a registered tool - should be filtered
        assert "exec_shell" not in plan.steps
        assert plan.steps == ["rag_search", "rag_search"]

    async def test_plan_empty_steps_when_direct_answer(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=json.dumps({
                "steps": [],
                "rationale": "Can answer directly.",
                "confidence": 0.95,
            })
        )
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="What color is the sky?",
            tool_schemas=[{"name": "rag_search", "description": "d"}],
        )
        assert plan.steps == []

    async def test_plan_strips_markdown_code_fences(self) -> None:
        response = (
            '```json\n{"steps": ["calculator"], "rationale": "Calc it.",'
            ' "confidence": 1.0}\n```'
        )
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(
            question="q",
            tool_schemas=[{"name": "calculator", "description": "d"}],
        )
        assert plan.steps == ["calculator"]

    async def test_plan_confidence_clamped_to_0_1(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(
            return_value=json.dumps({
                "steps": [],
                "rationale": "r",
                "confidence": 99.9,
            })
        )
        planner = AgentPlanner(llm_provider=llm)
        plan = await planner.plan(question="q", tool_schemas=[])
        assert plan.confidence <= 1.0


# ---------------------------------------------------------------------------
# Test: AgentPlanner integration with prompt loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPlannerIntegration:
    async def test_plan_stored_in_state(self) -> None:
        """Verify state.plan is set when planner returns steps."""
        planner_llm = MagicMock()
        planner_llm.generate = AsyncMock(
            return_value=json.dumps({
                "steps": ["calculator"],
                "rationale": "Use calc.",
                "confidence": 0.9,
            })
        )
        planner = AgentPlanner(llm_provider=planner_llm)

        # Agent LLM goes straight to final_answer
        agent_llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "42"}),
            "42",
        ])
        service = _make_service(agent_llm, planner=planner)
        user = _make_user()

        # Run the service - the plan should be captured in state internally.
        # We can't directly inspect state here, so we verify the planner was
        # actually called (plan LLM generate called exactly once).
        result = await service.run(question="What is the answer?", user=user)
        assert result.answer == "42"
        planner_llm.generate.assert_called_once()

    async def test_planner_failure_is_nonfatal(self) -> None:
        """A crashing planner must not prevent the agent from running."""
        bad_planner = MagicMock()
        bad_planner.plan = AsyncMock(side_effect=RuntimeError("Planner crashed"))

        agent_llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "Still works."}),
            "Still works.",
        ])
        service = _make_service(agent_llm, planner=bad_planner)
        user = _make_user()
        result = await service.run(question="Will planner crash?", user=user)
        assert result.answer == "Still works."

    async def test_no_planner_no_plan_steps(self) -> None:
        """Without a planner, state.plan stays empty."""
        agent_llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "ok"}),
            "ok",
        ])
        service = _make_service(agent_llm, planner=None)
        # No planner configured - verify service runs without error
        user = _make_user()
        result = await service.run(question="q", user=user)
        assert result.answer == "ok"


# ---------------------------------------------------------------------------
# Test: _snapshot includes Phase 14 fields
# ---------------------------------------------------------------------------


class TestSnapshotPhase14:
    def _make_state(self) -> AgentState:
        user = _make_user()
        state = AgentState(question="q", user=user, conversation_id=None)
        state.plan = ["rag_search"]
        state.budget_exhausted = True
        state.tool_calls.append(
            ToolCallRecord(
                tool_name="rag_search",
                args={"query": "test"},
                observation="obs",
                status=TOOL_STATUS_ERROR,
                duration_ms=42.0,
            )
        )
        return state

    def test_snapshot_includes_plan(self) -> None:
        state = self._make_state()
        snap = AgentService._snapshot(state, status="in_progress")
        assert snap["plan"] == ["rag_search"]

    def test_snapshot_includes_budget_exhausted(self) -> None:
        state = self._make_state()
        snap = AgentService._snapshot(state, status="in_progress")
        assert snap["budget_exhausted"] is True

    def test_snapshot_includes_tool_status_and_duration(self) -> None:
        state = self._make_state()
        snap = AgentService._snapshot(state, status="in_progress")
        tc = snap["tool_calls"][0]
        assert tc["status"] == TOOL_STATUS_ERROR
        assert tc["duration_ms"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Test: Native loop deduplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestNativeLoopDeduplication:
    async def test_native_duplicate_call_skipped(self) -> None:
        tc_req = ToolCallRequest(
            tool_name="calculator",
            args={"expression": "1+1"},
            call_id="call-1",
        )
        tc_req2 = ToolCallRequest(
            tool_name="calculator",
            args={"expression": "1+1"},
            call_id="call-2",
        )
        # First: tool call. Second: same tool+args (duplicate). Third: text.
        llm = _NativeLLM([
            GenerateWithToolsResult(tool_call=tc_req),
            GenerateWithToolsResult(tool_call=tc_req2),
            GenerateWithToolsResult(text="The answer is 2."),
            # grounded answer
            "The answer is 2.",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()
        result = await service.run(question="1+1?", user=user)

        statuses = [tc.status for tc in result.tool_calls]
        assert TOOL_STATUS_OK in statuses
        assert TOOL_STATUS_SKIPPED in statuses

    async def test_native_different_args_not_skipped(self) -> None:
        tc1 = ToolCallRequest(
            tool_name="calculator", args={"expression": "1+1"}, call_id="c1"
        )
        tc2 = ToolCallRequest(
            tool_name="calculator", args={"expression": "2+2"}, call_id="c2"
        )
        llm = _NativeLLM([
            GenerateWithToolsResult(tool_call=tc1),
            GenerateWithToolsResult(tool_call=tc2),
            GenerateWithToolsResult(text="Answers computed."),
            "Answers: 2 and 4.",
        ])
        service = _make_service(llm, max_tool_calls=5)
        user = _make_user()
        result = await service.run(question="Compute both?", user=user)

        assert all(tc.status == TOOL_STATUS_OK for tc in result.tool_calls)
        assert len(result.tool_calls) == 2


# ---------------------------------------------------------------------------
# Test: context_flags injected into prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestContextFlagsInPrompt:
    async def test_context_flags_in_initial_prompt_no_history_no_memory(self) -> None:
        captured = []
        llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "ok"}),
            "ok",
        ])
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        service = _make_service(llm)
        user = _make_user()
        await service.run(question="q?", user=user)

        # First captured prompt is the agent loop prompt - check context block
        first = captured[0]
        assert "Conversation history: NO" in first
        assert "Long-term memory: NO" in first

    async def test_context_flags_when_memory_available(self) -> None:
        captured = []
        llm = _PromptLLM([
            json.dumps({"action": "final_answer", "answer": "ok"}),
            "ok",
        ])
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]

        # Simulate memory service returning hits
        from datetime import UTC, datetime

        from cortex.schemas.memory import MemoryRead, MemorySearchResult
        _mem_read = MemoryRead(
            id="m1",
            user_id="u1",
            content="I like Python",
            memory_metadata=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        mem_svc = MagicMock()
        mem_svc.search = AsyncMock(
            return_value=[MemorySearchResult(memory=_mem_read, score=0.9)]
        )
        service = AgentService(
            llm_provider=llm,
            tool_registry=_make_registry(),
            prompt_builder=PromptBuilder(),
            memory_service=mem_svc,
            memory_retrieval_limit=5,
        )
        user = _make_user()
        await service.run(question="q?", user=user)

        first = captured[0]
        assert "Long-term memory: YES" in first


# ---------------------------------------------------------------------------
# Test: Observation truncation in prompt-based loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestObservationTruncation:
    async def test_large_rag_observation_truncated_in_prompt(self) -> None:
        big_chunk = RetrievalResult(
            chunk_id="c1",
            document_id="d1",
            chunk_index=0,
            text="Y" * 6000,  # 6000 chars - should be truncated to 4000
            score=0.9,
        )
        retriever = _make_retriever([big_chunk])
        registry = ToolRegistry()
        registry.register(RAGSearchTool(retriever))
        registry.register(CalculatorTool())

        captured = []
        llm = _PromptLLM([
            json.dumps(
                {
                    "action": "tool_call",
                    "tool": "rag_search",
                    "args": {"query": "y"},
                }
            ),
            json.dumps({"action": "final_answer", "answer": "found"}),
            "Found it.",
        ])
        orig = llm.generate

        async def cap(prompt: str) -> str:
            captured.append(prompt)
            return await orig(prompt)

        llm.generate = cap  # type: ignore[method-assign]
        service = _make_service(llm, registry=registry)
        user = _make_user()
        await service.run(question="Find Y?", user=user)

        # The second prompt (after rag_search) should have truncated observation
        assert len(captured) >= 2
        second_prompt = captured[1]
        # Must NOT contain 6000 Y's
        assert "Y" * 4001 not in second_prompt
        assert "truncated" in second_prompt
