"""Unit tests for ToolRegistry._validate_args and dispatch validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from cortex.agent.base import Tool
from cortex.agent.registry import ToolRegistry
from cortex.core.exceptions import BadRequestError

# ---------------------------------------------------------------------------
# Helpers — minimal Tool stubs for registry tests
# ---------------------------------------------------------------------------


class _StubTool(Tool):
    """Minimal Tool whose schema is configurable at construction time."""

    def __init__(
        self,
        name: str,
        schema: dict[str, Any],
        observation: str = "ok",
    ) -> None:
        self._name = name
        self._schema = schema
        self._observation = observation

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "stub"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._schema

    async def execute(self, args: dict[str, Any], user: Any) -> str:
        return self._observation


def _registry_with(tool: _StubTool) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(tool)
    return reg


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


class TestRegistryRequiredFields:
    async def test_missing_required_field_raises(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(name="search", args={}, user=sample_user)

        err = exc_info.value
        assert "query" in err.message
        assert "search" in err.message

    async def test_all_required_fields_present_passes(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        obs = await reg.dispatch(
            name="search", args={"query": "hello"}, user=sample_user
        )
        assert obs == "ok"

    async def test_multiple_missing_required_fields_reported(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
        }
        tool = _StubTool("multi", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(name="multi", args={}, user=sample_user)

        err = exc_info.value
        assert "a" in err.message or "b" in err.message

    async def test_no_required_key_in_schema_allows_empty_args(
        self, sample_user
    ) -> None:
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            # no "required" key
        }
        tool = _StubTool("list_docs", schema)
        reg = _registry_with(tool)

        obs = await reg.dispatch(name="list_docs", args={}, user=sample_user)
        assert obs == "ok"


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


class TestRegistryTypeValidation:
    async def test_wrong_type_string_field_raises(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(
                name="search", args={"query": 42}, user=sample_user
            )
        assert "string" in exc_info.value.message

    async def test_wrong_type_integer_field_raises(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {"top_k": {"type": "integer"}},
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(
                name="search", args={"top_k": "five"}, user=sample_user
            )
        assert "integer" in exc_info.value.message

    async def test_boolean_rejected_for_integer_field(self, sample_user) -> None:
        """bool is a subclass of int in Python — must be explicitly rejected."""
        schema = {
            "type": "object",
            "properties": {"top_k": {"type": "integer"}},
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(
                name="search", args={"top_k": True}, user=sample_user
            )
        assert "boolean" in exc_info.value.message.lower()

    async def test_extra_unknown_fields_allowed(self, sample_user) -> None:
        """Extra args not in schema properties are silently permitted."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        # 'extra_field' is not in schema but should not raise
        obs = await reg.dispatch(
            name="search",
            args={"query": "hello", "extra_field": "ignored"},
            user=sample_user,
        )
        assert obs == "ok"


# ---------------------------------------------------------------------------
# Numeric bounds validation
# ---------------------------------------------------------------------------


class TestRegistryNumericBounds:
    async def test_value_below_minimum_raises(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(name="search", args={"top_k": 0}, user=sample_user)
        assert ">=" in exc_info.value.message

    async def test_value_above_maximum_raises(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(name="search", args={"top_k": 21}, user=sample_user)
        assert "<=" in exc_info.value.message

    async def test_value_at_minimum_passes(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {"top_k": {"type": "integer", "minimum": 1}},
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        obs = await reg.dispatch(name="search", args={"top_k": 1}, user=sample_user)
        assert obs == "ok"

    async def test_value_at_maximum_passes(self, sample_user) -> None:
        schema = {
            "type": "object",
            "properties": {"top_k": {"type": "integer", "maximum": 20}},
            "required": ["top_k"],
        }
        tool = _StubTool("search", schema)
        reg = _registry_with(tool)

        obs = await reg.dispatch(name="search", args={"top_k": 20}, user=sample_user)
        assert obs == "ok"


# ---------------------------------------------------------------------------
# Non-object schemas — validation skipped
# ---------------------------------------------------------------------------


class TestRegistryNonObjectSchema:
    async def test_non_object_schema_skips_validation(self, sample_user) -> None:
        """If a tool schema is not 'object' type, validation is bypassed."""
        schema = {"type": "string"}  # unusual but valid
        tool = _StubTool("weird", schema)
        reg = _registry_with(tool)

        # Should not raise even with arbitrary args
        obs = await reg.dispatch(name="weird", args={"anything": 123}, user=sample_user)
        assert obs == "ok"


# ---------------------------------------------------------------------------
# Unknown tool raises
# ---------------------------------------------------------------------------


class TestRegistryUnknownTool:
    async def test_unknown_tool_raises_bad_request(self, sample_user) -> None:
        reg = ToolRegistry()  # empty registry

        with pytest.raises(BadRequestError) as exc_info:
            await reg.dispatch(name="ghost_tool", args={}, user=sample_user)

        err = exc_info.value
        assert "ghost_tool" in err.message
        assert err.details["available_tools"] == []


# ---------------------------------------------------------------------------
# Validate_args called before tool.execute
# ---------------------------------------------------------------------------


class TestRegistryValidationOrder:
    async def test_validation_fires_before_tool_execute(self, sample_user) -> None:
        """Tool.execute must NOT be called when schema validation fails."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        execute_mock = AsyncMock(return_value="should not reach")
        tool = _StubTool("search", schema)
        tool.execute = execute_mock  # type: ignore[method-assign]

        reg = _registry_with(tool)

        with pytest.raises(BadRequestError):
            await reg.dispatch(name="search", args={}, user=sample_user)

        execute_mock.assert_not_called()
