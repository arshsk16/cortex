"""Unit tests for the safe CalculatorTool and evaluate_expression function."""

from __future__ import annotations

import pytest

from cortex.agent.tools.calculator import CalculatorTool, evaluate_expression
from cortex.core.exceptions import BadRequestError

# ---------------------------------------------------------------------------
# evaluate_expression — pure arithmetic correctness
# ---------------------------------------------------------------------------


class TestEvaluateExpression:
    """Tests for the standalone evaluate_expression helper."""

    def test_integer_addition(self) -> None:
        assert evaluate_expression("2 + 3") == 5.0

    def test_integer_subtraction(self) -> None:
        assert evaluate_expression("10 - 4") == 6.0

    def test_multiplication(self) -> None:
        assert evaluate_expression("3 * 7") == 21.0

    def test_true_division(self) -> None:
        assert evaluate_expression("10 / 4") == 2.5

    def test_floor_division(self) -> None:
        assert evaluate_expression("10 // 3") == 3.0

    def test_modulo(self) -> None:
        assert evaluate_expression("10 % 3") == 1.0

    def test_exponentiation(self) -> None:
        assert evaluate_expression("2 ** 8") == 256.0

    def test_parentheses_precedence(self) -> None:
        assert evaluate_expression("(2 + 3) * 4") == 20.0

    def test_nested_parentheses(self) -> None:
        assert evaluate_expression("((1 + 2) * (3 + 4)) / 7") == 3.0

    def test_unary_negation(self) -> None:
        assert evaluate_expression("-5 + 10") == 5.0

    def test_unary_positive(self) -> None:
        assert evaluate_expression("+5") == 5.0

    def test_float_literal(self) -> None:
        result = evaluate_expression("3.14 * 2")
        assert abs(result - 6.28) < 1e-9

    def test_complex_expression(self) -> None:
        # (100 - 25) / 5 ** 2 = 75 / 25 = 3
        assert evaluate_expression("(100 - 25) / 5 ** 2") == 3.0

    def test_whitespace_around_expression(self) -> None:
        assert evaluate_expression("  2 + 2  ") == 4.0

    def test_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            evaluate_expression("1 / 0")

    def test_floor_division_by_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            evaluate_expression("5 // 0")


# ---------------------------------------------------------------------------
# Security: forbidden constructs must raise BadRequestError
# ---------------------------------------------------------------------------


class TestEvaluateExpressionSecurity:
    """Verify that no unsafe Python construct can slip through."""

    def test_function_call_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("abs(-5)")

    def test_attribute_access_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("(1).__class__")

    def test_import_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("__import__('os')")

    def test_list_comprehension_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("[x for x in range(10)]")

    def test_lambda_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("(lambda: 42)()")

    def test_string_literal_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("'hello'")

    def test_name_reference_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("os.system('id')")

    def test_empty_expression_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("")

    def test_expression_too_long_rejected(self) -> None:
        long_expr = "1 + " * 200 + "1"
        with pytest.raises(BadRequestError):
            evaluate_expression(long_expr)

    def test_invalid_syntax_rejected(self) -> None:
        with pytest.raises(BadRequestError):
            evaluate_expression("2 * * 3")  # space between * * is a syntax error

    def test_exponent_at_limit_is_allowed(self) -> None:
        """An exponent exactly at _MAX_EXPONENT (10 000) must succeed."""
        # 1 ** 10000 = 1.0, fast to compute regardless of size
        result = evaluate_expression("1 ** 10000")
        assert result == 1.0

    def test_exponent_just_above_limit_rejected(self) -> None:
        """An exponent of MAX_EXPONENT + 1 must be rejected."""
        with pytest.raises(BadRequestError) as exc_info:
            evaluate_expression("2 ** 10001")
        assert "exponent" in exc_info.value.message.lower()

    def test_large_exponent_rejected(self) -> None:
        """2 ** 1000000 must never reach operator.pow — rejected immediately."""
        import time
        start = time.perf_counter()
        with pytest.raises(BadRequestError):
            evaluate_expression("2 ** 1000000")
        elapsed = time.perf_counter() - start
        # Must return in well under 1 s — if it computed the result it would
        # take many seconds and hundreds of MB of RAM.
        assert elapsed < 1.0, f"Exponent check took too long: {elapsed:.3f}s"

    def test_negative_exponent_allowed(self) -> None:
        """Negative exponents produce fractions — no memory risk."""
        # 2 ** -3 = 0.125
        result = evaluate_expression("2 ** -3")
        assert abs(result - 0.125) < 1e-9

    def test_zero_exponent_allowed(self) -> None:
        result = evaluate_expression("999 ** 0")
        assert result == 1.0


# ---------------------------------------------------------------------------
# CalculatorTool.execute — async wrapper
# ---------------------------------------------------------------------------


class TestCalculatorTool:
    """Integration tests for CalculatorTool.execute()."""

    @pytest.fixture
    def tool(self) -> CalculatorTool:
        return CalculatorTool()

    @pytest.fixture
    def mock_user(self, sample_user):  # type: ignore[override]
        return sample_user

    async def test_valid_expression_returns_result(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": "(2 + 3) * 4"}, mock_user)
        assert "20" in observation

    async def test_integer_result_has_no_decimal(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": "6 / 2"}, mock_user)
        assert "3" in observation
        # Should not have unnecessary trailing zeros
        assert "3.0" not in observation

    async def test_float_result_formatted(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": "1 / 3"}, mock_user)
        assert "Result:" in observation

    async def test_division_by_zero_returns_error_string(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": "1 / 0"}, mock_user)
        assert "Error" in observation
        assert "zero" in observation.lower()

    async def test_unsafe_expression_returns_error_string(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": "abs(5)"}, mock_user)
        assert "Error" in observation

    async def test_empty_expression_returns_error_string(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({"expression": ""}, mock_user)
        assert "Error" in observation

    async def test_missing_expression_returns_error_string(
        self, tool: CalculatorTool, mock_user
    ) -> None:
        observation = await tool.execute({}, mock_user)
        assert "Error" in observation

    def test_tool_name(self, tool: CalculatorTool) -> None:
        assert tool.name == "calculator"

    def test_tool_description_non_empty(self, tool: CalculatorTool) -> None:
        assert len(tool.description) > 10

    def test_parameters_schema_has_required_expression(
        self, tool: CalculatorTool
    ) -> None:
        schema = tool.parameters_schema
        assert "expression" in schema["properties"]
        assert "expression" in schema["required"]
