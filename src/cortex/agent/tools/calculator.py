"""Safe arithmetic calculator tool.

Security
--------
Uses Python's ``ast`` module to parse and evaluate expressions.  Only a
strict allowlist of node types is permitted — no function calls, attribute
access, imports, comprehensions, or any other construct that could lead to
code injection.

``eval()`` is **never called** on user-supplied input.

Resource limits
---------------
* Expressions longer than 500 characters are rejected.
* Exponents larger than 10 000 are rejected before evaluation to prevent
  runaway CPU / memory usage (e.g. ``2 ** 1000000``).

Supported operations
--------------------
* Numeric literals (int, float)
* Unary: ``+``, ``-``
* Binary: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``
* Parentheses for grouping
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import TYPE_CHECKING, Any

from cortex.agent.base import Tool
from cortex.core.exceptions import BadRequestError

if TYPE_CHECKING:
    from cortex.db.models.user import User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AST node → operator mappings (only safe operations)
# ---------------------------------------------------------------------------

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelist of allowed AST node types
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    *_BINARY_OPS.keys(),
    *_UNARY_OPS.keys(),
)

_MAX_EXPRESSION_LEN = 500
# Maximum permitted exponent value for ** to prevent runaway computation.
# e.g. 2 ** 1000001 is rejected; 2 ** 10000 is allowed.
_MAX_EXPONENT = 10_000


def _safe_eval(node: ast.expr | ast.Expression) -> float:
    """Recursively evaluate an AST node against the operator allowlist.

    Raises
    ------
    BadRequestError
        If the expression contains an unsupported node type (e.g. a
        function call, attribute access, or name lookup).
    ZeroDivisionError
        Propagated as-is when the expression divides by zero.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise BadRequestError(
                "Calculator only supports numeric literals",
                details={"unsupported_value": repr(node.value)},
            )
        return float(node.value)

    if isinstance(node, ast.BinOp):
        op_func = _BINARY_OPS.get(type(node.op))
        if op_func is None:
            raise BadRequestError(
                f"Unsupported binary operator: {type(node.op).__name__}",
            )
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        # Guard against runaway exponentiation *before* calling operator.pow.
        # We only restrict positive exponents — negative exponents produce
        # fractions so they cannot cause memory exhaustion.
        if isinstance(node.op, ast.Pow) and right > _MAX_EXPONENT:
            raise BadRequestError(
                f"Exponent too large (max {_MAX_EXPONENT})",
                details={"exponent": right, "max_exponent": _MAX_EXPONENT},
            )
        return float(op_func(left, right))

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise BadRequestError(
                f"Unsupported unary operator: {type(node.op).__name__}",
            )
        return float(op_func(_safe_eval(node.operand)))

    raise BadRequestError(
        f"Unsafe expression node: {type(node).__name__}",
        details={"node_type": type(node).__name__},
    )


def evaluate_expression(expression: str) -> float:
    """Parse and safely evaluate an arithmetic expression string.

    Parameters
    ----------
    expression:
        An arithmetic expression, e.g. ``"(2 + 3) * 4 / 2"``.

    Returns
    -------
    float
        The numeric result.

    Raises
    ------
    BadRequestError
        Expression is too long, contains unsafe nodes, exponent exceeds
        ``_MAX_EXPONENT``, or fails to parse.
    ZeroDivisionError
        The expression divides by zero.
    """
    stripped = expression.strip()
    if not stripped:
        raise BadRequestError(
            "Expression must not be empty",
            details={"field": "expression"},
        )
    if len(stripped) > _MAX_EXPRESSION_LEN:
        raise BadRequestError(
            f"Expression too long (max {_MAX_EXPRESSION_LEN} characters)",
            details={"length": len(stripped)},
        )

    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise BadRequestError(
            "Invalid arithmetic expression",
            details={"reason": str(exc), "expression": stripped[:200]},
        ) from exc

    # Walk all nodes and verify every one is in the allowlist *before*
    # evaluation — prevents any side-channel execution of unsafe nodes.
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise BadRequestError(
                f"Unsafe expression: node type {type(node).__name__!r} is not allowed",
                details={"node_type": type(node).__name__},
            )

    return _safe_eval(tree)


class CalculatorTool(Tool):
    """Evaluate safe arithmetic expressions.

    Supports ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``,
    parentheses, and numeric literals. Never uses ``eval()``.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate an arithmetic expression and return the numeric result."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "An arithmetic expression to evaluate, e.g. '(2 + 3) * 4'"
                    ),
                },
            },
            "required": ["expression"],
        }

    async def execute(self, args: dict[str, Any], user: User) -> str:  # noqa: ARG002
        """Evaluate the expression and return the result as a string.

        The ``user`` argument is accepted for interface compliance but is not
        used — the calculator does not access any user data.
        """
        expression: str = args.get("expression", "")
        logger.info("CalculatorTool: evaluating expression=%r", expression[:200])

        try:
            result = evaluate_expression(expression)
        except BadRequestError as exc:
            return f"Error: {exc.message}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as exc:  # noqa: BLE001
            logger.exception("CalculatorTool unexpected error")
            return f"Error: Could not evaluate expression — {exc}"

        # Format: drop unnecessary trailing zeros for integers
        formatted = str(int(result)) if result == int(result) else f"{result:.10g}"

        logger.info("CalculatorTool: %r = %s", expression[:200], formatted)
        return f"Result: {formatted}"
