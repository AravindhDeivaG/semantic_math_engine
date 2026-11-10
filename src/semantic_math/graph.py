"""C++-style typed AST/graph representation and post-order DFS evaluator."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Optional, TypeVar, Union

from semantic_math.schema import MathExpression, Operand

T = TypeVar("T")


class NodeType(str, Enum):
    """Node kind stored in each graph node."""

    NUMBER = "number"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    NTH_ROOT = "nth_root"


class MathGraphError(ValueError):
    """Raised when a MathGraph has an invalid structure or cycle."""


class MathEvaluationError(MathGraphError):
    """Raised when evaluating a MathGraph hits an invalid mathematical domain."""


@dataclass
class MathNode:
    """Single typed node in the expression graph (C++-style struct with children/parent)."""

    id: int
    node_type: NodeType
    value: Optional[float] = None
    children: list["MathNode"] = field(default_factory=list)
    parent: Optional["MathNode"] = field(default=None, repr=False)


_OP_CHILD_ATTRS: dict[NodeType, tuple[str, ...]] = {
    NodeType.ADD: ("operands",),
    NodeType.MULTIPLY: ("operands",),
    NodeType.SUBTRACT: ("left", "right"),
    NodeType.DIVIDE: ("numerator", "denominator"),
    NodeType.POWER: ("base", "exponent"),
    NodeType.NTH_ROOT: ("radicand", "n"),
}

_BINARY_TYPES = frozenset(
    {NodeType.SUBTRACT, NodeType.DIVIDE, NodeType.POWER, NodeType.NTH_ROOT}
)
_VARIADIC_TYPES = frozenset({NodeType.ADD, NodeType.MULTIPLY})


def _eval_add(values: list[float]) -> float:
    return math.fsum(values)


def _eval_multiply(values: list[float]) -> float:
    return math.prod(values)


def _eval_subtract(values: list[float]) -> float:
    return values[0] - values[1]


def _eval_divide(values: list[float]) -> float:
    numerator, denominator = values
    if denominator == 0.0:
        raise MathEvaluationError("Division by zero.")
    return numerator / denominator


def _eval_power(values: list[float]) -> float:
    base, exponent = values
    if base == 0.0 and exponent < 0.0:
        raise MathEvaluationError("Zero cannot be raised to a negative power.")
    if base < 0.0 and not exponent.is_integer():
        raise MathEvaluationError(
            "Negative base with non-integer exponent results in a non-real value."
        )
    try:
        result = base**exponent
    except OverflowError as exc:
        raise MathEvaluationError("Power operation overflowed.") from exc
    return float(result)


def _eval_nth_root(values: list[float]) -> float:
    radicand, degree = values
    if degree == 0.0:
        raise MathEvaluationError("Root degree 'n' cannot be zero.")
    if radicand == 0.0:
        if degree < 0.0:
            raise MathEvaluationError("Zero cannot have a negative-degree root.")
        return 0.0
    if radicand < 0.0:
        if not degree.is_integer() or int(degree) % 2 == 0:
            raise MathEvaluationError(
                "Even or non-odd-integer root of a negative number is not real."
            )
        try:
            return -((-radicand) ** (1.0 / degree))
        except OverflowError as exc:
            raise MathEvaluationError("Nth root operation overflowed.") from exc
    try:
        return radicand ** (1.0 / degree)
    except OverflowError as exc:
        raise MathEvaluationError("Nth root operation overflowed.") from exc


_OPERATORS: dict[NodeType, Callable[[list[float]], float]] = {
    NodeType.ADD: _eval_add,
    NodeType.MULTIPLY: _eval_multiply,
    NodeType.SUBTRACT: _eval_subtract,
    NodeType.DIVIDE: _eval_divide,
    NodeType.POWER: _eval_power,
    NodeType.NTH_ROOT: _eval_nth_root,
}


class MathGraph:
    """C++-style node-arena graph representing a validated mathematical expression."""

    def __init__(self, root: MathNode, nodes: Optional[list[MathNode]] = None) -> None:
        """Initialize the graph from a root node (and optional arena list) and validate."""
        self.root = root
        self.nodes: list[MathNode] = nodes if nodes is not None else self._collect_nodes(root)
        self.validate()

    @classmethod
    def from_expression(
        cls, expr: Union[MathExpression, Operand, dict[str, Any]]
    ) -> "MathGraph":
        """Build a MathGraph from a Pydantic MathExpression, Operand, or raw dict."""
        if isinstance(expr, MathExpression):
            operand: Operand = expr.root
        else:
            operand = MathExpression.model_validate(expr).root

        nodes: list[MathNode] = []

        def build_node(item: Operand, parent: Optional[MathNode] = None) -> MathNode:
            node_id = len(nodes)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                node = MathNode(
                    id=node_id,
                    node_type=NodeType.NUMBER,
                    value=float(item),
                    parent=parent,
                )
                nodes.append(node)
                return node

            node_type = NodeType(item.op)
            node = MathNode(id=node_id, node_type=node_type, parent=parent)
            nodes.append(node)

            child_attrs = _OP_CHILD_ATTRS[node_type]
            if node_type in _VARIADIC_TYPES:
                raw_children = getattr(item, child_attrs[0])
            else:
                raw_children = [getattr(item, attr) for attr in child_attrs]

            node.children = [build_node(child, parent=node) for child in raw_children]
            return node

        root = build_node(operand)
        return cls(root=root, nodes=nodes)

    @staticmethod
    def _collect_nodes(root: MathNode) -> list[MathNode]:
        """Collect reachable nodes in pre-order while guarding against cycles."""
        collected: list[MathNode] = []
        visited: set[int] = set()
        visiting: set[int] = set()

        def walk(node: MathNode) -> None:
            key = id(node)
            if key in visiting:
                raise MathGraphError("Cycle detected while traversing MathGraph.")
            if key in visited:
                return
            visiting.add(key)
            collected.append(node)
            for child in node.children:
                walk(child)
            visiting.remove(key)
            visited.add(key)

        walk(root)
        return collected

    def validate(self) -> None:
        """Verify structural invariants (cycles, node types, values, and child arities)."""
        visiting: set[int] = set()

        def check_node(node: MathNode) -> None:
            key = id(node)
            if key in visiting:
                raise MathGraphError("Cycle detected in MathGraph.")
            visiting.add(key)

            if not isinstance(node.node_type, NodeType):
                raise MathGraphError(f"Invalid node_type on node {node.id}: {node.node_type!r}")

            if node.node_type == NodeType.NUMBER:
                if (
                    node.value is None
                    or isinstance(node.value, bool)
                    or not isinstance(node.value, (int, float))
                    or not math.isfinite(float(node.value))
                ):
                    raise MathGraphError(
                        f"NUMBER node {node.id} must hold a finite numeric value."
                    )
                if node.children:
                    raise MathGraphError(
                        f"NUMBER node {node.id} must not have child nodes."
                    )
            else:
                if node.value is not None:
                    raise MathGraphError(
                        f"Operation node {node.id} ({node.node_type.value}) cannot hold a literal value."
                    )
                if node.node_type in _VARIADIC_TYPES and len(node.children) < 2:
                    raise MathGraphError(
                        f"{node.node_type.value} node {node.id} requires at least 2 children."
                    )
                if node.node_type in _BINARY_TYPES and len(node.children) != 2:
                    raise MathGraphError(
                        f"{node.node_type.value} node {node.id} requires exactly 2 children."
                    )
                for child in node.children:
                    check_node(child)

            visiting.remove(key)

        check_node(self.root)

    def traverse_postorder(self, visit: Callable[[MathNode, list[T]], T]) -> T:
        """Traverse the graph post-order: descend to operand children, then return up."""
        visiting: set[int] = set()

        def dfs(node: MathNode) -> T:
            key = id(node)
            if key in visiting:
                raise MathGraphError("Cycle detected during graph traversal.")
            visiting.add(key)
            try:
                if node.node_type == NodeType.NUMBER:
                    return visit(node, [])
                child_results = [dfs(child) for child in node.children]
                return visit(node, child_results)
            finally:
                visiting.remove(key)

        return dfs(self.root)

    def evaluate(self) -> float:
        """Solve the expression graph via post-order traversal."""
        self.validate()

        def eval_visit(node: MathNode, child_values: list[float]) -> float:
            if node.node_type == NodeType.NUMBER:
                assert node.value is not None
                return float(node.value)
            operator = _OPERATORS[node.node_type]
            result = float(operator(child_values))
            if not math.isfinite(result):
                raise MathEvaluationError(
                    f"Non-finite result produced at node {node.id} ({node.node_type.value})."
                )
            return result

        return self.traverse_postorder(eval_visit)
