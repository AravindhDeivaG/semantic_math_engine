"""Graph visualization utilities for Mermaid flowcharts and LaTeX formatting."""

from dataclasses import dataclass
from typing import Any, Callable, Union

from semantic_math.graph import MathGraph, MathNode, NodeType
from semantic_math.schema import MathExpression, Operand

ExpressionInput = Union[MathGraph, MathExpression, Operand, dict[str, Any]]

_PREC_ADD = 10
_PREC_MUL = 20
_PREC_POW = 30
_PREC_ATOM = 40

_MERMAID_OP_LABELS: dict[NodeType, str] = {
    NodeType.ADD: "add (+)",
    NodeType.SUBTRACT: "subtract (-)",
    NodeType.MULTIPLY: "multiply (*)",
    NodeType.DIVIDE: "divide (/)",
    NodeType.POWER: "power (^)",
    NodeType.NTH_ROOT: "nth_root (sqrt)",
}

_EDGE_LABELS: dict[NodeType, tuple[str, str]] = {
    NodeType.SUBTRACT: ("left", "right"),
    NodeType.DIVIDE: ("numerator", "denominator"),
    NodeType.POWER: ("base", "exponent"),
    NodeType.NTH_ROOT: ("radicand", "n"),
}


@dataclass(frozen=True)
class _LatexFrag:
    """Intermediate LaTeX fragment tracked during post-order traversal."""

    tex: str
    prec: int
    is_negative: bool = False
    is_compound_base: bool = False


def _format_number(value: float) -> str:
    """Format a finite float cleanly (no trailing '.0' for integers, no '-0')."""
    if value == 0.0:
        return "0"
    if abs(value) < 1e15 and value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _wrap_parens(tex: str) -> str:
    return f"({tex})"


def _as_graph(source: ExpressionInput) -> MathGraph:
    """Normalize a MathGraph, Pydantic expression, or raw dict into a validated MathGraph."""
    if isinstance(source, MathGraph):
        source.validate()
        return source
    return MathGraph.from_expression(source)


def _latex_number(node: MathNode, _children: list[_LatexFrag]) -> _LatexFrag:
    assert node.value is not None
    val = float(node.value)
    is_neg = val < 0.0
    return _LatexFrag(
        tex=_format_number(val),
        prec=_PREC_ATOM,
        is_negative=is_neg,
        is_compound_base=is_neg,
    )


def _latex_add(_node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    parts = [children[0].tex]
    for child in children[1:]:
        parts.append(_wrap_parens(child.tex) if child.is_negative else child.tex)
    return _LatexFrag(
        tex=" + ".join(parts),
        prec=_PREC_ADD,
        is_compound_base=True,
    )


def _latex_subtract(_node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    left, right = children
    right_tex = (
        _wrap_parens(right.tex)
        if (right.prec <= _PREC_ADD or right.is_negative)
        else right.tex
    )
    return _LatexFrag(
        tex=f"{left.tex} - {right_tex}",
        prec=_PREC_ADD,
        is_compound_base=True,
    )


def _latex_multiply(_node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    parts: list[str] = []
    for idx, child in enumerate(children):
        needs_parens = child.prec < _PREC_MUL or (idx > 0 and child.is_negative)
        parts.append(_wrap_parens(child.tex) if needs_parens else child.tex)
    return _LatexFrag(
        tex=r" \cdot ".join(parts),
        prec=_PREC_MUL,
        is_compound_base=True,
    )


def _latex_divide(_node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    numerator, denominator = children
    return _LatexFrag(
        tex=rf"\frac{{{numerator.tex}}}{{{denominator.tex}}}",
        prec=_PREC_ATOM,
        is_compound_base=True,
    )


def _latex_power(_node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    base, exponent = children
    base_tex = _wrap_parens(base.tex) if base.is_compound_base else base.tex
    return _LatexFrag(
        tex=f"{base_tex}^{{{exponent.tex}}}",
        prec=_PREC_POW,
        is_compound_base=True,
    )


def _latex_nth_root(node: MathNode, children: list[_LatexFrag]) -> _LatexFrag:
    radicand, degree = children
    degree_node = node.children[1]
    if degree_node.node_type == NodeType.NUMBER and degree_node.value == 2.0:
        tex = rf"\sqrt{{{radicand.tex}}}"
    else:
        tex = rf"\sqrt[{degree.tex}]{{{radicand.tex}}}"
    return _LatexFrag(
        tex=tex,
        prec=_PREC_ATOM,
        is_compound_base=True,
    )


_LATEX_BUILDERS: dict[NodeType, Callable[[MathNode, list[_LatexFrag]], _LatexFrag]] = {
    NodeType.NUMBER: _latex_number,
    NodeType.ADD: _latex_add,
    NodeType.SUBTRACT: _latex_subtract,
    NodeType.MULTIPLY: _latex_multiply,
    NodeType.DIVIDE: _latex_divide,
    NodeType.POWER: _latex_power,
    NodeType.NTH_ROOT: _latex_nth_root,
}


class GraphVisualizer:
    """Generates Mermaid flowcharts and LaTeX representations from a MathGraph or schema."""

    def to_latex(self, source: ExpressionInput) -> str:
        """Render the expression as a LaTeX math string via post-order graph traversal."""
        graph = _as_graph(source)
        root_frag = graph.traverse_postorder(
            lambda node, child_frags: _LATEX_BUILDERS[node.node_type](node, child_frags)
        )
        return root_frag.tex

    def to_equation_latex(self, source: ExpressionInput) -> str:
        """Render the expression and its evaluated answer as '<latex> = <answer>'."""
        graph = _as_graph(source)
        expr_tex = self.to_latex(graph)
        result = graph.evaluate()
        return f"{expr_tex} = {_format_number(result)}"

    def to_mermaid(self, source: ExpressionInput) -> str:
        """Render the expression graph as a Mermaid 'flowchart TD' diagram."""
        graph = _as_graph(source)
        node_ids: dict[int, str] = {
            id(node): f"n{idx}" for idx, node in enumerate(graph.nodes)
        }

        def visit(
            node: MathNode, child_items: list[tuple[str, list[str]]]
        ) -> tuple[str, list[str]]:
            node_key = node_ids.setdefault(id(node), f"n{len(node_ids)}")
            if node.node_type == NodeType.NUMBER:
                assert node.value is not None
                label = _format_number(float(node.value))
            else:
                label = _MERMAID_OP_LABELS[node.node_type]

            lines = [f'    {node_key}["{label}"]']
            edge_labels = _EDGE_LABELS.get(node.node_type)
            for idx, (child_key, _child_lines) in enumerate(child_items):
                if edge_labels is not None:
                    lines.append(f"    {node_key} -->|{edge_labels[idx]}| {child_key}")
                else:
                    lines.append(f"    {node_key} --> {child_key}")

            for _child_key, child_lines in child_items:
                lines.extend(child_lines)

            return node_key, lines

        _root_key, body_lines = graph.traverse_postorder(visit)
        return "\n".join(["flowchart TD", *body_lines])
