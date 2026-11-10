"""Unit tests for semantic_math.graph."""

import sys
import unittest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.graph import (  # noqa: E402
    MathEvaluationError,
    MathGraph,
    MathGraphError,
    MathNode,
    NodeType,
)
from semantic_math.schema import AddOperation, MathExpression  # noqa: E402


class TestMathGraph(unittest.TestCase):
    """Tests for C++-style MathGraph construction, validation, and DFS evaluation."""

    def test_single_number_node(self) -> None:
        graph = MathGraph.from_expression(42)
        self.assertEqual(graph.root.node_type, NodeType.NUMBER)
        self.assertEqual(graph.root.value, 42.0)
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.evaluate(), 42.0)

    def test_individual_operations(self) -> None:
        cases = [
            ({"op": "add", "operands": [1, 2, 3, 4.5]}, 10.5),
            ({"op": "multiply", "operands": [2, 3, -4]}, -24.0),
            ({"op": "subtract", "left": 10, "right": 3.5}, 6.5),
            ({"op": "divide", "numerator": 15, "denominator": 4}, 3.75),
            ({"op": "power", "base": 2, "exponent": 5}, 32.0),
            ({"op": "nth_root", "radicand": 81, "n": 4}, 3.0),
        ]
        for payload, expected in cases:
            with self.subTest(op=payload["op"]):
                graph = MathGraph.from_expression(payload)
                self.assertAlmostEqual(graph.evaluate(), expected)

    def test_build_from_pydantic_models_and_parent_links(self) -> None:
        model = AddOperation(op="add", operands=[3, 7, 10])
        graph = MathGraph.from_expression(model)
        self.assertEqual(graph.evaluate(), 20.0)
        self.assertIsNone(graph.root.parent)
        self.assertEqual(len(graph.root.children), 3)
        for child in graph.root.children:
            self.assertIs(child.parent, graph.root)
            self.assertEqual(child.node_type, NodeType.NUMBER)

        wrapped = MathExpression.model_validate({"op": "power", "base": 3, "exponent": 4})
        self.assertEqual(MathGraph.from_expression(wrapped).evaluate(), 81.0)

    def test_nested_multi_layer_expression(self) -> None:
        payload = {
            "op": "add",
            "operands": [
                {
                    "op": "multiply",
                    "operands": [
                        {"op": "subtract", "left": 10, "right": 3},
                        {"op": "divide", "numerator": 12, "denominator": 4},
                        2,
                    ],
                },
                {
                    "op": "power",
                    "base": {"op": "nth_root", "radicand": 27, "n": 3},
                    "exponent": 2,
                },
                5.5,
            ],
        }
        graph = MathGraph.from_expression(payload)
        self.assertEqual(len(graph.nodes), 15)
        self.assertAlmostEqual(graph.evaluate(), 56.5)

    def test_negative_bases_and_odd_roots(self) -> None:
        self.assertAlmostEqual(
            MathGraph.from_expression({"op": "nth_root", "radicand": -8, "n": 3}).evaluate(),
            -2.0,
        )
        self.assertAlmostEqual(
            MathGraph.from_expression({"op": "nth_root", "radicand": -8, "n": -3}).evaluate(),
            -0.5,
        )
        self.assertAlmostEqual(
            MathGraph.from_expression({"op": "nth_root", "radicand": 0, "n": 4}).evaluate(),
            0.0,
        )
        self.assertAlmostEqual(
            MathGraph.from_expression({"op": "power", "base": -2, "exponent": 3}).evaluate(),
            -8.0,
        )
        self.assertAlmostEqual(
            MathGraph.from_expression({"op": "power", "base": -2, "exponent": 2}).evaluate(),
            4.0,
        )

    def test_manual_cpp_style_graph_construction_and_postorder(self) -> None:
        n1 = MathNode(id=1, node_type=NodeType.NUMBER, value=6.0)
        n2 = MathNode(id=2, node_type=NodeType.NUMBER, value=4.0)
        sub = MathNode(id=3, node_type=NodeType.SUBTRACT, children=[n1, n2])
        n3 = MathNode(id=4, node_type=NodeType.NUMBER, value=5.0)
        root = MathNode(id=0, node_type=NodeType.MULTIPLY, children=[sub, n3])

        graph = MathGraph(root)
        self.assertAlmostEqual(graph.evaluate(), 10.0)

        visited_order: list[int] = []

        def record(node: MathNode, children: list[str]) -> str:
            visited_order.append(node.id)
            if node.node_type == NodeType.NUMBER:
                return str(int(node.value or 0))
            return f"({node.node_type.value}: {', '.join(children)})"

        summary = graph.traverse_postorder(record)
        self.assertEqual(visited_order, [1, 2, 3, 4, 0])
        self.assertEqual(summary, "(multiply: (subtract: 6, 4), 5)")

    def test_domain_errors(self) -> None:
        bad_math = [
            {"op": "divide", "numerator": 5, "denominator": 0},
            {"op": "power", "base": 0, "exponent": -2},
            {"op": "power", "base": -4, "exponent": 0.5},
            {"op": "nth_root", "radicand": 16, "n": 0},
            {"op": "nth_root", "radicand": 0, "n": -2},
            {"op": "nth_root", "radicand": -9, "n": 2},
            {"op": "nth_root", "radicand": -8, "n": 2.5},
        ]
        for payload in bad_math:
            with self.subTest(payload=payload):
                graph = MathGraph.from_expression(payload)
                with self.assertRaises(MathEvaluationError):
                    graph.evaluate()

    def test_structural_and_cycle_validation_errors(self) -> None:
        # NUMBER node with child
        bad_num = MathNode(
            id=0,
            node_type=NodeType.NUMBER,
            value=1.0,
            children=[MathNode(id=1, node_type=NodeType.NUMBER, value=2.0)],
        )
        with self.assertRaises(MathGraphError):
            MathGraph(bad_num)

        # ADD node with fewer than 2 children
        bad_add = MathNode(
            id=0,
            node_type=NodeType.ADD,
            children=[MathNode(id=1, node_type=NodeType.NUMBER, value=2.0)],
        )
        with self.assertRaises(MathGraphError):
            MathGraph(bad_add)

        # SUBTRACT node with 3 children
        bad_sub = MathNode(
            id=0,
            node_type=NodeType.SUBTRACT,
            children=[
                MathNode(id=1, node_type=NodeType.NUMBER, value=1.0),
                MathNode(id=2, node_type=NodeType.NUMBER, value=2.0),
                MathNode(id=3, node_type=NodeType.NUMBER, value=3.0),
            ],
        )
        with self.assertRaises(MathGraphError):
            MathGraph(bad_sub)

        # Cycle in graph
        left = MathNode(id=1, node_type=NodeType.NUMBER, value=1.0)
        cyc = MathNode(id=0, node_type=NodeType.ADD, children=[left])
        cyc.children.append(cyc)
        with self.assertRaises(MathGraphError):
            MathGraph(cyc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
