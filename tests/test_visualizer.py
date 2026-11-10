"""Unit tests for semantic_math.visualizer."""

import sys
import unittest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.graph import MathGraph  # noqa: E402
from semantic_math.schema import MathExpression  # noqa: E402
from semantic_math.visualizer import GraphVisualizer  # noqa: E402


class TestGraphVisualizer(unittest.TestCase):
    """Tests for Mermaid flowchart and LaTeX rendering via GraphVisualizer."""

    def setUp(self) -> None:
        self.viz = GraphVisualizer()

    def test_latex_precedence_and_parentheses(self) -> None:
        # (2 + 3) * 4
        expr1 = {
            "op": "multiply",
            "operands": [{"op": "add", "operands": [2, 3]}, 4],
        }
        self.assertEqual(self.viz.to_latex(expr1), r"(2 + 3) \cdot 4")

        # 2 + 3 * 4 (no parens needed around multiplication)
        expr2 = {
            "op": "add",
            "operands": [2, {"op": "multiply", "operands": [3, 4]}],
        }
        self.assertEqual(self.viz.to_latex(expr2), r"2 + 3 \cdot 4")

        # 10 - (4 - 1) vs (10 - 4) - 1
        expr3 = {
            "op": "subtract",
            "left": 10,
            "right": {"op": "subtract", "left": 4, "right": 1},
        }
        self.assertEqual(self.viz.to_latex(expr3), "10 - (4 - 1)")

        expr4 = {
            "op": "subtract",
            "left": {"op": "subtract", "left": 10, "right": 4},
            "right": 1,
        }
        self.assertEqual(self.viz.to_latex(expr4), "10 - 4 - 1")

    def test_latex_fractions_powers_roots_and_negatives(self) -> None:
        self.assertEqual(
            self.viz.to_latex({"op": "divide", "numerator": 3, "denominator": 4}),
            r"\frac{3}{4}",
        )
        self.assertEqual(
            self.viz.to_latex({"op": "nth_root", "radicand": 16, "n": 2}),
            r"\sqrt{16}",
        )
        self.assertEqual(
            self.viz.to_latex({"op": "nth_root", "radicand": -8, "n": 3}),
            r"\sqrt[3]{-8}",
        )
        self.assertEqual(
            self.viz.to_latex({"op": "power", "base": -2, "exponent": 3}),
            "(-2)^{3}",
        )
        self.assertEqual(
            self.viz.to_latex(
                {
                    "op": "power",
                    "base": {"op": "divide", "numerator": 1, "denominator": 2},
                    "exponent": 3,
                }
            ),
            r"(\frac{1}{2})^{3}",
        )
        self.assertEqual(
            self.viz.to_latex(
                {
                    "op": "power",
                    "base": {"op": "power", "base": 2, "exponent": 3},
                    "exponent": 4,
                }
            ),
            "(2^{3})^{4}",
        )
        self.assertEqual(
            self.viz.to_latex({"op": "add", "operands": [2, -3]}),
            "2 + (-3)",
        )

    def test_latex_from_schema_graph_and_equation(self) -> None:
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
        schema_obj = MathExpression.model_validate(payload)
        graph = MathGraph.from_expression(schema_obj)

        expected_tex = r"(10 - 3) \cdot \frac{12}{4} \cdot 2 + (\sqrt[3]{27})^{2} + 5.5"
        self.assertEqual(self.viz.to_latex(schema_obj), expected_tex)
        self.assertEqual(self.viz.to_latex(graph), expected_tex)
        self.assertEqual(self.viz.to_equation_latex(graph), f"{expected_tex} = 56.5")

    def test_mermaid_flowchart_output(self) -> None:
        payload = {
            "op": "power",
            "base": {"op": "nth_root", "radicand": 27, "n": 3},
            "exponent": 2,
        }
        mermaid = self.viz.to_mermaid(payload)
        lines = mermaid.splitlines()
        self.assertEqual(lines[0], "flowchart TD")
        self.assertIn('    n0["power (^)"]', lines)
        self.assertIn("    n0 -->|base| n1", lines)
        self.assertIn("    n0 -->|exponent| n4", lines)
        self.assertIn('    n1["nth_root (sqrt)"]', lines)
        self.assertIn("    n1 -->|radicand| n2", lines)
        self.assertIn("    n1 -->|n| n3", lines)
        self.assertIn('    n2["27"]', lines)
        self.assertIn('    n3["3"]', lines)
        self.assertIn('    n4["2"]', lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
