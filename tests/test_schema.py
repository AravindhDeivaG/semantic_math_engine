"""Unit tests and schema printer for semantic_math.schema."""

import json
import math
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.schema import (  # noqa: E402
    AddOperation,
    DivideOperation,
    MathExpression,
    MultiplyOperation,
    NthRootOperation,
    PowerOperation,
    SubtractOperation,
    get_expression_json_schema,
)


class TestMathSchema(unittest.TestCase):
    """Tests validating constraints and nesting rules on the Pydantic math schema."""

    def test_json_schema_contains_all_six_operations(self) -> None:
        schema = get_expression_json_schema()
        defs = schema.get("$defs", {})
        expected_models = {
            "AddOperation",
            "MultiplyOperation",
            "SubtractOperation",
            "DivideOperation",
            "PowerOperation",
            "NthRootOperation",
        }
        self.assertTrue(expected_models.issubset(defs.keys()))
        self.assertEqual(defs["AddOperation"]["properties"]["operands"]["minItems"], 2)
        self.assertEqual(
            defs["MultiplyOperation"]["properties"]["operands"]["minItems"], 2
        )
        for model_name in expected_models:
            self.assertFalse(defs[model_name].get("additionalProperties", True))

    def test_variadic_add_and_multiply_accept_lists(self) -> None:
        expr_add = MathExpression.model_validate(
            {"op": "add", "operands": [1, 2, 3.5, 4]}
        )
        self.assertIsInstance(expr_add.root, AddOperation)
        self.assertEqual(len(expr_add.root.operands), 4)

        expr_mul = MathExpression.model_validate(
            {"op": "multiply", "operands": [2, 3, 5]}
        )
        self.assertIsInstance(expr_mul.root, MultiplyOperation)
        self.assertEqual(len(expr_mul.root.operands), 3)

    def test_variadic_rejects_fewer_than_two_operands(self) -> None:
        for op in ("add", "multiply"):
            with self.subTest(op=op):
                with self.assertRaises(ValidationError):
                    MathExpression.model_validate({"op": op, "operands": [5]})
                with self.assertRaises(ValidationError):
                    MathExpression.model_validate({"op": op, "operands": []})

    def test_binary_operations_accept_strictly_two_operands(self) -> None:
        cases = [
            ({"op": "subtract", "left": 10, "right": 4}, SubtractOperation),
            ({"op": "divide", "numerator": 9, "denominator": 3}, DivideOperation),
            ({"op": "power", "base": 2, "exponent": 8}, PowerOperation),
            ({"op": "nth_root", "radicand": 27, "n": 3}, NthRootOperation),
        ]
        for payload, expected_cls in cases:
            with self.subTest(op=payload["op"]):
                expr = MathExpression.model_validate(payload)
                self.assertIsInstance(expr.root, expected_cls)

    def test_binary_operations_reject_extra_or_missing_operands(self) -> None:
        invalid_payloads = [
            {"op": "subtract", "left": 10},
            {"op": "subtract", "left": 10, "right": 4, "extra": 1},
            {"op": "divide", "numerator": 10},
            {"op": "divide", "numerator": 10, "denominator": 2, "third": 3},
            {"op": "power", "base": 2},
            {"op": "nth_root", "radicand": 16},
            {"op": "subtract", "operands": [10, 4, 1]},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    MathExpression.model_validate(payload)

    def test_nested_operations_across_all_types(self) -> None:
        nested = {
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
        expr = MathExpression.model_validate(nested)
        self.assertIsInstance(expr.root, AddOperation)
        self.assertIsInstance(expr.root.operands[0], MultiplyOperation)
        self.assertIsInstance(expr.root.operands[1], PowerOperation)

    def test_numeric_literal_and_invalid_values(self) -> None:
        self.assertEqual(MathExpression.model_validate(42).root, 42)
        self.assertEqual(MathExpression.model_validate(-3.25).root, -3.25)

        invalid_inputs = [
            True,
            False,
            "42",
            math.nan,
            math.inf,
            -math.inf,
            {"op": "modulo", "left": 5, "right": 2},
            {"op": "add", "operands": [True, 2]},
            {"op": "divide", "numerator": "10", "denominator": 2},
        ]
        for bad in invalid_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValidationError):
                    MathExpression.model_validate(bad)


def main() -> int:
    """Print the complete Pydantic JSON schema and run schema validation tests."""
    schema = get_expression_json_schema()
    print("=== Complete MathExpression JSON Schema ===")
    print(json.dumps(schema, indent=2))
    print("\n=== Running Schema Tests ===")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMathSchema)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

