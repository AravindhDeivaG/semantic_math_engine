"""Unit tests (and optional single-prompt live runner) for semantic_math.engine."""

import argparse
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.engine import EngineOutput, SemanticMathEngine  # noqa: E402
from semantic_math.graph import MathGraph  # noqa: E402
from semantic_math.interpreter import LLMCaller  # noqa: E402


class _SequencedFakeClient:
    """Fake Ollama client returning mapped JSON expressions by user prompt substring."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.recorded_prompts: list[str] = []

    def chat(self, **kwargs: Any) -> Any:
        user_msg = kwargs["messages"][-1]["content"]
        self.recorded_prompts.append(user_msg)
        for needle, payload in self.responses.items():
            if needle in user_msg:
                return SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
        raise AssertionError(f"Unexpected prompt in test double: {user_msg!r}")


class TestSemanticMathEngine(unittest.TestCase):
    """Tests for end-to-end SemanticMathEngine state, results, and LaTeX outputs."""

    def test_sample_problems_and_instance_variables(self) -> None:
        cases = [
            (
                "Add 1 to 3",
                {"op": "add", "operands": [1, 3]},
                4.0,
                "1 + 3",
                "1 + 3 = 4",
            ),
            (
                "Can you please add 5 and 4 and 1000",
                {"op": "add", "operands": [5, 4, 1000]},
                1009.0,
                "5 + 4 + 1000",
                "5 + 4 + 1000 = 1009",
            ),
            (
                "Can you subtract 10 from sum of 20 and 33",
                {
                    "op": "subtract",
                    "left": {"op": "add", "operands": [20, 33]},
                    "right": 10,
                },
                43.0,
                "20 + 33 - 10",
                "20 + 33 - 10 = 43",
            ),
            (
                "Can you add 1 to the 20 subtracted from 90",
                {
                    "op": "add",
                    "operands": [1, {"op": "subtract", "left": 90, "right": 20}],
                },
                71.0,
                "1 + 90 - 20",
                "1 + 90 - 20 = 71",
            ),
            (
                "Can you find the product of 6, 7, and 2",
                {"op": "multiply", "operands": [6, 7, 2]},
                84.0,
                r"6 \cdot 7 \cdot 2",
                r"6 \cdot 7 \cdot 2 = 84",
            ),
        ]

        fake_client = _SequencedFakeClient({text: payload for text, payload, *_ in cases})
        caller = LLMCaller(model="llama3.2:1b", client=fake_client)
        engine = SemanticMathEngine(llm_caller=caller)

        # Initial instance state is empty
        self.assertIsNone(engine.graph)
        self.assertIsNone(engine.result)
        self.assertIsNone(engine.latex)
        self.assertIsNone(engine.mermaid)

        for user_text, expected_json, expected_val, expected_tex, expected_eq in cases:
            with self.subTest(prompt=user_text):
                output = engine.solve(user_text)
                self.assertIsInstance(output, EngineOutput)

                # Returned bundle matches expectations
                self.assertEqual(output.raw_json, expected_json)
                self.assertAlmostEqual(output.result, expected_val)
                self.assertEqual(output.latex, expected_tex)
                self.assertEqual(output.equation_latex, expected_eq)
                self.assertTrue(output.mermaid.startswith("flowchart TD"))

                # Instance attributes on engine reflect the latest solve call
                self.assertEqual(engine.last_prompt, f"Math problem: {user_text}")
                self.assertEqual(engine.raw_json, expected_json)
                self.assertIsInstance(engine.graph, MathGraph)
                self.assertAlmostEqual(engine.result or 0.0, expected_val)
                self.assertEqual(engine.latex, expected_tex)
                self.assertEqual(engine.equation_latex, expected_eq)
                self.assertEqual(engine.mermaid, output.mermaid)

    def test_empty_prompt_rejected(self) -> None:
        engine = SemanticMathEngine(
            llm_caller=LLMCaller(client=_SequencedFakeClient({}))
        )
        with self.assertRaises(ValueError):
            engine.solve("   ")


def _run_live_prompt(prompt_text: str) -> int:
    engine = SemanticMathEngine()
    out = engine.solve(prompt_text)
    print(f"Prompt   : {out.prompt}")
    print(f"JSON     : {json.dumps(out.raw_json)}")
    print(f"Result   : {out.result}")
    print(f"LaTeX    : {out.latex}")
    print(f"Equation : {out.equation_latex}")
    print("Mermaid  :")
    print(out.mermaid)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--live", type=str, default=None)
    args, remaining = parser.parse_known_args()
    if args.live is not None:
        raise SystemExit(_run_live_prompt(args.live))
    sys.argv = [sys.argv[0], *remaining]
    unittest.main(verbosity=2)

