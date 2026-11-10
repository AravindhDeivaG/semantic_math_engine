"""Unit tests for semantic_math.interpreter (LLMCaller and call_llm)."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from semantic_math.interpreter import LLMCaller, LLMCallerError, call_llm  # noqa: E402
from semantic_math.schema import MathExpression  # noqa: E402


class _FakeOllamaClient:
    """Deterministic stub mimicking ollama.Client.chat."""

    def __init__(self, response_text: str, as_dict: bool = False, exc: Exception | None = None):
        self.response_text = response_text
        self.as_dict = as_dict
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        if self.as_dict:
            return {"message": {"content": self.response_text}}
        return SimpleNamespace(message=SimpleNamespace(content=self.response_text))


class TestLLMCaller(unittest.TestCase):
    """Tests for LLMCaller and call_llm using mocked Ollama clients."""

    def test_call_with_pydantic_model_and_custom_system_prompt(self) -> None:
        payload = {"op": "add", "operands": [1, 3]}
        fake = _FakeOllamaClient(json.dumps(payload))
        caller = LLMCaller(model="llama3.2:1b", system_prompt="default sys", client=fake)

        result = caller.call("Add 1 to 3", MathExpression)
        self.assertEqual(result, payload)
        self.assertEqual(len(fake.calls), 1)
        recorded = fake.calls[0]
        self.assertEqual(recorded["model"], "llama3.2:1b")
        self.assertEqual(recorded["format"], MathExpression.model_json_schema())
        self.assertEqual(recorded["options"], {"temperature": 0.0})
        self.assertEqual(
            recorded["messages"],
            [
                {"role": "system", "content": "default sys"},
                {"role": "user", "content": "Add 1 to 3"},
            ],
        )

    def test_call_with_raw_dict_schema_and_dict_response(self) -> None:
        raw_schema = {"type": "object", "properties": {"answer": {"type": "number"}}}
        fake = _FakeOllamaClient('{"answer": 42}', as_dict=True)
        result = call_llm(
            prompt="What is the answer?",
            schema=raw_schema,
            model="llama3.2:1b",
            client=fake,
        )
        self.assertEqual(result, {"answer": 42})
        self.assertEqual(fake.calls[0]["format"], raw_schema)

    def test_validation_and_error_handling(self) -> None:
        fake = _FakeOllamaClient("{}")
        caller = LLMCaller(model="llama3.2:1b", client=fake)

        with self.assertRaises(ValueError):
            caller.call("   ", MathExpression)

        with self.assertRaises(TypeError):
            caller.call("Add 1 to 2", schema="not-a-schema")  # type: ignore[arg-type]

        # Empty response text
        with self.assertRaises(LLMCallerError):
            LLMCaller(client=_FakeOllamaClient("   ")).call("Add 1 to 2", MathExpression)

        # Malformed JSON
        with self.assertRaises(LLMCallerError):
            LLMCaller(client=_FakeOllamaClient("not json")).call("Add 1 to 2", MathExpression)

        # Valid JSON violating Pydantic schema (subtract missing 'right')
        with self.assertRaises(LLMCallerError):
            LLMCaller(
                client=_FakeOllamaClient('{"op": "subtract", "left": 5}')
            ).call("Subtract", MathExpression)

        # Client exception propagation wrapped cleanly
        with self.assertRaises(LLMCallerError):
            LLMCaller(
                client=_FakeOllamaClient("{}", exc=ConnectionError("daemon offline"))
            ).call("Add 1 to 2", MathExpression)


if __name__ == "__main__":
    unittest.main(verbosity=2)

