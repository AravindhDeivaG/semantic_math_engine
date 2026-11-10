"""End-to-end semantic math engine tying LLM extraction, graph solving, and visualization."""

from dataclasses import dataclass
from typing import Any, Optional

from semantic_math.graph import MathGraph, MathGraphError
from semantic_math.interpreter import LLMCaller
from semantic_math.schema import MathExpression, ParserOutput
from semantic_math.visualizer import GraphVisualizer

DEFAULT_MATH_SYSTEM_PROMPT = """You are a mathematical expression parser.

Convert the user's request into a nested mathematical expression.

Available operations:

1. add:
   - Accepts two or more operands in "operands".
   - Each operand can be a number or another expression.

2. multiply:
   - Accepts two or more operands in "operands".
   - Each operand can be a number or another expression.

3. subtract:
   - Accepts strictly two operands: "left" and "right" (computes left - right).
   - "subtract A from B" and "A subtracted from B" mean left = B, right = A.

4. divide:
   - Accepts strictly two operands: "numerator" and "denominator" (computes numerator / denominator).

5. power:
   - Accepts strictly two operands: "base" and "exponent" (computes base ** exponent).

6. nth_root:
   - Accepts strictly two operands: "radicand" and "n" (computes the nth root of radicand).

Rules:

- Represent mathematical operations using the specified JSON structure.
- Nest operations whenever one operation is an operand of another.
- Identify the outer operation first when clauses are nested.
- Do not calculate the expression.
- Do not invent operations.
- If the request cannot be represented using ONLY add, subtract, multiply, divide, power, and nth_root, return null for expression.

Examples:

Input: Add 1 and 2.

Output:
{
    "expression": {
        "op": "add",
        "operands": [1, 2]
    }
}

Input: Find the product of 3, 4, and 5.

Output:
{
    "expression": {
        "op": "multiply",
        "operands": [3, 4, 5]
    }
}

Input: Subtract 10 from the sum of 20 and 33.

Output:
{
    "expression": {
        "op": "subtract",
        "left": {
            "op": "add",
            "operands": [20, 33]
        },
        "right": 10
    }
}

Input: Add 3 to the 5 subtracted from 20.

Output:
{
    "expression": {
        "op": "add",
        "operands": [
            3,
            {
                "op": "subtract",
                "left": 20,
                "right": 5
            }
        ]
    }
}

Input: Divide the cube root of 8 by 2 raised to the power of 3.

Output:
{
    "expression": {
        "op": "divide",
        "numerator": {
            "op": "nth_root",
            "radicand": 8,
            "n": 3
        },
        "denominator": {
            "op": "power",
            "base": 2,
            "exponent": 3
        }
    }
}

Input: Compute the sine of 30 degrees.

Output:
{
    "expression": null
}"""


class UnsupportedMathIntentError(MathGraphError):
    """Raised when the LLM determines the input cannot use the 6 supported operations."""


@dataclass(frozen=True)
class EngineOutput:
    """Structured result bundle returned by SemanticMathEngine.solve."""

    prompt: str
    raw_json: Any
    expression: MathExpression
    graph: MathGraph
    result: float
    latex: str
    equation_latex: str
    mermaid: str


class SemanticMathEngine:
    """Orchestrates prompt creation, LLM JSON extraction, MathGraph solving, and visualization."""

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        visualizer: Optional[GraphVisualizer] = None,
        system_prompt: str = DEFAULT_MATH_SYSTEM_PROMPT,
        model: Optional[str] = None,
    ) -> None:
        """Initialize engine components and reset cached instance state."""
        self.system_prompt = system_prompt
        self.llm_caller = (
            llm_caller
            if llm_caller is not None
            else LLMCaller(model=model, system_prompt=system_prompt)
        )
        self.visualizer = visualizer if visualizer is not None else GraphVisualizer()

        self.last_prompt: Optional[str] = None
        self.raw_json: Any = None
        self.expression: Optional[MathExpression] = None
        self.graph: Optional[MathGraph] = None
        self.result: Optional[float] = None
        self.latex: Optional[str] = None
        self.equation_latex: Optional[str] = None
        self.mermaid: Optional[str] = None

    @property
    def model(self) -> str:
        """Return the configured Ollama model name for this engine."""
        return self.llm_caller.model

    def build_prompt(self, user_input: str) -> str:
        """Normalize and construct the prompt sent to the LLM caller."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("User input must be a non-empty math statement.")
        return user_input.strip()

    def solve(self, user_input: str) -> EngineOutput:
        """Run the full pipeline and store graph, result, and visualizations on self."""
        prompt = self.build_prompt(user_input)
        raw_payload = self.llm_caller.call(
            prompt=prompt,
            schema=ParserOutput,
            system_prompt=self.system_prompt,
        )
        parsed_output = ParserOutput.model_validate(raw_payload)
        if parsed_output.expression is None:
            raise UnsupportedMathIntentError(
                "Request cannot be represented using only add, subtract, multiply, divide, power, and nth_root."
            )

        expression = MathExpression(parsed_output.expression)
        raw_expr_json = raw_payload["expression"]
        graph = MathGraph.from_expression(expression)
        result = graph.evaluate()
        latex = self.visualizer.to_latex(graph)
        equation_latex = self.visualizer.to_equation_latex(graph)
        mermaid = self.visualizer.to_mermaid(graph)

        self.last_prompt = prompt
        self.raw_json = raw_expr_json
        self.expression = expression
        self.graph = graph
        self.result = result
        self.latex = latex
        self.equation_latex = equation_latex
        self.mermaid = mermaid

        return EngineOutput(
            prompt=prompt,
            raw_json=raw_expr_json,
            expression=expression,
            graph=graph,
            result=result,
            latex=latex,
            equation_latex=equation_latex,
            mermaid=mermaid,
        )
