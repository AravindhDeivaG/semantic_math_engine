"""End-to-end semantic math engine tying LLM extraction, graph solving, and visualization."""

from dataclasses import dataclass
from typing import Any, Optional

from semantic_math.graph import MathGraph
from semantic_math.interpreter import LLMCaller
from semantic_math.schema import MathExpression
from semantic_math.visualizer import GraphVisualizer

DEFAULT_MATH_SYSTEM_PROMPT = """Convert the user's math statement into a JSON expression tree matching the schema.
Use ONLY these 6 operations (do NOT compute the final numeric answer yourself):
- add: {"op": "add", "operands": [a, b, ...]} (2 or more operands)
- multiply: {"op": "multiply", "operands": [a, b, ...]} (2 or more operands)
- subtract: {"op": "subtract", "left": a, "right": b} (computes left - right)
- divide: {"op": "divide", "numerator": a, "denominator": b} (computes numerator / denominator)
- power: {"op": "power", "base": a, "exponent": b} (computes base ** exponent)
- nth_root: {"op": "nth_root", "radicand": a, "n": b} (computes nth root of radicand)

Word-order and nesting rules:
- Identify the OUTERMOST action first; nested clauses go inside operands.
- "subtract X from Y" and "X subtracted from Y" ALWAYS mean left=Y, right=X.
- "add X to <expr>" ALWAYS has outer "op": "add" with operands [X, <expr>].

Examples:
"Add 2 to 5" -> {"op": "add", "operands": [2, 5]}
"Multiply 3, 4 and 5" -> {"op": "multiply", "operands": [3, 4, 5]}
"Divide 12 by 4" -> {"op": "divide", "numerator": 12, "denominator": 4}
"2 to the power of 5" -> {"op": "power", "base": 2, "exponent": 5}
"Cube root of 27" -> {"op": "nth_root", "radicand": 27, "n": 3}
"Subtract 4 from sum of 10 and 3" -> {"op": "subtract", "left": {"op": "add", "operands": [10, 3]}, "right": 4}
"Add 2 to the 5 subtracted from 20" -> {"op": "add", "operands": [2, {"op": "subtract", "left": 20, "right": 5}]}"""


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
    ) -> None:
        """Initialize engine components and reset cached instance state."""
        self.system_prompt = system_prompt
        self.llm_caller = (
            llm_caller
            if llm_caller is not None
            else LLMCaller(model="llama3.2:1b", system_prompt=system_prompt)
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

    def build_prompt(self, user_input: str) -> str:
        """Normalize and construct the prompt sent to the LLM caller."""
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("User input must be a non-empty math statement.")
        return f"Math problem: {user_input.strip()}"

    def solve(self, user_input: str) -> EngineOutput:
        """Run the full pipeline and store graph, result, and visualizations on self."""
        prompt = self.build_prompt(user_input)
        raw_json = self.llm_caller.call(
            prompt=prompt,
            schema=MathExpression,
            system_prompt=self.system_prompt,
        )
        expression = MathExpression.model_validate(raw_json)
        graph = MathGraph.from_expression(expression)
        result = graph.evaluate()
        latex = self.visualizer.to_latex(graph)
        equation_latex = self.visualizer.to_equation_latex(graph)
        mermaid = self.visualizer.to_mermaid(graph)

        self.last_prompt = prompt
        self.raw_json = raw_json
        self.expression = expression
        self.graph = graph
        self.result = result
        self.latex = latex
        self.equation_latex = equation_latex
        self.mermaid = mermaid

        return EngineOutput(
            prompt=prompt,
            raw_json=raw_json,
            expression=expression,
            graph=graph,
            result=result,
            latex=latex,
            equation_latex=equation_latex,
            mermaid=mermaid,
        )

