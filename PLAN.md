# Semantic Math Text Engine — Implementation Plan

A layered natural-language math solver that converts statement problems into constrained tool calls via `llama3.2:1b`, validates them with Pydantic, builds and evaluates an explicit C++-style typed AST/graph, and visualizes the result in LaTeX and Mermaid.

---

## 1. Architecture Overview

```text
User Statement
   │
   ▼
[1. IntentInterpreter]  ── Ollama Python API (llama3.2:1b + function/tool calling)
   │  (raw tool call JSON)
   ▼
[2. Pydantic Schema]    ── Strict validation & nested operand constraints
   │  (validated expression AST)
   ▼
[3. MathGraph]          ── C++-style typed nodes (NodeType Enum) + post-order DFS solver
   │  ├── answer (float)
   │  └── graph root / node arena
   ▼
[4. GraphVisualizer]    ── Shared graph traversal → Mermaid flowchart & LaTeX expression
   │
   ▼
[5. Chat Interface]     ── Phased integration: (i) text input → (ii) parse/solve → (iii) render LaTeX
```

---

## 2. Proposed Module Layout

```text
semantic_math_text_engine/
├── PLAN.md
├── pyproject.toml / requirements.txt
├── src/
│   └── semantic_math/
│       ├── __init__.py
│       ├── schema.py        # Pydantic models & constrained tool schemas
│       ├── graph.py         # NodeType enum, MathNode, MathGraph builder & DFS evaluator
│       ├── visualizer.py    # GraphVisualizer (to_mermaid, to_latex)
│       ├── interpreter.py   # IntentInterpreter wrapping Ollama (llama3.2:1b)
│       └── app.py           # Phased chat interface (integration stage)
└── tests/
    ├── test_schema.py
    ├── test_graph.py
    ├── test_visualizer.py
    └── test_interpreter.py
```

---

## 3. Build & Test Plan (Step-by-Step Upon Request)

Each component will be implemented and unit-tested independently before any end-to-end integration.

### Step 1: Constrained Operation Schema (`schema.py`)
- **Purpose:** Define valid JSON schemas and recursive validation models using **Pydantic v2**.
- **Constrained Operations (6 total):**
  - **Variadic (`operands: list[Operand]`, `min_length=2`):** `add`, `multiply`
  - **Strictly Binary (2 operands):** `subtract(left, right)`, `divide(numerator, denominator)`, `power(base, exponent)`, `nth_root(radicand, n)`
- **Recursive Nesting:** An `Operand` is `StrictNumber | OperationExpr` (rejecting `bool` values), allowing arbitrarily nested operations.
- **Test Focus:** Arity enforcement (variadic vs. binary-only), deep nesting, invalid type/op rejection, JSON schema export.

### Step 2: C++-Style Graph & Solver (`graph.py`)
- **Purpose:** Represent and evaluate the validated expression using explicit typed nodes instead of `eval`.
- **Core Types:**
  - `NodeType(Enum)`: `NUMBER`, `ADD`, `SUBTRACT`, `MULTIPLY`, `DIVIDE`, `POWER`, `NTH_ROOT`.
  - `MathNode`: `id: int`, `node_type: NodeType`, `value: float | None`, `children: list[MathNode]`.
  - `MathGraph`: builds nodes from the Pydantic AST (arena list + `root` reference) and checks structural arity invariants.
- **Traversal (`evaluate()`):** Post-order DFS — descend down operand children until reaching `NUMBER` leaves, bubble numeric values back up a layer to parent operations, and reduce.
- **Safety / Math Rules:** Guard against division by zero, `n == 0` in `nth_root`, and even roots of negative numbers (while supporting real odd roots such as $\sqrt[3]{-8} = -2$).
- **Test Focus:** All 6 operations, multi-level nested trees, precedence/order for non-commutative ops, math domain error cases.

### Step 3: Graph Visualizer (`visualizer.py`)
- **Purpose:** Produce human-readable visualizations from `MathGraph` using the same graph traversal pattern.
- **Methods on `GraphVisualizer`:**
  - `to_mermaid(graph) -> str`: Emits a `flowchart TD` graph with node IDs, type/value labels, and labeled edges where argument order matters (e.g., `base` / `exp`, `radicand` / `n`).
  - `to_latex(graph) -> str`: Recursively formats LaTeX (`+`, `-`, `\cdot`, `\frac{a}{b}`, `{a}^{b}`, `\sqrt{a}`, `\sqrt[n]{a}`) with automatic precedence-aware parentheses.
- **Test Focus:** Correct parenthesization (e.g., `(2 + 3) \cdot 4` vs. `2 + 3 \cdot 4`), nested fractions/roots, valid Mermaid syntax.

### Step 4: Fuzzy Intent Interpreter (`interpreter.py`)
- **Purpose:** Translate natural-language math problems into constrained tool calls using the **Ollama Python API** (`llama3.2:1b`).
- **Behavior:**
  - Registers the Pydantic-derived tool schema(s) with `ollama.chat(..., tools=...)`.
  - Parses tool-call output through `schema.py` validation and raises clear diagnostics on malformed/unsupported tool calls.
- **Note to Verify During Testing:** Small 1B models (`llama3.2:1b`) can be sensitive to recursive `$defs`/`$ref` in JSON tool schemas. During Step 4 testing, we will benchmark whether 6 individual operation tools or a single structured root tool yields more reliable nested extraction.
- **Test Focus:** Mocked Ollama responses for deterministic CI tests + live `llama3.2:1b` prompt tests on single-step and multi-step word problems.

### Step 5: Phased Chat Interface (`app.py` — Integration Only After Steps 1–4 Pass)
1. **Phase i — Text Input:** Minimal chat loop accepting and validating user statement input.
2. **Phase ii — Tool Call & Solver:** Wire input → `IntentInterpreter` → `schema` → `MathGraph.evaluate()`, inspecting raw tool JSON and numeric result.
3. **Phase iii — LaTeX (& Mermaid) Rendering:** Display properly rendered LaTeX equations (and optional Mermaid graph) alongside the final answer.

---

## 4. Environment Prerequisites (To Set Up When Starting Code)
- Python 3.10 (available) + virtual environment (`.venv`)
- Packages needed: `pydantic>=2.0`, `ollama`, `pytest` (not yet installed in system Python)
- Ollama daemon running locally with `llama3.2:1b` pulled (`ollama pull llama3.2:1b`)

---

## 5. Review Checkpoints Before Coding
1. **Binary Argument Names:** Confirm clear names (`subtract(left, right)`, `divide(numerator, denominator)`, `power(base, exponent)`, `nth_root(radicand, n)`) — explicit names help `llama3.2:1b` preserve order.
2. **Phase 5.iii Rendering Target:** Choose preferred UI for rendering LaTeX in the chat interface (e.g., lightweight local web app with KaTeX + Mermaid.js, vs. Streamlit/Gradio).
3. **First Module to Build:** Confirm starting with **Step 1 (`schema.py`)** (or Steps 1 & 2 together) and its unit tests.

