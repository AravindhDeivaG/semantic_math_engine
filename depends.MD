# Setup & Dependencies Guide

Follow these steps to create a clean Python virtual environment and install all dependencies required to run and test **Semantic Math Text Engine**.

---

## 1. Prerequisites
- **Python 3.10+** (`python3 --version`)
- **Ollama** (required once the LLM intent interpreter component is added; install from https://ollama.com and pull the target model):
  ```bash
  ollama pull llama3.2:1b
  ```

---

## 2. Create & Activate a Virtual Environment

From the project root (`semantic_math_text_engine/`):

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

*(Note: If you already have a preconfigured environment such as `~/research_env`, you can run commands directly with `~/research_env/bin/python` instead of creating `.venv`.)*

---

## 3. Install Python Packages

### Currently Required (Steps 1–3: `schema.py`, `graph.py` & `visualizer.py`)
```bash
pip install "pydantic>=2.0" typing_extensions
```

### Full Stack (including Step 4: `interpreter.py` with Ollama)
```bash
pip install "pydantic>=2.0" typing_extensions ollama
```

### Package Reference

| Component | Package | Tested Version (`~/research_env`) | Purpose |
| --- | --- | --- | --- |
| Step 1 — Schema (`schema.py`) | `pydantic` | `2.14.3` | Constrained nested operation validation & JSON Schema generation |
| Step 1 — Schema (`schema.py`) | `typing_extensions` | `4.16.0` | `TypeAliasType` support for recursive `Operand` schemas on Python 3.10 |
| Step 2 — Graph (`graph.py`) | *(stdlib only)* | — | `enum`, `dataclasses`, `math`, `typing` |
| Step 3 — Visualizer (`visualizer.py`) | *(stdlib only)* | — | `dataclasses`, `typing` (Mermaid & LaTeX rendering) |
| Step 4 — Interpreter (`interpreter.py`, upcoming) | `ollama` | `0.6.3` | Ollama Python API client for `llama3.2:1b` tool calling |
| Unit Tests (`tests/`) | *(stdlib `unittest`)* | — | Standalone runnable test scripts |

---

## 4. Run Component Tests

With your virtual environment activated:

```bash
python tests/test_schema.py
python tests/test_graph.py
python tests/test_visualizer.py
python -m unittest discover -s tests -v
```

Or using an explicit interpreter path (e.g. `~/research_env`):

```bash
~/research_env/bin/python tests/test_schema.py
~/research_env/bin/python tests/test_graph.py
~/research_env/bin/python tests/test_visualizer.py
~/research_env/bin/python -m unittest discover -s tests -v
```
