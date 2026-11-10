"""Central configuration defaults for the Semantic Math Text Engine."""

import os

# DEFAULT_MODEL: str = "llama3.2:1b"
DEFAULT_MODEL: str = "qwen3:4b"
DEFAULT_TEMPERATURE: float = 0.0
MODEL_ENV_VAR: str = "SEMANTIC_MATH_MODEL"


def get_default_model() -> str:
    """Return the configured Ollama model name (env override or DEFAULT_MODEL)."""
    env_model = os.environ.get(MODEL_ENV_VAR, "").strip()
    return env_model or DEFAULT_MODEL

