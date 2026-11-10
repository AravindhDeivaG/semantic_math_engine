"""Ollama LLM caller for schema-constrained math intent extraction."""

import json
from typing import Any, Optional, Union

import ollama
from pydantic import BaseModel, ValidationError

from semantic_math.config import DEFAULT_TEMPERATURE, get_default_model


class LLMCallerError(RuntimeError):
    """Raised when the Ollama call or schema parsing/validation fails."""


def _resolve_json_schema(schema: Union[type[BaseModel], dict[str, Any]]) -> dict[str, Any]:
    """Extract a JSON schema dict from a Pydantic model class or pass through a dict."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_json_schema()
    if isinstance(schema, dict):
        return schema
    raise TypeError("schema must be a Pydantic BaseModel subclass or a JSON schema dict.")


def _extract_message_content(response: Any) -> str:
    """Read message text from either an Ollama ChatResponse object or a dict mock."""
    message = response["message"] if isinstance(response, dict) else response.message
    content = message["content"] if isinstance(message, dict) else message.content
    if not isinstance(content, str) or not content.strip():
        raise LLMCallerError("LLM returned empty message content.")
    return content


class LLMCaller:
    """Calls an Ollama model with a prompt and JSON schema, returning parsed JSON."""

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Configure target model, optional default system prompt, and Ollama client."""
        if model is None:
            resolved_model = get_default_model()
        elif isinstance(model, str) and model.strip():
            resolved_model = model.strip()
        else:
            raise ValueError("Model name must be a non-empty string.")

        self.model = resolved_model
        self.system_prompt = system_prompt
        self.temperature = (
            DEFAULT_TEMPERATURE if temperature is None else temperature
        )
        self.client = client if client is not None else ollama.Client()

    def call(
        self,
        prompt: str,
        schema: Union[type[BaseModel], dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Send prompt and JSON schema to the model and return the parsed JSON result."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        json_schema = _resolve_json_schema(schema)
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        messages: list[dict[str, str]] = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt.strip()})

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=json_schema,
                options={"temperature": self.temperature},
            )
        except Exception as exc:
            raise LLMCallerError(
                f"Ollama call failed for model '{self.model}': {exc}"
            ) from exc

        raw_text = _extract_message_content(response)
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise LLMCallerError(
                f"Model '{self.model}' returned invalid JSON: {raw_text!r}"
            ) from exc

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema.model_validate(parsed)
            except ValidationError as exc:
                raise LLMCallerError(
                    f"Model '{self.model}' JSON failed schema validation: {exc}"
                ) from exc

        return parsed


def call_llm(
    prompt: str,
    schema: Union[type[BaseModel], dict[str, Any]],
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    client: Optional[Any] = None,
) -> Any:
    """Convenience function wrapping LLMCaller for one-off prompt+schema calls."""
    caller = LLMCaller(
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        client=client,
    )
    return caller.call(prompt=prompt, schema=schema)

