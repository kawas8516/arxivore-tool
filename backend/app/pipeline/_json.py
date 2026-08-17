"""Shared plumbing for parsing and accounting LLM responses.

Every pipeline stage speaks JSON to the model, so fence stripping, schema
validation, and token accounting live here rather than being reimplemented (or
forgotten) per stage.
"""

import json
from typing import TypeVar

from openai import BadRequestError
from pydantic import BaseModel, ValidationError

M = TypeVar("M", bound=BaseModel)

# Ask the provider for JSON mode. Not every model on a routing provider honours
# it, so this is an optimisation on top of parsing, never a substitute for it.
_JSON_MODE = {"type": "json_object"}


class LLMOutputError(ValueError):
    """Raised when a model response is missing, unparseable, or off-schema."""


def strip_fences(raw: str) -> str:
    """Remove markdown code fences if the model wraps its JSON output."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = [l for l in lines[1:] if l.strip() != "```"]
        return "\n".join(inner).strip()
    return raw


def content_of(response) -> str:
    """Return the assistant message text, or raise if the response is empty."""
    choices = getattr(response, "choices", None)
    content = choices[0].message.content if choices else None
    if not content:
        raise LLMOutputError("model returned empty content")
    return strip_fences(content.strip())


def parse_model(raw: str, model: type[M]) -> M:
    """Parse `raw` JSON and validate it against `model`.

    Wraps both failure modes in LLMOutputError so callers handle one exception
    type regardless of whether the model emitted bad JSON or valid JSON with the
    wrong shape.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"response was not valid JSON: {exc}") from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"response did not match {model.__name__}: {exc}") from exc


def call_json(
    client,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    schema: type[M],
) -> tuple[M, int, int]:
    """Call the model, expecting JSON, and validate it against `schema`.

    Returns (validated, prompt_tokens, completion_tokens). Raises LLMOutputError
    if the response is empty, unparseable, or off-schema — one exception type for
    every way a model can disappoint.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            response_format=_JSON_MODE,
            messages=messages,
        )
    except BadRequestError:
        # Model or route rejected response_format; parsing handles it anyway.
        # Only a 400 falls through here — rate limits and auth errors propagate
        # rather than burning a second call.
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )

    validated = parse_model(content_of(response), schema)
    prompt_tokens, completion_tokens = usage_of(response)
    return validated, prompt_tokens, completion_tokens


def usage_of(response) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a response.

    Defaults to (0, 0) when the provider omits usage or reports a non-integer, so
    accounting never crashes a run over telemetry it didn't get.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    def _int(value) -> int:
        return value if isinstance(value, int) else 0

    return _int(getattr(usage, "prompt_tokens", 0)), _int(
        getattr(usage, "completion_tokens", 0)
    )
