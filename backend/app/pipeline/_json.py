"""Shared plumbing for parsing and accounting LLM responses.

Every pipeline stage speaks JSON to the model, so fence stripping, schema
validation, and token accounting live here rather than being reimplemented (or
forgotten) per stage.
"""

import json
import logging
from typing import TypeVar

from openai import BadRequestError, RateLimitError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# Ask the provider for JSON mode. Not every model on a routing provider honours
# it, so this is an optimisation on top of parsing, never a substitute for it.
_JSON_MODE = {"type": "json_object"}


class LLMOutputError(ValueError):
    """Raised when a model response is missing, unparseable, or off-schema."""


class RateLimitExceeded(LLMOutputError):
    """Raised on a 429 from the provider.

    Distinguished from other LLMOutputErrors so callers can retry against a
    fallback model instead of exhausting the OpenAI client's own retry budget
    against a daily quota wall that retrying can't move. See RELEASE.md's
    recorded 7/18-vs-16/18 extraction gap — that inconsistency is this failure
    mode, not a transient one.
    """


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

    Returns (validated, prompt_tokens, completion_tokens). Raises
    RateLimitExceeded on a 429, or LLMOutputError if the response is empty,
    unparseable, or off-schema — everything else a model can disappoint with.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                response_format=_JSON_MODE,
                messages=messages,
            )
        except BadRequestError:
            # Model or route rejected response_format; parsing handles it anyway.
            # Only a 400 falls through here — rate limits and auth errors
            # propagate rather than burning a second call.
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
    except RateLimitError as exc:
        raise RateLimitExceeded(f"model {model!r} rate limited: {exc}") from exc

    validated = parse_model(content_of(response), schema)
    prompt_tokens, completion_tokens = usage_of(response)
    return validated, prompt_tokens, completion_tokens


def call_json_with_fallback(
    client,
    *,
    model: str,
    fallback_model: str,
    system: str,
    user: str,
    max_tokens: int,
    schema: type[M],
) -> tuple[M, int, int]:
    """Call `model`; on a 429, retry once against `fallback_model`.

    A daily free-tier quota doesn't recover mid-run, so client-side retries
    against the same model just burn wall-clock. `fallback_model=""` disables
    this and behaves exactly like `call_json`.
    """
    try:
        return call_json(
            client, model=model, system=system, user=user, max_tokens=max_tokens, schema=schema
        )
    except RateLimitExceeded:
        if not fallback_model:
            raise
        logger.warning("model %r rate limited — falling back to %r", model, fallback_model)
        return call_json(
            client,
            model=fallback_model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            schema=schema,
        )


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
