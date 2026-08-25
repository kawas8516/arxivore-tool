"""LLM wrapper for HF Spaces — uses HF Inference API with Phi-4-mini-instruct."""

import os
import time
import logging

from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

_MODEL = "microsoft/Phi-4-mini-instruct"
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        _client = InferenceClient(token=token)
    return _client


def strip_fences(raw: str) -> str:
    """Remove wrapping markdown code fence if the model added one."""
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = [l for l in lines[1:] if l.strip() != "```"]
        return "\n".join(inner).strip()
    return raw


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def complete(
    messages: list[dict],
    *,
    max_tokens: int = 1024,
    pool: list[str] | None = None,  # ignored — kept for API compat with backend
) -> str:
    """Run a chat completion against Phi-4-mini-instruct via HF Inference API."""
    client = _get_client()
    start = time.monotonic()

    response = client.chat_completion(
        messages=messages,
        model=_MODEL,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    elapsed = time.monotonic() - start
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError("model returned empty content")

    logger.debug("llm call model=%s tokens=%d ms=%d", _MODEL, max_tokens, int(elapsed * 1000))
    return content
