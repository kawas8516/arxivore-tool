"""Daily token ledger enforcing DAILY_TOKEN_BUDGET.

In-process and single-instance, mirroring the rate limiter in app/api/search.py:
fine for v1, needs a shared store before horizontal scaling. Closing the loop
here is what makes the configured spend ceiling real rather than decorative.
"""

import logging
import threading
from datetime import date, datetime, timezone

from app.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_day: date = datetime.now(timezone.utc).date()
_prompt_tokens = 0
_completion_tokens = 0


def _roll_over_locked() -> None:
    """Reset the ledger when the UTC day changes. Caller must hold _lock."""
    global _day, _prompt_tokens, _completion_tokens
    today = datetime.now(timezone.utc).date()
    if today != _day:
        logger.info(
            "token budget day rolled over from %s (prompt=%d completion=%d)",
            _day,
            _prompt_tokens,
            _completion_tokens,
        )
        _day = today
        _prompt_tokens = 0
        _completion_tokens = 0


def add(prompt_tokens: int, completion_tokens: int) -> None:
    """Record tokens spent by a completed run."""
    global _prompt_tokens, _completion_tokens
    with _lock:
        _roll_over_locked()
        _prompt_tokens += max(0, prompt_tokens)
        _completion_tokens += max(0, completion_tokens)


def spent() -> int:
    """Total tokens spent today."""
    with _lock:
        _roll_over_locked()
        return _prompt_tokens + _completion_tokens


def remaining() -> int:
    return max(0, get_settings().daily_token_budget - spent())


def exhausted() -> bool:
    """True when today's spend has reached the configured ceiling.

    Checked before a run starts. A run already in flight is allowed to finish, so
    the ceiling can be overshot by at most one run — bounded by the per-run token
    caps, and far preferable to abandoning a half-paid-for pipeline.
    """
    return remaining() <= 0


def reset() -> None:
    """Clear the ledger. For tests."""
    global _day, _prompt_tokens, _completion_tokens
    with _lock:
        _day = datetime.now(timezone.utc).date()
        _prompt_tokens = 0
        _completion_tokens = 0
