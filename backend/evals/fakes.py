"""Deterministic stand-ins for the arXiv and LLM clients.

Offline mode exists because a full live eval is ~315 provider calls, which
exhausts a free-tier daily quota in one invocation. These fakes exercise the real
parsing, validation, and accounting paths end to end at zero cost, so CI can
catch wiring regressions without a key.
"""

import json
import re
import zlib
from datetime import datetime, timezone
from types import SimpleNamespace

_TOPIC_RE = re.compile(r"<topic>\s*(.*?)\s*</topic>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", re.DOTALL)
_ARXIV_ID_RE = re.compile(r'"arxiv_id":\s*"([^"]+)"')


class FakeArxivResult:
    def __init__(self, index: int, query: str):
        # Distinct ids per query keep the union/dedupe path meaningful, while a
        # deliberate overlap on the first few exercises deduplication.
        # crc32, not hash(): Python randomises string hashing per process, which
        # would make "deterministic" fixtures differ between runs and render the
        # committed baseline undiffable.
        shared = index < 3
        slug = "shared" if shared else f"{zlib.crc32(query.encode()) % 9973:04d}"
        self.entry_id = f"https://arxiv.org/abs/24{index:02d}.{slug}v1"
        self.title = f"Fake Paper {index} on {query[:40]}"
        self.summary = (
            f"We study {query[:60]}. We propose a method and report results on "
            "standard benchmarks, improving over prior work."
        )
        author = SimpleNamespace(name="A. Researcher")
        self.authors = [author]
        self.categories = ["cs.LG"]
        self.published = datetime(2024, 6, 1, tzinfo=timezone.utc)


class FakeArxivClient:
    """Returns a deterministic result set per query."""

    def __init__(self, *args, **kwargs):
        pass

    def results(self, search):
        count = min(search.max_results, 20)
        return iter(FakeArxivResult(i, search.query) for i in range(count))


def _completion(payload: dict, prompt_tokens: int, completion_tokens: int):
    text = json.dumps(payload)
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def create(self, *, model, max_tokens, messages, response_format=None):
        user = messages[-1]["content"]
        # Token counts approximate real usage closely enough for trend tracking:
        # ~4 characters per token.
        prompt_tokens = max(1, len(user) // 4)

        if "arXiv API search queries" in user:
            topic = _TOPIC_RE.search(user)
            topic = topic.group(1) if topic else "topic"
            queries = [f'all:"{topic}"', f'abs:"{topic}"', f'ti:"{topic}"']
            return _completion({"queries": queries}, prompt_tokens, 60)

        if "Rank the following candidate papers" in user:
            ids = _ARXIV_ID_RE.findall(user)
            items = [
                {
                    "arxiv_id": arxiv_id,
                    "score": round(1.0 - (i / max(1, len(ids))), 3),
                    "rationale": "Deterministic fake score.",
                }
                for i, arxiv_id in enumerate(ids)
            ]
            return _completion({"items": items}, prompt_tokens, 25 * len(items))

        if "Extract structured information" in user:
            title = _TITLE_RE.search(user)
            title = title.group(1) if title else "paper"
            return _completion(
                {
                    "problem": f"The gap addressed by {title}.",
                    "method": "A proposed approach with a training objective.",
                    "results": "Improves over baselines on standard benchmarks.",
                    "contribution": "A novel component and its evaluation.",
                },
                prompt_tokens,
                120,
            )

        if "Synthesize them into a research landscape" in user:
            ids = _ARXIV_ID_RE.findall(user)
            half = max(1, len(ids) // 2)
            return _completion(
                {
                    "clusters": [
                        {
                            "name": "Primary Approaches",
                            "summary": "Papers sharing the dominant method family.",
                            "arxiv_ids": ids[:half],
                        },
                        {
                            "name": "Alternative Directions",
                            "summary": "Papers pursuing a different tactic.",
                            "arxiv_ids": ids[half:] or ids[:1],
                        },
                    ],
                    "relationships": [
                        {
                            "from_cluster": "Alternative Directions",
                            "to_cluster": "Primary Approaches",
                            "kind": "alternative-to",
                            "description": "Pursues the same goal by other means.",
                        }
                    ],
                    "tensions": ["Scale versus efficiency remains contested."],
                    "open_problems": ["Evaluation beyond standard benchmarks."],
                },
                prompt_tokens,
                400,
            )

        raise AssertionError("fake client received an unrecognised prompt")


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())
