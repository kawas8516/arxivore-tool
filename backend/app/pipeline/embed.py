"""Embedding-based prefilter, run before the (expensive) LLM rerank call.

Off by default (EMBED_PREFILTER_ENABLED=false) — see config.py's comment on
why. Every failure path here falls back to returning `candidates` unchanged,
so enabling this can only cost latency on a misconfigured provider/model, never
recall: a candidate this stage drops in error would otherwise have gone to
rerank anyway, and rerank is what actually decides relevance.
"""

import logging
import math

from openai import OpenAI

from app import storage
from app.config import get_settings
from app.models import Paper

logger = logging.getLogger(__name__)

# Abstracts run long; embedding models have their own token limits and cost
# scales with input size. A few thousand characters is far more than enough
# signal for a similarity ranking.
_MAX_CHARS = 4000


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_and_cache(
    client: OpenAI, model: str, arxiv_ids: list[str], texts: list[str]
) -> dict[str, list[float]]:
    if not texts:
        return {}
    response = client.embeddings.create(model=model, input=texts)
    result: dict[str, list[float]] = {}
    for arxiv_id, item in zip(arxiv_ids, response.data):
        result[arxiv_id] = item.embedding
        try:
            storage.save_embedding(arxiv_id, model, item.embedding)
        except Exception:
            logger.exception("failed to cache embedding arxiv_id=%s", arxiv_id)
    return result


def prefilter_by_similarity(topic: str, candidates: list[Paper]) -> list[Paper]:
    """Keep the top embed_prefilter_keep candidates by cosine similarity to the
    topic. A no-op (returns `candidates` unchanged) when disabled, when there
    are already at or below the keep threshold, or on any failure.
    """
    settings = get_settings()
    if not settings.embed_prefilter_enabled or len(candidates) <= settings.embed_prefilter_keep:
        return candidates

    try:
        client = OpenAI(
            api_key=settings.llm_api_key, base_url=settings.llm_base_url, max_retries=1
        )
        model = settings.llm_embedding_model

        cached: dict[str, list[float]] = {}
        to_embed_ids: list[str] = []
        to_embed_texts: list[str] = []
        for paper in candidates:
            vector = storage.get_embedding(paper.arxiv_id, model)
            if vector is not None:
                cached[paper.arxiv_id] = vector
            else:
                to_embed_ids.append(paper.arxiv_id)
                to_embed_texts.append(f"{paper.title}\n{paper.abstract}"[:_MAX_CHARS])

        fresh = _embed_and_cache(client, model, to_embed_ids, to_embed_texts)
        vectors = {**cached, **fresh}

        # The topic changes every call, so it's never worth caching.
        topic_vector = client.embeddings.create(model=model, input=[topic[:_MAX_CHARS]]).data[
            0
        ].embedding

        scored = sorted(
            candidates,
            key=lambda p: _cosine(topic_vector, vectors[p.arxiv_id])
            if p.arxiv_id in vectors
            else -1.0,
            reverse=True,
        )
        kept = scored[: settings.embed_prefilter_keep]
        logger.info(
            "embed prefilter kept %d/%d candidates (cache_hits=%d, embedded=%d)",
            len(kept),
            len(candidates),
            len(cached),
            len(fresh),
        )
        return kept
    except Exception:
        logger.exception("embed prefilter failed — falling back to unfiltered candidates")
        return candidates
