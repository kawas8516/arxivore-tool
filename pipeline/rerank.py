import time
import logging

from sentence_transformers import CrossEncoder

from models import Paper

logger = logging.getLogger(__name__)

# Loaded once at module level so it caches after the first Spaces cold start.
_RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")


def rerank_candidates(topic: str, candidates: list[Paper], top_k: int = 10) -> tuple[list[Paper], int]:
    """Score and rank candidates using BAAI/bge-reranker-v2-m3 (no LLM cost)."""
    start = time.monotonic()

    pairs = [(topic, f"{p.title}. {p.abstract}") for p in candidates]
    scores = _RERANKER.predict(pairs)

    for paper, score in zip(candidates, scores):
        paper.relevance_score = float(score)

    ranked = sorted(candidates, key=lambda p: p.relevance_score or 0.0, reverse=True)
    top = ranked[:top_k]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("rerank done candidates=%d top_k=%d ms=%d", len(candidates), top_k, elapsed_ms)
    return top, elapsed_ms
