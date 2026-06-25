import logging

from app.llm import AllModelsRateLimited
from app.models import SearchResponse
from app.pipeline.retrieve import retrieve_candidates
from app.pipeline.rerank import rerank_candidates
from app.pipeline.extract import extract_papers
from app.pipeline.synthesize import synthesize_landscape

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a hard pipeline stage (retrieve/rerank) fails."""

    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


def run_pipeline(topic: str) -> SearchResponse:
    """Run the full retrieve -> rerank -> extract -> synthesize pipeline.

    Raises PipelineError on a hard failure (retrieve or rerank). Extraction
    and synthesis failures are tolerated and reflected in the response.
    """
    logger.info("pipeline started topic=%r", topic)

    try:
        candidates, retrieve_ms = retrieve_candidates(topic)
    except Exception as exc:
        logger.exception("retrieve failed topic=%r", topic)
        raise PipelineError("retrieve", "Failed to retrieve papers from arXiv") from exc

    if not candidates:
        logger.info("pipeline zero candidates topic=%r", topic)
        return SearchResponse(
            topic=topic,
            candidates_retrieved=0,
            papers_returned=0,
            papers=[],
            retrieve_ms=retrieve_ms,
            rerank_ms=0,
        )

    try:
        ranked, rerank_ms = rerank_candidates(topic, candidates)
    except AllModelsRateLimited:
        # Every model is rate-limited — let the API surface a 429, not a 502.
        logger.warning("rerank rate-limited topic=%r", topic)
        raise
    except Exception as exc:
        logger.exception("rerank failed topic=%r", topic)
        raise PipelineError("rerank", "Failed to rerank papers") from exc

    # Extraction: per-paper failures are tolerated — extract_errors tracks them
    papers, extract_ms, extract_errors = extract_papers(ranked)

    # Synthesis: works from successfully-extracted papers only. If every paper
    # failed extraction there is nothing to synthesize, so skip the call.
    landscape = None
    synthesize_ms = 0
    if any(p.extract_status == "done" for p in papers):
        try:
            landscape, synthesize_ms = synthesize_landscape(topic, papers)
        except Exception:
            logger.exception("synthesize failed topic=%r", topic)
            landscape = None  # non-fatal: still return papers + extractions

    logger.info(
        "pipeline done topic=%r candidates=%d retained=%d extract_ok=%d extract_errors=%d "
        "synthesized=%s retrieve_ms=%d rerank_ms=%d extract_ms=%d synthesize_ms=%d",
        topic,
        len(candidates),
        len(ranked),
        len(ranked) - extract_errors,
        extract_errors,
        landscape is not None,
        retrieve_ms,
        rerank_ms,
        extract_ms,
        synthesize_ms,
    )
    return SearchResponse(
        topic=topic,
        candidates_retrieved=len(candidates),
        papers_returned=len(papers),
        papers=papers,
        landscape=landscape,
        retrieve_ms=retrieve_ms,
        rerank_ms=rerank_ms,
        extract_ms=extract_ms,
        extract_errors=extract_errors,
        synthesize_ms=synthesize_ms,
    )
