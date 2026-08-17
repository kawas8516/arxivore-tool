import logging

from app import budget
from app.models import SearchResponse
from app.pipeline.expand import expand_query
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
    """Run the full expand -> retrieve -> rerank -> extract -> synthesize pipeline.

    Raises PipelineError on a hard failure (retrieve or rerank). Expansion,
    extraction, and synthesis failures are tolerated and reflected in the
    response.
    """
    logger.info("pipeline started topic=%r", topic)

    prompt_tokens = 0
    completion_tokens = 0

    # Expansion never raises — it falls back to [topic] — so a failure here costs
    # recall, not the run.
    queries, expand_ms, used_prompt, used_completion = expand_query(topic)
    prompt_tokens += used_prompt
    completion_tokens += used_completion

    try:
        candidates, retrieve_ms = retrieve_candidates(queries)
    except Exception as exc:
        logger.exception("retrieve failed topic=%r", topic)
        raise PipelineError("retrieve", "Failed to retrieve papers from arXiv") from exc

    if not candidates:
        logger.info("pipeline zero candidates topic=%r", topic)
        budget.add(prompt_tokens, completion_tokens)
        return SearchResponse(
            topic=topic,
            candidates_retrieved=0,
            papers_returned=0,
            papers=[],
            queries=queries,
            retrieve_ms=retrieve_ms,
            rerank_ms=0,
            expand_ms=expand_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    try:
        ranked, rerank_ms, used_prompt, used_completion = rerank_candidates(topic, candidates)
        prompt_tokens += used_prompt
        completion_tokens += used_completion
    except Exception as exc:
        logger.exception("rerank failed topic=%r", topic)
        # Tokens already spent still count against the budget.
        budget.add(prompt_tokens, completion_tokens)
        raise PipelineError("rerank", "Failed to rerank papers") from exc

    # Extraction: per-paper failures are tolerated — extract_errors tracks them
    papers, extract_ms, extract_errors, used_prompt, used_completion = extract_papers(ranked)
    prompt_tokens += used_prompt
    completion_tokens += used_completion

    # Synthesis: works from successfully-extracted papers only. If every paper
    # failed extraction there is nothing to synthesize, so skip the call.
    landscape = None
    synthesize_ms = 0
    if any(p.extract_status == "done" for p in papers):
        try:
            landscape, synthesize_ms, used_prompt, used_completion = synthesize_landscape(
                topic, papers
            )
            prompt_tokens += used_prompt
            completion_tokens += used_completion
        except Exception:
            logger.exception("synthesize failed topic=%r", topic)
            landscape = None  # non-fatal: still return papers + extractions

    budget.add(prompt_tokens, completion_tokens)

    logger.info(
        "pipeline done topic=%r queries=%d candidates=%d retained=%d extract_ok=%d "
        "extract_errors=%d synthesized=%s expand_ms=%d retrieve_ms=%d rerank_ms=%d "
        "extract_ms=%d synthesize_ms=%d prompt_tokens=%d completion_tokens=%d "
        "budget_remaining=%d",
        topic,
        len(queries),
        len(candidates),
        len(ranked),
        len(ranked) - extract_errors,
        extract_errors,
        landscape is not None,
        expand_ms,
        retrieve_ms,
        rerank_ms,
        extract_ms,
        synthesize_ms,
        prompt_tokens,
        completion_tokens,
        budget.remaining(),
    )
    return SearchResponse(
        topic=topic,
        candidates_retrieved=len(candidates),
        papers_returned=len(papers),
        papers=papers,
        landscape=landscape,
        queries=queries,
        retrieve_ms=retrieve_ms,
        rerank_ms=rerank_ms,
        expand_ms=expand_ms,
        extract_ms=extract_ms,
        extract_errors=extract_errors,
        synthesize_ms=synthesize_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
