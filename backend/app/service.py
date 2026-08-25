import logging
import uuid
from datetime import date

from app import budget, run_status, storage
from app.llm import AllModelsRateLimited
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


def run_pipeline(
    topic: str, published_after: date | None = None, run_id: str | None = None
) -> SearchResponse:
    """Run the full expand -> retrieve -> rerank -> extract -> synthesize pipeline.

    run_id is normally generated here; the API layer passes one in when it has
    already pre-registered the run with run_status (so an SSE client connecting
    right after POST /api/search sees QUEUED instead of a 404). Stage-transition
    calls into run_status are harmless no-ops for a run_id nothing registered —
    direct callers (tests, the eval harness) are unaffected either way.

    Raises PipelineError on a hard failure (retrieve or rerank). Expansion,
    extraction, and synthesis failures are tolerated and reflected in the
    response.
    """
    run_id = run_id or uuid.uuid4().hex
    logger.info("pipeline started run_id=%s topic=%r", run_id, topic)

    prompt_tokens = 0
    completion_tokens = 0

    # Expansion never raises — it falls back to [topic] — so a failure here costs
    # recall, not the run.
    queries, expand_ms, used_prompt, used_completion = expand_query(topic)
    prompt_tokens += used_prompt
    completion_tokens += used_completion

    run_status.set_state(run_id, "RETRIEVING")
    run_status.set_stage(run_id, "retrieve", status="running")
    try:
        candidates, retrieve_ms = retrieve_candidates(queries, published_after)
    except Exception as exc:
        logger.exception("retrieve failed topic=%r", topic)
        run_status.set_stage(run_id, "retrieve", status="error")
        run_status.set_error(run_id, "Failed to retrieve papers from arXiv")
        raise PipelineError("retrieve", "Failed to retrieve papers from arXiv") from exc
    run_status.set_stage(run_id, "retrieve", status="done", count=len(candidates), ms=retrieve_ms)

    if not candidates:
        logger.info("pipeline zero candidates run_id=%s topic=%r", run_id, topic)
        budget.add(prompt_tokens, completion_tokens)
        run_status.set_state(run_id, "COMPLETE")
        response = SearchResponse(
            run_id=run_id,
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
        _save_run_best_effort(run_id, response)
        return response

    run_status.set_state(run_id, "RERANKING")
    run_status.set_stage(run_id, "rerank", status="running")
    try:
        ranked, rerank_ms, used_prompt, used_completion = rerank_candidates(topic, candidates)
        prompt_tokens += used_prompt
        completion_tokens += used_completion
    except AllModelsRateLimited:
        # Every model in the pool is rate-limited. That is a "come back later",
        # not a broken pipeline, so it must reach the client as 429 — wrapping it
        # in PipelineError here would surface it as a 502 and read as our bug.
        logger.warning("rerank rate-limited topic=%r", topic)
        budget.add(prompt_tokens, completion_tokens)
        run_status.set_stage(run_id, "rerank", status="error")
        run_status.set_error(run_id, "All models are rate-limited")
        raise
    except Exception as exc:
        logger.exception("rerank failed topic=%r", topic)
        # Tokens already spent still count against the budget.
        budget.add(prompt_tokens, completion_tokens)
        run_status.set_stage(run_id, "rerank", status="error")
        run_status.set_error(run_id, "Failed to rerank papers")
        raise PipelineError("rerank", "Failed to rerank papers") from exc
    run_status.set_stage(run_id, "rerank", status="done", kept=len(ranked), ms=rerank_ms)

    run_status.set_state(run_id, "EXTRACTING")
    run_status.set_stage(run_id, "extract", status="running", done=0, total=len(ranked))

    def _extract_progress(done: int, total: int) -> None:
        run_status.set_stage(run_id, "extract", done=done, total=total)

    # Extraction: per-paper failures are tolerated — extract_errors tracks them
    papers, extract_ms, extract_errors, used_prompt, used_completion = extract_papers(
        ranked, on_progress=_extract_progress
    )
    prompt_tokens += used_prompt
    completion_tokens += used_completion
    run_status.set_stage(run_id, "extract", status="done", ms=extract_ms)

    # Synthesis: works from successfully-extracted papers only. If every paper
    # failed extraction there is nothing to synthesize, so skip the call.
    landscape = None
    synthesize_ms = 0
    if any(p.extract_status == "done" for p in papers):
        run_status.set_state(run_id, "SYNTHESIZING")
        run_status.set_stage(run_id, "synthesize", status="running")
        try:
            landscape, synthesize_ms, used_prompt, used_completion = synthesize_landscape(
                topic, papers
            )
            prompt_tokens += used_prompt
            completion_tokens += used_completion
            run_status.set_stage(run_id, "synthesize", status="done", ms=synthesize_ms)
        except Exception:
            logger.exception("synthesize failed topic=%r", topic)
            landscape = None  # non-fatal: still return papers + extractions
            run_status.set_stage(run_id, "synthesize", status="error")
    else:
        run_status.set_stage(run_id, "synthesize", status="skipped")

    budget.add(prompt_tokens, completion_tokens)
    run_status.set_state(run_id, "COMPLETE")

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
    response = SearchResponse(
        run_id=run_id,
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
    _save_run_best_effort(run_id, response)
    return response


def _save_run_best_effort(run_id: str, response: SearchResponse) -> None:
    """Persist a run's outcome. A storage failure must not fail an otherwise
    successful pipeline run — the caller already has a good result in hand."""
    try:
        storage.save_run(run_id, response)
    except Exception:
        logger.exception("failed to persist run_id=%s", run_id)
