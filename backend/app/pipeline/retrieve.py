import logging
import re
import time

import arxiv

from app.config import get_settings
from app.models import Author, Paper

logger = logging.getLogger(__name__)

# arXiv ids carry a revision suffix (2401.12345v2). Two revisions are the same
# paper, so the suffix is stripped for the canonical id — otherwise every revised
# paper is retrieved, extracted, and later cached twice.
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _canonical_id(entry_id: str) -> str:
    return _VERSION_SUFFIX.sub("", entry_id.rstrip("/").split("/")[-1])


def _category_clause(settings) -> str:
    cats = [c.strip() for c in settings.arxiv_categories.split(",") if c.strip()]
    if not cats:
        return ""
    return " OR ".join(f"cat:{c}" for c in cats)


def _to_paper(result) -> Paper:
    return Paper(
        arxiv_id=_canonical_id(result.entry_id),
        title=result.title,
        abstract=result.summary,
        authors=[Author(name=a.name) for a in result.authors],
        categories=result.categories,
        published=result.published.date().isoformat(),
        url=result.entry_id,
    )


def retrieve_candidates(queries: str | list[str]) -> tuple[list[Paper], int]:
    """Retrieve candidates for one or more arXiv queries.

    Results are unioned and deduped on the canonical arxiv_id, keeping the first
    occurrence — queries are ordered most- to least-faithful to the user's topic,
    so earlier hits win. The max_candidates cap applies to the union, keeping cost
    bounded regardless of how many queries were issued.
    """
    settings = get_settings()
    if isinstance(queries, str):
        queries = [queries]

    start = time.monotonic()
    client = arxiv.Client(page_size=settings.arxiv_page_size, num_retries=3)
    category_clause = _category_clause(settings)

    papers: list[Paper] = []
    seen: set[str] = set()
    # Divide the budget across queries so no single query starves the rest, but
    # allow each to overshoot when others return little.
    per_query = max(1, settings.max_candidates // max(1, len(queries)))

    for query in queries:
        if len(papers) >= settings.max_candidates:
            break
        full_query = f"({query}) AND ({category_clause})" if category_clause else query
        remaining = settings.max_candidates - len(papers)
        search = arxiv.Search(
            query=full_query,
            max_results=min(settings.max_candidates, max(per_query, remaining)),
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = list(client.results(search))
        except Exception:
            # One malformed or unlucky query must not sink retrieval when other
            # queries can still supply candidates.
            logger.exception("arxiv query failed query=%r", full_query)
            continue

        added = 0
        for result in results:
            if len(papers) >= settings.max_candidates:
                break
            paper = _to_paper(result)
            if paper.arxiv_id in seen:
                continue
            seen.add(paper.arxiv_id)
            papers.append(paper)
            added += 1
        logger.debug("arxiv query=%r hits=%d new=%d", full_query, len(results), added)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "retrieve done queries=%d unique_papers=%d ms=%d",
        len(queries),
        len(papers),
        elapsed_ms,
    )
    return papers, elapsed_ms
