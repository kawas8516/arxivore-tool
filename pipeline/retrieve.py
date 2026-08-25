import time

import arxiv

from models import Author, Paper

_MAX_CANDIDATES = 50
_PAGE_SIZE = 50


def retrieve_candidates(topic: str) -> tuple[list[Paper], int]:
    start = time.monotonic()
    client = arxiv.Client(page_size=_PAGE_SIZE, num_retries=3)
    search = arxiv.Search(
        query=topic,
        max_results=_MAX_CANDIDATES,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        papers.append(
            Paper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                abstract=result.summary,
                authors=[Author(name=a.name) for a in result.authors],
                categories=result.categories,
                published=result.published.date().isoformat(),
                url=result.entry_id,
            )
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    return papers, elapsed_ms
