import time
import arxiv

from app.config import get_settings
from app.models import Author, Paper


def retrieve_candidates(topic: str) -> tuple[list[Paper], int]:
    settings = get_settings()

    start = time.monotonic()
    client = arxiv.Client(page_size=settings.arxiv_page_size, num_retries=3)
    search = arxiv.Search(
        query=topic,
        max_results=settings.max_candidates,
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
