"""Full-text excerpt fetching for the highest-relevance papers.

Abstracts rarely contain hard numbers, so extraction's `results` field is often
vague ("achieves strong performance") rather than concrete ("+5.2 EM on
NaturalQuestions"). For the top few papers by rerank score, this pulls the PDF
and isolates a results/conclusion-shaped excerpt so extraction has real numbers
to draw from. Off by default (FULL_TEXT_ENABLED) — see config.py's comment.

Every failure here (network, timeout, unparseable PDF, no matching section)
returns None. This stage must degrade to abstract-only extraction, never block
or fail a run over a single stubborn PDF.
"""

import logging
import re
import urllib.request
from io import BytesIO

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Matches a line that's just a section heading, optionally numbered
# ("5. Results", "Conclusion", "Discussion and Future Work"). Deliberately
# broad — arXiv PDF text extraction is inconsistent about whitespace/casing,
# so a strict header format would miss most real papers.
_SECTION_HEADERS = re.compile(
    r"\n\s*(\d+\.?\s*)?(results|experiments?|evaluation|conclusions?|discussion)\b[^\n]{0,40}\n",
    re.IGNORECASE,
)

_FETCH_TIMEOUT_SECONDS = 15
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024  # arXiv PDFs are essentially never larger


def _pdf_url(arxiv_id: str) -> str:
    # arxiv_id is already version-stripped (retrieve.py); the unversioned PDF
    # URL resolves to the latest revision.
    return f"https://arxiv.org/pdf/{arxiv_id}"


def fetch_excerpt(arxiv_id: str, max_chars: int) -> str | None:
    """Download a paper's PDF and return a results/conclusion-shaped excerpt,
    or None on any failure."""
    url = _pdf_url(arxiv_id)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Arxivore/0.2"})
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:
            pdf_bytes = response.read(_MAX_DOWNLOAD_BYTES + 1)
        if len(pdf_bytes) > _MAX_DOWNLOAD_BYTES:
            logger.warning("pdf too large, skipping arxiv_id=%s url=%s", arxiv_id, url)
            return None

        reader = PdfReader(BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not full_text.strip():
            return None

        match = _SECTION_HEADERS.search(full_text)
        excerpt = full_text[match.start() :] if match else full_text
        excerpt = excerpt.strip()[:max_chars]
        return excerpt or None
    except Exception:
        logger.exception("full-text fetch failed arxiv_id=%s url=%s", arxiv_id, url)
        return None
