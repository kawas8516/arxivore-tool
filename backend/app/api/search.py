import logging

from fastapi import APIRouter, HTTPException

from app.models import SearchRequest, SearchResponse
from app.service import run_pipeline, PipelineError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return run_pipeline(request.topic)
    except PipelineError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc
