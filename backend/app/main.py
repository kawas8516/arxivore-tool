import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.search import router as search_router

_STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail fast on startup if required env vars are missing
    yield


app = FastAPI(title="Arxivore", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_allow_origins.split(",")],
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Content-Type"],
)

app.include_router(search_router, prefix="/api")

# Serve the single-page UI at / (html=True serves index.html for the root).
# Mounted last so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
