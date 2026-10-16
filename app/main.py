from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import classify, extract, health, signature
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.runtime import get_pipeline, purge_old_pages

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the shared pipeline once (classifier, OCR engine, detectors) and drop stale page images."""
    purge_old_pages(force=True)
    get_pipeline()
    logger.info("ai_service_ready", env=settings.app_env)
    yield


app = FastAPI(
    title="ai-service",
    description=(
        "Document understanding service (OCR, classification, extraction, "
        "signature presence detection) untuk 3PL Invoice Reconciliation Engine. "
        "Stateless — tidak menyimpan data bisnis atau membuat keputusan finansial."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(health.router)
app.include_router(classify.router)
app.include_router(extract.router)
app.include_router(signature.router)


@app.get("/")
def root() -> dict:
    return {"service": "ai-service", "env": settings.app_env, "status": "ok"}
