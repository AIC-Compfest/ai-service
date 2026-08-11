from fastapi import FastAPI

from app.api import classify, extract, health, signature
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="ai-service",
    description=(
        "Document understanding service (OCR, classification, extraction, "
        "signature presence detection) untuk 3PL Invoice Reconciliation Engine. "
        "Stateless — tidak menyimpan data bisnis atau membuat keputusan finansial."
    ),
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(classify.router)
app.include_router(extract.router)
app.include_router(signature.router)


@app.get("/")
def root() -> dict:
    return {"service": "ai-service", "env": settings.app_env, "status": "ok"}
