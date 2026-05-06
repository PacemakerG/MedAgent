"""
MediGenius — api/v1/endpoints/health.py
Health check endpoint.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.config import REDIS_ENABLED
from app.services.database_service import db_service
from app.services.redis_service import redis_service

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Returns service health status."""
    return {"status": "healthy", "service": "MediGenius Backend v2"}


@router.get("/healthz")
async def healthz():
    """Cheap liveness probe."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness probe for core backing services."""
    checks = {"database": "unknown", "redis": "disabled" if not REDIS_ENABLED else "unknown"}
    try:
        with db_service.get_session() as session:
            session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"failed: {exc}"

    if REDIS_ENABLED:
        checks["redis"] = "ok" if redis_service.available() else "failed"

    if checks["database"] != "ok":
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}
