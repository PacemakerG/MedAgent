"""
MediGenius — services/task_queue_service.py
Local worker-pool task queue with Redis-compatible status storage.
"""

import asyncio
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional

from app.core.config import TASK_QUEUE_MAX_WORKERS, TASK_STATUS_TTL_SECONDS
from app.core.logging_config import logger
from app.services.redis_service import redis_service

JOB_PREFIX = "mg:job"


class TaskQueueService:
    """A process-local queue facade that can be replaced by Celery/RQ later."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(TASK_QUEUE_MAX_WORKERS)))

    @staticmethod
    def _job_key(job_id: str) -> str:
        return f"{JOB_PREFIX}:{job_id}"

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    def _set_status(self, job_id: str, payload: dict[str, Any]) -> None:
        existing = redis_service.get_json(self._job_key(job_id)) or {}
        existing.update(payload)
        existing["updated_at"] = self._now()
        redis_service.set_json(
            self._job_key(job_id),
            existing,
            ex=TASK_STATUS_TTL_SECONDS,
        )

    def create_job(
        self,
        *,
        task_type: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        redis_service.set_json(
            self._job_key(job_id),
            {
                "job_id": job_id,
                "status": "queued",
                "type": task_type,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "session_id": session_id,
                "metadata": metadata or {},
                "result": None,
                "error": None,
                "created_at": self._now(),
                "updated_at": self._now(),
            },
            ex=TASK_STATUS_TTL_SECONDS,
        )
        return job_id

    def submit(
        self,
        *,
        task_type: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        fn: Callable[..., Any],
        args: Optional[tuple] = None,
        kwargs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        job_id = self.create_job(
            task_type=task_type,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )
        self._executor.submit(
            self._run_job,
            job_id,
            fn,
            args or (),
            kwargs or {},
        )
        return job_id

    def submit_async(
        self,
        *,
        task_type: str,
        tenant_id: str,
        user_id: str,
        session_id: str,
        coroutine_factory: Callable[[], Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        def _runner():
            return asyncio.run(coroutine_factory())

        return self.submit(
            task_type=task_type,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            fn=_runner,
            metadata=metadata,
        )

    def _run_job(
        self,
        job_id: str,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict[str, Any],
    ) -> None:
        self._set_status(job_id, {"status": "running", "started_at": self._now()})
        try:
            result = fn(*args, **kwargs)
            self._set_status(
                job_id,
                {
                    "status": "succeeded",
                    "result": result,
                    "error": None,
                    "finished_at": self._now(),
                },
            )
        except Exception as exc:
            logger.exception("Task job failed job_id=%s: %s", job_id, exc)
            self._set_status(
                job_id,
                {
                    "status": "failed",
                    "result": None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=6),
                    "finished_at": self._now(),
                },
            )

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        payload = redis_service.get_json(self._job_key(job_id))
        return payload if isinstance(payload, dict) else None


task_queue_service = TaskQueueService()
