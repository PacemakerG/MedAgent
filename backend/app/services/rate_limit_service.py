"""
MediGenius — services/rate_limit_service.py
Fixed-window rate limiting backed by Redis or the local fallback store.
"""

import time

from app.services.redis_service import redis_service


class RateLimitService:
    """Redis-compatible fixed-window counter."""

    @staticmethod
    def _window_key(scope: str, identity: str, window_seconds: int) -> str:
        window = int(time.time() // window_seconds)
        safe_identity = (identity or "anonymous").replace(" ", "_")[:160]
        return f"mg:rate:{scope}:{safe_identity}:{window_seconds}:{window}"

    def allow(
        self,
        *,
        scope: str,
        identity: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        key = self._window_key(scope, identity, window_seconds)
        count = redis_service.incr(key, ex=window_seconds + 5)
        remaining = max(0, int(limit) - count)
        return count <= int(limit), remaining


rate_limit_service = RateLimitService()
