"""
MediGenius — services/redis_service.py
Optional Redis client with in-memory fallbacks for local development and tests.
"""

import json
import time
from typing import Any, Optional

from app.core.config import REDIS_ENABLED, REDIS_SOCKET_TIMEOUT_SECONDS, REDIS_URL
from app.core.logging_config import logger


class RedisService:
    """Small Redis wrapper that degrades cleanly when Redis is disabled."""

    def __init__(self):
        self._client = None
        self._memory: dict[str, tuple[Any, Optional[float]]] = {}
        self._connect_attempted = False

    def client(self):
        if not REDIS_ENABLED:
            return None
        if self._client is not None:
            return self._client
        if self._connect_attempted:
            return None
        self._connect_attempted = True
        try:
            import redis

            client = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                socket_connect_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
            )
            client.ping()
            self._client = client
            logger.info("Redis connected at %s", REDIS_URL)
        except Exception as exc:
            logger.warning("Redis unavailable, using in-memory fallback: %s", exc)
            self._client = None
        return self._client

    def available(self) -> bool:
        return self.client() is not None

    def _purge_if_expired(self, key: str) -> None:
        item = self._memory.get(key)
        if not item:
            return
        _, expires_at = item
        if expires_at is not None and expires_at <= time.time():
            self._memory.pop(key, None)

    def get_json(self, key: str) -> Any:
        client = self.client()
        if client:
            raw = client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        self._purge_if_expired(key)
        item = self._memory.get(key)
        return item[0] if item else None

    def set_json(self, key: str, value: Any, ex: Optional[int] = None) -> None:
        client = self.client()
        if client:
            client.set(key, json.dumps(value, ensure_ascii=False), ex=ex)
            return
        expires_at = time.time() + ex if ex else None
        self._memory[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        client = self.client()
        if client:
            client.delete(key)
            return
        self._memory.pop(key, None)

    def incr(self, key: str, ex: Optional[int] = None) -> int:
        client = self.client()
        if client:
            value = int(client.incr(key))
            if value == 1 and ex:
                client.expire(key, ex)
            return value
        self._purge_if_expired(key)
        current = 0
        if key in self._memory:
            current = int(self._memory[key][0])
        current += 1
        expires_at = time.time() + ex if ex else self._memory.get(key, (None, None))[1]
        self._memory[key] = (current, expires_at)
        return current

    def set_nx(self, key: str, value: Any, ex: int) -> bool:
        client = self.client()
        if client:
            return bool(client.set(key, json.dumps(value, ensure_ascii=False), nx=True, ex=ex))
        self._purge_if_expired(key)
        if key in self._memory:
            return False
        self._memory[key] = (value, time.time() + ex)
        return True


redis_service = RedisService()
