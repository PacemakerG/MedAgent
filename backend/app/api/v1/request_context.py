"""
MediGenius — api/v1/request_context.py
Resolve user/session identity from access tokens, headers, and cookie sessions.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from fastapi import Request

from app.core.config import AUTH_TRUST_IDENTITY_HEADERS
from app.services.auth_service import auth_service

DEFAULT_USER_ID = "anonymous"


def _sanitize_id(value: str | None, default: str, max_len: int = 128) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        return default
    text = (value or "").strip()
    if not text:
        return default
    safe = re.sub(r"[^a-zA-Z0-9_.:@/-]", "_", text)
    safe = safe.strip("._-") or default
    return safe[:max_len]


def _get_query_param(request: Request, key: str) -> str | None:
    params = getattr(request, "query_params", None)
    if params is None:
        return None
    try:
        value = params.get(key)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def _get_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization") or ""
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


@dataclass(frozen=True)
class RequestContext:
    user_id: str
    session_id: str


def get_request_context(request: Request) -> RequestContext:
    token_payload = auth_service.verify_access_token(_get_bearer_token(request))
    if token_payload:
        user_source = token_payload["user_id"]
        session_source = (
            request.headers.get("X-Session-ID")
            or _get_query_param(request, "session_id")
            or request.session.get("session_id")
            or token_payload["session_id"]
        )
    else:
        trusted_user = request.headers.get("X-User-ID") or _get_query_param(
            request,
            "user_id",
        )
        user_source = request.session.get("user_id")
        if AUTH_TRUST_IDENTITY_HEADERS:
            user_source = trusted_user or user_source
        session_source = (
            request.headers.get("X-Session-ID")
            or _get_query_param(request, "session_id")
            or request.session.get("session_id")
        )

    user_id = _sanitize_id(user_source, DEFAULT_USER_ID)

    if not session_source:
        session_source = str(uuid.uuid4())
    session_id = _sanitize_id(str(session_source), str(uuid.uuid4()), max_len=255)

    # Keep cookie session aligned for browser-only clients without custom headers.
    request.session["user_id"] = user_id
    request.session["session_id"] = session_id

    return RequestContext(
        user_id=user_id,
        session_id=session_id,
    )
