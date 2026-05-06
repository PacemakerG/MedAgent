"""
MediGenius — api/v1/endpoints/auth.py
Password-based auth endpoints for browser login state.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request

from app.api.v1.request_context import (
    DEFAULT_TENANT_ID,
    DEFAULT_USER_ID,
    get_request_context,
)
from app.core.config import AUTH_LOGIN_RATE_LIMIT_PER_MINUTE
from app.schemas.auth import AuthStatusResponse, LoginRequest
from app.services.auth_service import auth_service
from app.services.rate_limit_service import rate_limit_service

router = APIRouter(prefix="/auth", tags=["Auth"])


def _client_ip(req: Request) -> str:
    forwarded = req.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return req.client.host if req.client else "unknown"


@router.get("/me", response_model=AuthStatusResponse)
async def auth_me_endpoint(req: Request):
    ctx = get_request_context(req)
    return AuthStatusResponse(
        logged_in=ctx.user_id != DEFAULT_USER_ID,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        success=True,
    )


@router.post("/login", response_model=AuthStatusResponse)
async def auth_login_endpoint(payload: LoginRequest, req: Request):
    identifier = f"{_client_ip(req)}:{payload.tenant_id or DEFAULT_TENANT_ID}:{payload.user_id}"
    allowed, _remaining = rate_limit_service.allow(
        scope="auth_login",
        identity=identifier,
        limit=AUTH_LOGIN_RATE_LIMIT_PER_MINUTE,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many login attempts")

    session_id = req.session.get("session_id") or str(uuid.uuid4())
    result = auth_service.authenticate(
        tenant_id=payload.tenant_id or DEFAULT_TENANT_ID,
        user_id=payload.user_id,
        password=payload.password,
        session_id=session_id,
    )
    if not result:
        raise HTTPException(status_code=401, detail="Invalid user ID or password")

    req.session["tenant_id"] = result.tenant_id
    req.session["user_id"] = result.user_id
    req.session["session_id"] = result.session_id
    return AuthStatusResponse(
        logged_in=True,
        tenant_id=result.tenant_id,
        user_id=result.user_id,
        session_id=result.session_id,
        success=True,
        access_token=result.access_token,
        token_type=result.token_type,
        expires_at=result.expires_at,
        created=result.created,
    )


@router.post("/logout", response_model=AuthStatusResponse)
async def auth_logout_endpoint(req: Request):
    req.session["user_id"] = DEFAULT_USER_ID
    req.session.pop("access_token", None)
    ctx = get_request_context(req)
    return AuthStatusResponse(
        logged_in=False,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        success=True,
    )
