"""
MediGenius — api/v1/endpoints/chat.py
Chat-related endpoints: /chat, /clear, /new-chat, /welcome.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.responses import StreamingResponse

from app.api.v1.request_context import RequestContext, get_request_context
from app.core.config import CHAT_RATE_LIMIT_PER_MINUTE, TASK_QUEUE_ENABLED
from app.core.logging_config import logger
from app.schemas.chat import (
    ChatJobResponse,
    ChatRequest,
    ChatResponse,
    JobStatusResponse,
    WelcomeRequest,
    WelcomeResponse,
)
from app.services.chat_service import chat_service
from app.services.greeting_service import greeting_service
from app.services.rate_limit_service import rate_limit_service
from app.services.task_queue_service import task_queue_service

router = APIRouter(tags=["Chat"])


def _get_session_id(request: Request) -> str:
    """Backward-compatible session accessor used by tests."""
    return get_request_context(request).session_id


def _get_request_context(request: Request) -> RequestContext:
    return get_request_context(request)


def _to_sse_frame(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _client_ip(req: Request) -> str:
    forwarded = req.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return req.client.host if req.client else "unknown"


def _enforce_chat_rate_limit(req: Request, ctx: RequestContext) -> None:
    identity = f"{ctx.user_id}:{_client_ip(req)}"
    allowed, _remaining = rate_limit_service.allow(
        scope="chat",
        identity=identity,
        limit=CHAT_RATE_LIMIT_PER_MINUTE,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many chat requests")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Process a user message through the agentic pipeline."""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    ctx = _get_request_context(req)
    _enforce_chat_rate_limit(req, ctx)
    return await chat_service.process_message(
        ctx.session_id,
        request.message,
        user_id=ctx.user_id,
        selected_department=request.selected_department,
    )


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, req: Request):
    """Process chat with SSE push events: start -> delta* -> done/error."""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    ctx = _get_request_context(req)
    _enforce_chat_rate_limit(req, ctx)

    async def _event_generator():
        yield _to_sse_frame(
            "start",
            {
                "success": True,
                "session_id": ctx.session_id,
            },
        )
        try:
            async for event in chat_service.process_message_stream(
                ctx.session_id,
                request.message,
                user_id=ctx.user_id,
                selected_department=request.selected_department,
            ):
                event_name = event.get("event", "message")
                payload = {k: v for k, v in event.items() if k != "event"}
                yield _to_sse_frame(event_name, payload)
        except Exception as exc:
            logger.exception("chat stream failed: %s", exc)
            yield _to_sse_frame(
                "error",
                {
                    "success": False,
                    "message": str(exc),
                },
            )

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/jobs", response_model=ChatJobResponse)
async def chat_job_endpoint(request: ChatRequest, req: Request):
    """Queue a chat request and return a job ID for polling."""
    if not TASK_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="Task queue disabled")
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    ctx = _get_request_context(req)
    _enforce_chat_rate_limit(req, ctx)

    async def _run_chat():
        return await chat_service.process_message(
            ctx.session_id,
            request.message,
            user_id=ctx.user_id,
            selected_department=request.selected_department,
        )

    job_id = task_queue_service.submit_async(
        task_type="chat",
        user_id=ctx.user_id,
        session_id=ctx.session_id,
        coroutine_factory=_run_chat,
        metadata={"selected_department": request.selected_department},
    )
    return ChatJobResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_endpoint(job_id: str, req: Request):
    """Return queued task status."""
    ctx = _get_request_context(req)
    payload = task_queue_service.get_job(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found")
    if payload.get("user_id") != ctx.user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**payload, success=True)


@router.post("/clear")
async def clear_endpoint(req: Request):
    """Clear the in-memory conversation state for the current session."""
    ctx = _get_request_context(req)
    chat_service.clear_conversation(
        ctx.session_id,
        user_id=ctx.user_id,
    )
    return {"message": "Conversation cleared", "success": True}


@router.post("/new-chat")
async def new_chat_endpoint(req: Request):
    """Create a new chat session with a fresh session ID."""
    _ = _get_request_context(req)
    new_id = str(uuid.uuid4())
    req.session["session_id"] = new_id
    return {"message": "New chat created", "session_id": new_id, "success": True}


@router.post("/welcome", response_model=WelcomeResponse)
async def welcome_endpoint(request: WelcomeRequest, req: Request):
    """Generate a proactive welcome message for the current session."""
    ctx = _get_request_context(req)
    return greeting_service.generate_greeting(
        ctx.session_id,
        latitude=request.latitude,
        longitude=request.longitude,
        timezone_name=request.timezone,
        locale=request.locale,
        user_id=ctx.user_id,
    )
