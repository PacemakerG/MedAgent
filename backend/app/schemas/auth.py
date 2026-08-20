"""
MediGenius — schemas/auth.py
Pydantic schemas for password-based web login.
"""

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=256)


class AuthStatusResponse(BaseModel):
    logged_in: bool
    user_id: str
    session_id: str
    success: bool = True
    access_token: str | None = None
    token_type: str = "Bearer"
    expires_at: int | None = None
    created: bool = False
