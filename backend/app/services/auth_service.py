"""
MediGenius — services/auth_service.py
Password hashing and signed access-token helpers.
"""

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import (
    AUTH_ACCESS_TOKEN_TTL_SECONDS,
    AUTH_AUTO_CREATE_USERS,
    AUTH_PASSWORD_HASH_ITERATIONS,
    AUTH_TOKEN_SECRET,
)
from app.core.logging_config import logger
from app.services.database_service import db_service

PASSWORD_HASH_PREFIX = "pbkdf2_sha256"


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _clean_identity(value: str, default: str = "") -> str:
    text = (value or "").strip()
    if not text:
        return default
    cleaned = re.sub(r"[^a-zA-Z0-9_.:@/-]", "_", text).strip("._-")
    return (cleaned or default)[:128]


@dataclass
class AuthResult:
    user_id: str
    session_id: str
    access_token: str
    token_type: str
    expires_at: int
    created: bool = False


class AuthService:
    """User-scoped authentication service."""

    def __init__(self):
        self._secret = AUTH_TOKEN_SECRET or secrets.token_urlsafe(48)
        if not AUTH_TOKEN_SECRET:
            logger.warning(
                "AUTH_TOKEN_SECRET/SESSION_SECRET_KEY is not configured; "
                "using a process-local signing secret."
            )

    @staticmethod
    def normalize_user_id(value: str | None) -> str:
        return _clean_identity(value or "", "")

    def hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(24)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            AUTH_PASSWORD_HASH_ITERATIONS,
        )
        return "$".join(
            [
                PASSWORD_HASH_PREFIX,
                str(AUTH_PASSWORD_HASH_ITERATIONS),
                _b64encode(salt),
                _b64encode(digest),
            ]
        )

    def verify_password(self, password: str, password_hash: str) -> bool:
        try:
            algorithm, iterations_raw, salt_raw, digest_raw = password_hash.split("$", 3)
            if algorithm != PASSWORD_HASH_PREFIX:
                return False
            iterations = int(iterations_raw)
            salt = _b64decode(salt_raw)
            expected = _b64decode(digest_raw)
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    def _sign(self, payload_segment: str) -> str:
        signature = hmac.new(
            self._secret.encode("utf-8"),
            payload_segment.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return _b64encode(signature)

    def create_access_token(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> tuple[str, int]:
        expires_at = int(time.time()) + int(AUTH_ACCESS_TOKEN_TTL_SECONDS)
        payload = {
            "typ": "access",
            "user_id": user_id,
            "session_id": session_id,
            "iat": int(time.time()),
            "exp": expires_at,
            "jti": secrets.token_urlsafe(18),
        }
        payload_segment = _b64encode(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        token = f"{payload_segment}.{self._sign(payload_segment)}"
        return token, expires_at

    def verify_access_token(self, token: str | None) -> Optional[dict[str, Any]]:
        if not token or "." not in token:
            return None
        try:
            payload_segment, signature = token.rsplit(".", 1)
            expected = self._sign(payload_segment)
            if not hmac.compare_digest(signature, expected):
                return None
            payload = json.loads(_b64decode(payload_segment).decode("utf-8"))
            if payload.get("typ") != "access":
                return None
            if int(payload.get("exp", 0)) < int(time.time()):
                return None
            user_id = self.normalize_user_id(payload.get("user_id"))
            session_id = _clean_identity(payload.get("session_id", ""), "")
            if not user_id or not session_id:
                return None
            return {
                "user_id": user_id,
                "session_id": session_id,
                "expires_at": int(payload["exp"]),
                "jti": payload.get("jti", ""),
            }
        except Exception:
            return None

    def authenticate(
        self,
        *,
        user_id: str,
        password: str,
        session_id: str,
    ) -> Optional[AuthResult]:
        user_id = self.normalize_user_id(user_id)
        if not user_id or not password:
            return None

        try:
            user = db_service.get_user(user_id)
        except Exception as exc:
            logger.warning("Auth: user lookup failed, attempting DB init: %s", exc)
            db_service.ensure_user_table()
            user = db_service.get_user(user_id)
        created = False
        if user is None:
            if not AUTH_AUTO_CREATE_USERS:
                return None
            try:
                db_service.create_user(
                    user_id=user_id,
                    password_hash=self.hash_password(password),
                )
                created = True
                logger.info("Auth: auto-created user=%s", user_id)
            except Exception as exc:
                logger.warning("Auth: auto-create race or failure, retrying lookup: %s", exc)
                user = db_service.get_user(user_id)
                if user is None or not self.verify_password(password, user.password_hash):
                    return None
        elif not user.is_active:
            return None
        elif not self.verify_password(password, user.password_hash):
            return None

        db_service.update_user_last_login(user_id)
        access_token, expires_at = self.create_access_token(
            user_id=user_id,
            session_id=session_id,
        )
        return AuthResult(
            user_id=user_id,
            session_id=session_id,
            access_token=access_token,
            token_type="Bearer",
            expires_at=expires_at,
            created=created,
        )


auth_service = AuthService()
