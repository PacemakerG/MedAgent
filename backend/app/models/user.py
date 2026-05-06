"""
MediGenius — models/user.py
SQLAlchemy ORM model for application users.
"""

from datetime import datetime
from typing import Dict

from sqlalchemy import Boolean, Column, DateTime, Integer, String, UniqueConstraint

from app.models.message import Base


class User(Base):
    """Tenant-scoped user identity with a non-reversible password hash."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "user_id", name="uq_users_tenant_user"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(128), nullable=False, default="default", index=True)
    user_id = Column(String(128), nullable=False, index=True)
    password_hash = Column(String(512), nullable=False)
    role = Column(String(64), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }
