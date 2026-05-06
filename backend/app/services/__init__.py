"""
MediGenius — services/__init__.py
Exports service singletons.
"""

from app.services.auth_service import AuthService, auth_service
from app.services.chat_service import ChatService, chat_service
from app.services.database_service import DatabaseService, db_service
from app.services.ecg_monitor_service import ECGMonitorService, ecg_monitor_service
from app.services.ecg_report_service import ECGReportService, ecg_report_service
from app.services.rate_limit_service import RateLimitService, rate_limit_service
from app.services.redis_service import RedisService, redis_service
from app.services.semantic_cache_service import (
    SemanticCacheService,
    semantic_cache_service,
)
from app.services.task_queue_service import TaskQueueService, task_queue_service

__all__ = [
    "AuthService",
    "auth_service",
    "DatabaseService",
    "db_service",
    "ChatService",
    "chat_service",
    "ECGMonitorService",
    "ecg_monitor_service",
    "ECGReportService",
    "ecg_report_service",
    "RedisService",
    "redis_service",
    "RateLimitService",
    "rate_limit_service",
    "SemanticCacheService",
    "semantic_cache_service",
    "TaskQueueService",
    "task_queue_service",
]
