"""MediGenius service exports, loaded lazily to avoid agent/service cycles."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AuthService": ("auth_service", "AuthService"),
    "auth_service": ("auth_service", "auth_service"),
    "ChatService": ("chat_service", "ChatService"),
    "chat_service": ("chat_service", "chat_service"),
    "DatabaseService": ("database_service", "DatabaseService"),
    "db_service": ("database_service", "db_service"),
    "ECGMonitorService": ("ecg_monitor_service", "ECGMonitorService"),
    "ecg_monitor_service": ("ecg_monitor_service", "ecg_monitor_service"),
    "ECGReportService": ("ecg_report_service", "ECGReportService"),
    "ecg_report_service": ("ecg_report_service", "ecg_report_service"),
    "RateLimitService": ("rate_limit_service", "RateLimitService"),
    "rate_limit_service": ("rate_limit_service", "rate_limit_service"),
    "RedisService": ("redis_service", "RedisService"),
    "redis_service": ("redis_service", "redis_service"),
    "SemanticCacheService": ("semantic_cache_service", "SemanticCacheService"),
    "semantic_cache_service": ("semantic_cache_service", "semantic_cache_service"),
    "TaskQueueService": ("task_queue_service", "TaskQueueService"),
    "task_queue_service": ("task_queue_service", "task_queue_service"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(f"app.services.{module_name}"), attribute)
    globals()[name] = value
    return value
