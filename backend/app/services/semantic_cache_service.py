"""
MediGenius — services/semantic_cache_service.py
Low-risk medical semantic cache keyed by structured meaning, not raw query text.
"""

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from app.core.config import (
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_MIN_ENTITY_COUNT,
    SEMANTIC_CACHE_PROMPT_VERSION,
    SEMANTIC_CACHE_RAG_VERSION,
    SEMANTIC_CACHE_TTL_SECONDS,
)
from app.services.redis_service import redis_service

ANSWER_PREFIX = "mg:semcache:answer:chat:v1"
FINGERPRINT_PREFIX = "mg:semcache:fingerprint:chat:v1"

ENTITY_ALIASES = {
    "高血压": ["高血压", "血压高", "血压偏高", "hypertension"],
    "低血压": ["低血压", "血压低", "hypotension"],
    "咖啡": ["咖啡", "coffee", "美式", "拿铁"],
    "茶": ["茶", "绿茶", "红茶"],
    "发热": ["发热", "发烧", "体温高", "fever"],
    "头痛": ["头痛", "头疼", "headache"],
    "咳嗽": ["咳嗽", "咳痰", "cough"],
    "睡眠": ["睡眠", "失眠", "睡不着", "insomnia"],
    "运动": ["运动", "跑步", "健身", "exercise"],
    "饮食": ["饮食", "吃饭", "食物", "diet"],
}

HIGH_RISK_KEYWORDS = [
    "胸痛",
    "呼吸困难",
    "昏迷",
    "意识模糊",
    "抽搐",
    "大出血",
    "晕厥",
    "剧烈头痛",
    "严重过敏",
    "自杀",
    "轻生",
]

PERSONALIZED_KEYWORDS = [
    "孕",
    "怀孕",
    "儿童",
    "婴儿",
    "老人",
    "老年",
    "剂量",
    "用量",
    "报告",
    "检查",
    "心电",
    "ecg",
    "病史",
    "正在吃",
    "服用",
]


@dataclass
class SemanticCacheLookup:
    eligible: bool
    fingerprint: str
    answer_key: str
    metadata: dict[str, Any]
    reason: str = ""


class SemanticCacheService:
    """Rule-based semantic cache that can be upgraded to Redis vector search later."""

    @staticmethod
    def _normalize_query(query: str) -> str:
        text = (query or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[，。！？、；：,.!?;:]+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_entities(normalized_query: str) -> list[str]:
        entities = []
        for entity, aliases in ENTITY_ALIASES.items():
            if any(alias.lower() in normalized_query for alias in aliases):
                entities.append(entity)
        return sorted(set(entities))

    @staticmethod
    def _question_type(normalized_query: str) -> str:
        if any(token in normalized_query for token in ["能不能", "可以", "可不可以", "能喝", "能吃"]):
            return "can_or_not"
        if any(token in normalized_query for token in ["为什么", "原因", "怎么回事"]):
            return "cause"
        if any(token in normalized_query for token in ["怎么办", "怎么处理", "如何缓解"]):
            return "management"
        if any(token in normalized_query for token in ["多少", "剂量", "用量"]):
            return "dosage"
        return "general_qa"

    @staticmethod
    def _contains_negation(normalized_query: str) -> bool:
        # Neutral yes/no question forms should not be treated as negative intent.
        neutral_patterns = (
            "能不能",
            "可不可以",
            "可以不可以",
            "能否",
            "是否可以",
            "是否能",
        )
        text = normalized_query
        for pattern in neutral_patterns:
            text = text.replace(pattern, "可以")
        return any(token in text for token in ["不能", "不要", "不可以", "禁忌", "避免"])

    @staticmethod
    def _risk_level(normalized_query: str) -> str:
        if any(keyword in normalized_query for keyword in HIGH_RISK_KEYWORDS):
            return "high"
        if any(keyword in normalized_query for keyword in PERSONALIZED_KEYWORDS):
            return "medium"
        if re.search(r"\b\d{1,3}\s*(岁|year|years)\b", normalized_query):
            return "medium"
        return "low"

    @staticmethod
    def _canonical_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def build_lookup(
        self,
        *,
        query: str,
        tenant_id: str,
        selected_department: Optional[str] = None,
    ) -> SemanticCacheLookup:
        if not SEMANTIC_CACHE_ENABLED:
            return SemanticCacheLookup(False, "", "", {}, "disabled")

        normalized = self._normalize_query(query)
        entities = self._extract_entities(normalized)
        risk_level = self._risk_level(normalized)
        question_type = self._question_type(normalized)
        metadata = {
            "tenant_id": tenant_id,
            "entities": entities,
            "intent": "medical_qa",
            "question_type": question_type,
            "negation": self._contains_negation(normalized),
            "risk_level": risk_level,
            "selected_department": selected_department or "",
            "prompt_version": SEMANTIC_CACHE_PROMPT_VERSION,
            "rag_collection_version": SEMANTIC_CACHE_RAG_VERSION,
        }

        if risk_level != "low":
            return SemanticCacheLookup(False, "", "", metadata, "risk_not_low")
        if question_type == "dosage":
            return SemanticCacheLookup(False, "", "", metadata, "dosage_question")
        if len(entities) < int(SEMANTIC_CACHE_MIN_ENTITY_COUNT):
            return SemanticCacheLookup(False, "", "", metadata, "not_enough_entities")

        fingerprint = hashlib.sha256(self._canonical_json(metadata).encode("utf-8")).hexdigest()
        answer_key = f"{ANSWER_PREFIX}:{fingerprint}"
        return SemanticCacheLookup(
            True,
            fingerprint,
            answer_key,
            metadata,
        )

    def get_answer(self, lookup: SemanticCacheLookup) -> Optional[dict[str, Any]]:
        if not lookup.eligible:
            return None
        payload = redis_service.get_json(lookup.answer_key)
        if not isinstance(payload, dict):
            return None
        payload["hit_count"] = int(payload.get("hit_count", 0)) + 1
        payload["cache_hit"] = True
        redis_service.set_json(
            lookup.answer_key,
            payload,
            ex=SEMANTIC_CACHE_TTL_SECONDS,
        )
        return payload

    def store_answer(
        self,
        lookup: SemanticCacheLookup,
        *,
        answer: str,
        source: str,
        flow_trace: list[str],
    ) -> None:
        if not lookup.eligible or not answer:
            return
        now = int(time.time())
        payload = {
            "semantic_id": str(uuid.uuid4()),
            "answer": answer,
            "canonical_question": " / ".join(lookup.metadata.get("entities", [])),
            "source": source,
            "flow_trace": flow_trace,
            "metadata": lookup.metadata,
            "created_at": now,
            "expires_at": now + int(SEMANTIC_CACHE_TTL_SECONDS),
            "hit_count": 0,
            "cache_hit": False,
        }
        redis_service.set_json(
            lookup.answer_key,
            payload,
            ex=SEMANTIC_CACHE_TTL_SECONDS,
        )
        redis_service.set_json(
            f"{FINGERPRINT_PREFIX}:{lookup.fingerprint}",
            {"answer_key": lookup.answer_key, "metadata": lookup.metadata},
            ex=SEMANTIC_CACHE_TTL_SECONDS,
        )


semantic_cache_service = SemanticCacheService()
