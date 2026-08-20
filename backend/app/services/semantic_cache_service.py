"""
MediGenius — services/semantic_cache_service.py
Redis Stack semantic cache with entity filtering and vector similarity search.
"""

import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.core.config import (
    SEMANTIC_CACHE_EMBEDDING_DIMENSION,
    SEMANTIC_CACHE_EMBEDDING_MODEL,
    SEMANTIC_CACHE_ENABLED,
    SEMANTIC_CACHE_INDEX_NAME,
    SEMANTIC_CACHE_KEY_PREFIX,
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SEMANTIC_CACHE_TTL_SECONDS,
)
from app.core.logging_config import logger
from app.services.redis_service import redis_service
from app.tools.llm_client import coerce_response_text, get_light_llm


# Common clinical aliases are normalized deterministically before the LLM
# fallback.  This keeps cache keys stable across abbreviations and colloquial
# wording while preserving answer-changing diseases and population qualifiers.
_CANONICAL_ENTITY_PATTERNS = (
    ("高血压", r"高血压|血压高"),
    ("低血压", r"低血压"),
    ("咖啡", r"咖啡"),
    ("妊娠期糖尿病", r"妊娠期?糖尿病"),
    ("糖尿病", r"糖尿病|高血糖|血糖高"),
    ("上呼吸道感染", r"感冒|上呼吸道感染"),
    ("咳嗽", r"咳嗽|咳个不停"),
    ("儿童", r"儿童|孩子|宝宝|婴儿|新生儿|婴幼儿|患儿"),
    ("成人", r"成人|成年人"),
    ("发热", r"发烧|发热|低热"),
    ("慢性肾病", r"慢性肾病|慢性肾脏病|(?<![A-Za-z])CKD(?![A-Za-z])"),
    ("白蛋白尿", r"尿蛋白|白蛋白尿"),
    ("尿路感染", r"尿路感染"),
    ("尿培养", r"尿培养"),
    ("眼底检查", r"眼底|视网膜检查"),
    ("近视", r"近视"),
    ("青光眼", r"青光眼"),
    ("耳鸣", r"耳鸣|耳朵响"),
    ("听力下降", r"听力下降|听不清"),
    ("眩晕", r"眩晕"),
    ("尿路结石", r"肾结石|泌尿系结石|尿路结石"),
    ("心力衰竭", r"心力衰竭|心衰"),
    ("饮水", r"喝水|饮水|饮水量"),
    ("慢性腰痛", r"慢性腰痛|长期下腰背疼|慢性下背痛|长期腰背痛"),
    ("急性腰椎骨折", r"急性腰椎骨折"),
    ("运动", r"运动|锻炼"),
    ("乙型肝炎", r"乙肝|乙型肝炎|(?<![A-Za-z])HBV(?![A-Za-z])"),
    ("丙型肝炎", r"丙肝|丙型肝炎|(?<![A-Za-z])HCV(?![A-Za-z])"),
    ("肝细胞癌", r"肝癌|肝细胞癌|(?<![A-Za-z])HCC(?![A-Za-z])"),
    ("孕期", r"孕期|怀孕|孕妇"),
    ("备孕男性", r"备孕男性"),
    ("叶酸", r"叶酸"),
    ("腹泻", r"腹泻|拉肚子"),
    ("脱水", r"脱水|缺水"),
    ("呕吐", r"呕吐"),
    ("脑膜炎", r"脑膜炎"),
    ("白内障", r"白内障|晶状体混浊"),
    ("视力下降", r"视力下降|看不清|视物模糊"),
    ("干眼症", r"干眼症|干眼"),
    ("安全聆听", r"安全使用耳机|戴耳机.*(?:噪声|听力损伤)|耳机.*(?:音量|听力)"),
    ("助听器", r"助听器"),
    ("疥疮", r"疥疮|疥螨感染"),
    ("湿疹", r"湿疹"),
    ("癫痫", r"癫痫"),
    ("偏头痛", r"偏头痛"),
    ("心血管风险评估", r"总体心血管风险|心血管.*不能只看.*血压|心血管管理.*血压"),
    ("心电图", r"心电图"),
    ("癌症", r"癌症|肿瘤"),
    ("早期诊断与筛查", r"早期诊断.*筛查|早诊.*筛查"),
    ("传染病", r"传染病"),
    ("低钙摄入", r"低钙|钙不足|钙摄入不足|饮食钙不足|膳食钙不足"),
    ("补钙", r"补钙|补充钙"),
    ("手术身份核查", r"手术.*(?:身份|姓名|术式)|手术室.*(?:姓名|术式)"),
    ("门诊取药", r"门诊取药"),
    ("听力筛查", r"听力筛查|检查听觉|听力检查"),
    (
        "慢性呼吸病",
        r"慢性呼吸病|哮喘|(?<![A-Za-z])COPD(?![A-Za-z])|慢性阻塞性肺疾病",
    ),
    ("戒烟", r"戒烟|停止吸烟"),
    ("ABCDE", r"(?<![A-Za-z])ABCDE(?![A-Za-z])|气道.*呼吸.*循环"),
    ("急危重症", r"急诊|危重患者"),
    ("常规体检", r"常规体检"),
)


@dataclass
class SemanticCacheLookup:
    eligible: bool
    entities: tuple[str, ...] = ()
    entity_filter: str = ""
    embedding: bytes = b""
    metadata: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class SemanticCacheService:
    """Global medical-answer cache backed by Redis Stack Search."""

    def __init__(self) -> None:
        self._embedding_model = None
        self._embedding_lock = threading.Lock()
        self._index_ready = False
        self._index_lock = threading.Lock()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except (TypeError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _normalize_entity(value: Any) -> str:
        entity = str(value or "").strip()
        entity = re.sub(r"[\s|]+", "", entity)
        entity = re.sub(
            r"^[，。！？、；：,.!?;:]+|[，。！？、；：,.!?;:]+$", "", entity
        )
        if re.fullmatch(r"[A-Za-z0-9_\-]+", entity):
            entity = entity.lower()
        return entity[:80]

    @staticmethod
    def _extract_known_entities(query: str) -> tuple[str, ...]:
        entities = {
            canonical
            for canonical, pattern in _CANONICAL_ENTITY_PATTERNS
            if re.search(pattern, query, flags=re.IGNORECASE)
        }
        if "妊娠期糖尿病" in entities:
            entities.discard("糖尿病")
        if "备孕男性" in entities:
            entities.discard("成人")
        return tuple(sorted(entities))

    def _extract_entities(self, query: str, *, user_id: str) -> tuple[str, ...]:
        known_entities = self._extract_known_entities(query)
        if known_entities:
            return known_entities

        llm = get_light_llm(user_id=user_id)
        if llm is None:
            return ()

        prompt = f"""你是医学实体抽取器。请从问题中抽取决定答案是否可复用的医学实体，并归一化为简短、标准的中文名称。

抽取范围：疾病、症状、药物、检查、治疗、饮食或行为对象、特殊人群。
归一化要求：同义词统一，例如“血压高”归一化为“高血压”；不要输出疑问词、程度副词或解释。
只返回严格 JSON，格式为：{{"entities":["实体1","实体2"]}}。

问题：{query}
"""
        try:
            response = llm.invoke(prompt)
            payload = self._extract_json_object(coerce_response_text(response))
        except Exception as exc:
            logger.warning("Semantic cache entity extraction failed: %s", exc)
            return ()

        raw_entities = payload.get("entities")
        if not isinstance(raw_entities, list):
            return ()
        normalized = {
            entity for item in raw_entities if (entity := self._normalize_entity(item))
        }
        return tuple(sorted(normalized))

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        with self._embedding_lock:
            if self._embedding_model is not None:
                return self._embedding_model
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(
                    SEMANTIC_CACHE_EMBEDDING_MODEL,
                    device="cpu",
                )
                logger.info(
                    "Semantic cache embedding model loaded (%s)",
                    SEMANTIC_CACHE_EMBEDDING_MODEL,
                )
            except Exception as exc:
                logger.error("Semantic cache embedding model unavailable: %s", exc)
                return None
        return self._embedding_model

    def _embed_query(self, query: str) -> bytes:
        model = self._get_embedding_model()
        if model is None:
            return b""
        try:
            vector = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
            array = np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            logger.warning("Semantic cache embedding failed: %s", exc)
            return b""
        if array.ndim != 1 or array.size != SEMANTIC_CACHE_EMBEDDING_DIMENSION:
            logger.error(
                "Semantic cache embedding dimension mismatch: expected=%s actual=%s",
                SEMANTIC_CACHE_EMBEDDING_DIMENSION,
                array.size,
            )
            return b""
        return array.tobytes()

    @staticmethod
    def _escape_tag_value(value: str) -> str:
        return re.sub(r"([\\,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/ ])", r"\\\1", value)

    def _ensure_index(self, client) -> bool:
        if self._index_ready:
            return True
        with self._index_lock:
            if self._index_ready:
                return True
            try:
                client.execute_command("FT.INFO", SEMANTIC_CACHE_INDEX_NAME)
            except Exception as exc:
                message = str(exc).lower()
                if "unknown command" in message:
                    logger.error(
                        "Redis Stack Search is unavailable; use redis/redis-stack-server"
                    )
                    return False
                missing_index_markers = (
                    "unknown index",
                    "no such index",
                    "index not found",
                    "search_index_not_found",
                )
                if not any(marker in message for marker in missing_index_markers):
                    logger.warning("Semantic cache index inspection failed: %s", exc)
                    return False
                try:
                    client.execute_command(
                        "FT.CREATE",
                        SEMANTIC_CACHE_INDEX_NAME,
                        "ON",
                        "HASH",
                        "PREFIX",
                        "1",
                        SEMANTIC_CACHE_KEY_PREFIX,
                        "SCHEMA",
                        "entities",
                        "TAG",
                        "SEPARATOR",
                        "|",
                        "embedding",
                        "VECTOR",
                        "HNSW",
                        "6",
                        "TYPE",
                        "FLOAT32",
                        "DIM",
                        str(SEMANTIC_CACHE_EMBEDDING_DIMENSION),
                        "DISTANCE_METRIC",
                        "COSINE",
                    )
                except Exception as create_exc:
                    logger.error("Semantic cache index creation failed: %s", create_exc)
                    return False
            self._index_ready = True
            return True

    def build_lookup(
        self,
        *,
        query: str,
        user_id: str = "anonymous",
    ) -> SemanticCacheLookup:
        if not SEMANTIC_CACHE_ENABLED:
            return SemanticCacheLookup(False, reason="disabled")
        if not redis_service.available():
            return SemanticCacheLookup(False, reason="redis_unavailable")

        entities = self._extract_entities(query, user_id=user_id)
        if not entities:
            return SemanticCacheLookup(False, reason="entities_unavailable")

        embedding_text = f"医学实体：{'；'.join(entities)}。问题：{query}"
        embedding = self._embed_query(embedding_text)
        if not embedding:
            return SemanticCacheLookup(
                False,
                entities=entities,
                metadata={"entities": list(entities)},
                reason="embedding_unavailable",
            )

        entity_filter = "__".join(entities)
        return SemanticCacheLookup(
            True,
            entities=entities,
            entity_filter=entity_filter,
            embedding=embedding,
            metadata={"entities": list(entities)},
        )

    @staticmethod
    def _decode(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value or "")

    def get_answer(self, lookup: SemanticCacheLookup) -> Optional[dict[str, Any]]:
        if not lookup.eligible:
            return None
        client = redis_service.client()
        if client is None or not self._ensure_index(client):
            return None

        escaped_entities = self._escape_tag_value(lookup.entity_filter)
        query = (
            f"(@entities:{{{escaped_entities}}})"
            "=>[KNN 1 @embedding $query_vector AS distance]"
        )
        try:
            result = client.execute_command(
                "FT.SEARCH",
                SEMANTIC_CACHE_INDEX_NAME,
                query,
                "PARAMS",
                "2",
                "query_vector",
                lookup.embedding,
                "SORTBY",
                "distance",
                "ASC",
                "RETURN",
                "2",
                "answer",
                "distance",
                "DIALECT",
                "2",
            )
        except Exception as exc:
            logger.warning("Semantic cache vector search failed: %s", exc)
            return None

        if not isinstance(result, (list, tuple)) or not result or int(result[0]) == 0:
            return None
        if len(result) < 3 or not isinstance(result[2], (list, tuple)):
            return None
        fields = result[2]
        row = {
            self._decode(fields[index]): fields[index + 1]
            for index in range(0, len(fields) - 1, 2)
        }
        try:
            distance = float(self._decode(row.get("distance")))
        except (TypeError, ValueError):
            return None
        similarity = 1.0 - distance
        if similarity < SEMANTIC_CACHE_SIMILARITY_THRESHOLD:
            return None

        answer = self._decode(row.get("answer"))
        if not answer:
            return None
        lookup.metadata["similarity"] = round(similarity, 6)
        return {
            "answer": answer,
            "source": "Semantic Cache",
            "flow_trace": ["semantic_cache"],
            "cache_hit": True,
            "similarity": similarity,
        }

    def store_answer(
        self,
        lookup: SemanticCacheLookup,
        *,
        answer: str,
        source: str = "",
        flow_trace: Optional[list[str]] = None,
    ) -> Optional[str]:
        del source, flow_trace
        if not lookup.eligible or not answer:
            return None
        client = redis_service.client()
        if client is None or not self._ensure_index(client):
            return None

        key = f"{SEMANTIC_CACHE_KEY_PREFIX}{uuid.uuid4()}"
        try:
            client.hset(
                key,
                mapping={
                    "entities": lookup.entity_filter,
                    "embedding": lookup.embedding,
                    "answer": answer,
                },
            )
            client.expire(key, int(SEMANTIC_CACHE_TTL_SECONDS))
            return key
        except Exception as exc:
            logger.warning("Semantic cache write failed: %s", exc)
            return None


semantic_cache_service = SemanticCacheService()
