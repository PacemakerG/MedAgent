"""Run independent and cumulative RAG experiments on dataset_150.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable

import httpx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agents.query_rewriter import QueryRewriterAgent  # noqa: E402
from app.core.config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    ES_HOST,
    ES_PASSWORD,
    ES_USERNAME,
    ES_VERIFY_CERTS,
    KNOWLEDGE_ROOT_DIR,
    LLM_MODEL,
    RERANKER_MODEL_NAME,
)
from app.core.medical_taxonomy import (  # noqa: E402
    GENERAL_MEDICAL_DEPARTMENT,
    normalize_department_code,
)
from app.core.state import initialize_conversation_state  # noqa: E402
from app.tools.llm_client import coerce_response_text, get_llm  # noqa: E402

DEFAULT_DATASET = BACKEND_ROOT / "data" / "eval" / "rag" / "dataset_150.json"
DEFAULT_OUTPUT = BACKEND_ROOT / "data" / "eval" / "rag" / "result.json"
DEFAULT_AUDIT = BACKEND_ROOT / "data" / "eval" / "rag" / "codex_audit_40.json"
RRF_K = 60
REWRITE_CACHE_VERSION = "english-pdf-v1"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    department: str
    text: str
    parent_text: str

    @property
    def evidence_key(self) -> tuple[str, int]:
        return self.source, self.page


@dataclass(frozen=True)
class Variant:
    name: str
    change: str
    chunk_strategy: str = "fixed"
    parent_child: bool = False
    cleaned: bool = False
    query_rewrite: bool = False
    hybrid_es: bool = False
    reranker: bool = False


INDEPENDENT_VARIANTS = (
    Variant("B0", "固定长度分块 + 单路向量召回"),
    Variant("B1", "B0 + 语义边界分块", chunk_strategy="semantic"),
    Variant("B2", "B0 + 父子索引", parent_child=True),
    Variant("B3", "B0 + 数据清洗", cleaned=True),
    Variant("B4", "B0 + Query Rewrite", query_rewrite=True),
    Variant("B5", "B0 + ES/向量并行召回 + RRF", hybrid_es=True),
    Variant("B6", "B0 + Reranker", reranker=True),
)

CUMULATIVE_VARIANTS = (
    Variant("C0", "固定长度分块 + 单路向量召回"),
    Variant("C1", "C0 + Reranker", reranker=True),
    Variant(
        "C2",
        "C1 + ES/向量并行召回 + RRF",
        hybrid_es=True,
        reranker=True,
    ),
    Variant(
        "C3",
        "C2 + Query Rewrite",
        query_rewrite=True,
        hybrid_es=True,
        reranker=True,
    ),
    Variant(
        "C4",
        "C3 + 数据清洗",
        cleaned=True,
        query_rewrite=True,
        hybrid_es=True,
        reranker=True,
    ),
    Variant(
        "C5",
        "C4 + 语义边界分块",
        chunk_strategy="semantic",
        cleaned=True,
        query_rewrite=True,
        hybrid_es=True,
        reranker=True,
    ),
    Variant(
        "C6",
        "C5 + 父子索引",
        chunk_strategy="semantic",
        parent_child=True,
        cleaned=True,
        query_rewrite=True,
        hybrid_es=True,
        reranker=True,
    ),
)


def local_model_path(model_name: str) -> str:
    """Use an existing Hugging Face snapshot without a redundant network probe."""
    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if len(samples) != 150:
        raise ValueError(f"RAG dataset must contain 150 samples, got {len(samples)}")
    return payload


def clean_ocr_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fixed_chunks(text: str, *, size: int, overlap: int) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    step = max(1, size - overlap)
    return [
        normalized[start : start + size] for start in range(0, len(normalized), step)
    ]


def semantic_chunks(text: str, *, target_size: int = 360) -> list[str]:
    normalized = " ".join((text or "").split())
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？.!?;；])\s+", normalized)
        if item.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_size:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or ([normalized] if normalized else [])


def _department_for_pdf(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts[:-1]:
        department = normalize_department_code(part)
        if department:
            return department
    return GENERAL_MEDICAL_DEPARTMENT


def build_corpus(knowledge_root: Path) -> list[dict[str, Any]]:
    """Build the retrieval corpus from every non-empty page in the PDF library."""
    from pypdf import PdfReader

    corpus = []
    for pdf_path in sorted(knowledge_root.rglob("*.pdf")):
        department = _department_for_pdf(pdf_path, knowledge_root)
        for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < 20:
                continue
            corpus.append(
                {
                    "source": pdf_path.name,
                    "page": page_number,
                    "department": department,
                    "text": text,
                }
            )
    if not corpus:
        raise ValueError(f"No PDF page text found under {knowledge_root}")
    return corpus


def build_chunks(corpus: list[dict[str, Any]], variant: Variant) -> list[Chunk]:
    chunks: list[Chunk] = []
    signature = corpus_signature(variant)
    for parent_index, page in enumerate(corpus):
        parent_text = clean_ocr_text(page["text"]) if variant.cleaned else page["text"]
        if variant.parent_child:
            if variant.chunk_strategy == "semantic":
                parts = semantic_chunks(parent_text, target_size=190)
            else:
                parts = fixed_chunks(parent_text, size=190, overlap=55)
        elif variant.chunk_strategy == "semantic":
            parts = semantic_chunks(parent_text)
        else:
            parts = fixed_chunks(parent_text, size=320, overlap=80)
        for child_index, text in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{signature}-{parent_index:05d}-{child_index:03d}",
                    source=page["source"],
                    page=page["page"],
                    department=page["department"],
                    text=text,
                    parent_text=parent_text,
                )
            )
    if not chunks:
        raise ValueError(f"Variant {variant.name} produced no chunks")
    return chunks


def corpus_signature(variant: Variant) -> str:
    return "-".join(
        (
            variant.chunk_strategy,
            "parent" if variant.parent_child else "flat",
            "clean" if variant.cleaned else "raw",
        )
    )


def variant_configuration(variant: Variant) -> tuple[Any, ...]:
    return (
        variant.chunk_strategy,
        variant.parent_child,
        variant.cleaned,
        variant.query_rewrite,
        variant.hybrid_es,
        variant.reranker,
    )


class VectorIndex:
    def __init__(
        self,
        model: SentenceTransformer,
        chunks: list[Chunk],
        *,
        matrix: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.chunks = chunks
        if matrix is None:
            matrix = model.encode(
                [chunk.text for chunk in chunks],
                batch_size=128,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        self.matrix = np.asarray(matrix, dtype=np.float32)
        if len(self.matrix) != len(chunks):
            raise ValueError("Cached embedding matrix does not match chunk count")

    def search(self, query: str, *, department: str, top_k: int) -> list[Chunk]:
        query_vector = np.asarray(
            self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        if department == "general_medical":
            allowed = np.arange(len(self.chunks))
        else:
            allowed = np.asarray(
                [
                    index
                    for index, chunk in enumerate(self.chunks)
                    if chunk.department == department
                ],
                dtype=np.int64,
            )
        if not len(allowed):
            return []
        scores = self.matrix[allowed] @ query_vector
        ranked = allowed[np.argsort(-scores)[:top_k]]
        return [self.chunks[int(index)] for index in ranked]


def corpus_fingerprint(corpus: list[dict[str, Any]], model_name: str) -> str:
    digest = hashlib.sha1(model_name.encode("utf-8"))
    for page in corpus:
        text = str(page["text"])
        digest.update(
            f"{page['source']}|{page['page']}|{page['department']}|{len(text)}|".encode(
                "utf-8"
            )
        )
        digest.update(text[:128].encode("utf-8", errors="ignore"))
        digest.update(text[-128:].encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


class ElasticsearchIndex:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.index_name = ""
        self._loaded_signature = ""
        self.auth = (ES_USERNAME, ES_PASSWORD or "") if ES_USERNAME else None

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=30.0,
            verify=ES_VERIFY_CERTS,
            auth=self.auth,
        )

    def assert_available(self) -> None:
        try:
            with self._client() as client:
                response = client.get("/")
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Elasticsearch is required for B5/C4-C6: {exc}"
            ) from exc

    def rebuild(self, chunks: list[Chunk], *, signature: str) -> None:
        if self._loaded_signature == signature:
            return
        safe_signature = re.sub(r"[^a-z0-9-]+", "-", signature.lower())
        self.index_name = f"medagent-rag-eval-{safe_signature}"
        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "department": {"type": "keyword"},
                }
            },
        }
        with self._client() as client:
            client.delete(f"/{self.index_name}")
            created = client.put(f"/{self.index_name}", json=mapping)
            created.raise_for_status()
            lines = []
            for chunk in chunks:
                lines.append(
                    json.dumps(
                        {"index": {"_index": self.index_name, "_id": chunk.chunk_id}}
                    )
                )
                lines.append(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "content": chunk.text,
                            "department": chunk.department,
                        },
                        ensure_ascii=False,
                    )
                )
            response = client.post(
                "/_bulk",
                content="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
            response.raise_for_status()
            client.post(f"/{self.index_name}/_refresh").raise_for_status()
        self._loaded_signature = signature

    def search(
        self,
        query: str,
        *,
        department: str,
        chunks_by_id: dict[str, Chunk],
        top_k: int,
    ) -> list[Chunk]:
        filters = []
        if department != "general_medical":
            filters.append({"term": {"department": department}})
        payload = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"content": {"query": query}}}],
                    "filter": filters,
                }
            },
        }
        with self._client() as client:
            response = client.post(f"/{self.index_name}/_search", json=payload)
            response.raise_for_status()
        hits = (response.json().get("hits") or {}).get("hits") or []
        return [
            chunks_by_id[item["_id"]]
            for item in hits
            if item.get("_id") in chunks_by_id
        ]


def rrf_merge(rankings: Iterable[list[Chunk]], *, top_k: int) -> list[Chunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )
            chunks[chunk.chunk_id] = chunk
    ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [chunks[chunk_id] for chunk_id in ranked_ids[:top_k]]


def rewrite_queries(question: str, department: str) -> tuple[list[str], str, float]:
    started = perf_counter()
    state = initialize_conversation_state()
    state.update(
        {
            "question": question,
            "domain": "medical",
            "use_rag": True,
            "primary_department": department,
            "department_candidates": [{"name": department, "score": 1.0}],
            "user_id": "rag-eval",
        }
    )
    result = QueryRewriterAgent(state)
    queries = (
        result.get("department_multi_queries", {}).get(department)
        or result.get("retrieval_queries")
        or [question]
    )
    return (
        [str(item) for item in queries if str(item).strip()],
        str(result.get("rewrite_reason") or ""),
        (perf_counter() - started) * 1000.0,
    )


def build_rewrite_cache(
    samples: list[dict[str, Any]],
    *,
    workers: int,
    cache_dir: Path,
) -> dict[tuple[str, str], tuple[list[str], str, float]]:
    keys = [(str(sample["question"]), str(sample["department"])) for sample in samples]

    def run(key: tuple[str, str]) -> tuple[list[str], str, float]:
        digest = hashlib.sha256(LLM_MODEL.encode("utf-8"))
        digest.update(REWRITE_CACHE_VERSION.encode("utf-8"))
        digest.update(key[0].encode("utf-8"))
        digest.update(key[1].encode("utf-8"))
        cache_path = cache_dir / f"{digest.hexdigest()}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                queries = [str(item) for item in cached["queries"] if str(item).strip()]
                if queries:
                    return queries, str(cached.get("reason") or ""), float(
                        cached["elapsed_ms"]
                    )
            except (KeyError, OSError, TypeError, ValueError):
                pass
        value = rewrite_queries(*key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "queries": value[0],
                    "reason": value[1],
                    "elapsed_ms": value[2],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return value

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        values = list(pool.map(run, keys))
    return dict(zip(keys, values))


def dedupe_parent_results(chunks: list[Chunk], *, top_k: int) -> list[Chunk]:
    results = []
    seen = set()
    for chunk in chunks:
        if chunk.evidence_key in seen:
            continue
        seen.add(chunk.evidence_key)
        results.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
                department=chunk.department,
                text=chunk.parent_text,
                parent_text=chunk.parent_text,
            )
        )
        if len(results) >= top_k:
            break
    return results


def metric_for_sample(sample: dict[str, Any], results: list[Chunk]) -> dict[str, float]:
    gold = {
        (str(item["source"]), int(item["page"]))
        for item in sample.get("gold_evidence") or []
    }
    top_five = results[:5]
    hit_at_1 = float(bool(top_five and top_five[0].evidence_key in gold))
    retrieved_gold = {
        chunk.evidence_key for chunk in top_five if chunk.evidence_key in gold
    }
    recall_at_5 = len(retrieved_gold) / max(1, len(gold))
    reciprocal_rank = 0.0
    for rank, chunk in enumerate(top_five, start=1):
        if chunk.evidence_key in gold:
            reciprocal_rank = 1.0 / rank
            break
    return {"hit_at_1": hit_at_1, "recall_at_5": recall_at_5, "mrr": reciprocal_rank}


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def score_faithfulness(question: str, contexts: list[Chunk], answer: str) -> float:
    llm = get_llm(user_id="rag-eval")
    if llm is None:
        raise RuntimeError(
            "OPENAI_API_KEY is required for answer faithfulness evaluation"
        )
    context = "\n\n".join(chunk.text for chunk in contexts[:5])
    prompt = f"""答案忠实度评测。请严格执行：
1. 将回答拆成可以独立核对的医学结论。
2. 逐条判断结论能否被检索证据直接支持。
3. 只能使用给定证据，不能使用自己的医学知识。
4. 忠实度 = 有证据支持的结论数 ÷ 全部可核对结论数 × 100。
只返回 JSON：{{"total_claims":整数,"supported_claims":整数,"score":0到100的数字}}。

用户问题：{question}
检索证据：{context}
待评回答：{answer}
"""
    payload = _parse_json_object(coerce_response_text(llm.invoke(prompt)))
    score = float(payload.get("score"))
    if not 0.0 <= score <= 100.0:
        raise ValueError(f"Invalid faithfulness score: {score}")
    return score


def generate_answer(question: str, contexts: list[Chunk]) -> str:
    llm = get_llm(user_id="rag-eval")
    if llm is None:
        raise RuntimeError("OPENAI_API_KEY is required for answer generation")
    context = "\n\n".join(chunk.text for chunk in contexts[:5])
    prompt = (
        "请只根据给定医学证据，用简体中文回答问题；证据不足时明确说明，不要补充证据外事实。\n\n"
        f"问题：{question}\n\n证据：{context}"
    )
    return coerce_response_text(llm.invoke(prompt)).strip()


def add_faithfulness_scores(
    pending: list[tuple[dict[str, Any], str, list[Chunk]]],
    *,
    workers: int,
    cache_dir: Path,
    cache_only: bool = False,
) -> None:
    def evaluate(
        item: tuple[dict[str, Any], str, list[Chunk]],
    ) -> tuple[str, float | None]:
        row, question, contexts = item
        digest = hashlib.sha256(LLM_MODEL.encode("utf-8"))
        digest.update(question.encode("utf-8"))
        for context in contexts[:5]:
            digest.update(context.text.encode("utf-8"))
        cache_path = cache_dir / f"{digest.hexdigest()}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                answer = str(cached["answer"])
                score = float(cached["faithfulness"])
                if answer and 0.0 <= score <= 100.0:
                    return answer, score
            except (KeyError, OSError, TypeError, ValueError):
                pass

        if cache_only:
            return "", None

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                answer = generate_answer(question, contexts)
                score = score_faithfulness(question, contexts, answer)
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {"answer": answer, "faithfulness": score},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                return answer, score
            except Exception as exc:
                last_error = exc
                print(
                    json.dumps(
                        {
                            "stage": "faithfulness_retry",
                            "id": row["id"],
                            "attempt": attempt,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        raise RuntimeError(
            f"Faithfulness evaluation failed for {row['id']}: {last_error}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scored = pool.map(evaluate, pending)
        for (row, _, _), (answer, score) in zip(pending, scored):
            row["answer"] = answer
            row["faithfulness"] = score


def _rerank_batches(
    model: CrossEncoder,
    questions: list[str],
    candidate_batches: list[list[Chunk]],
    *,
    top_k: int,
) -> list[list[Chunk]]:
    pairs = [
        [question, chunk.text]
        for question, candidates in zip(questions, candidate_batches)
        for chunk in candidates
    ]
    if not pairs:
        return [[] for _ in candidate_batches]
    all_scores = np.asarray(
        model.predict(pairs, batch_size=12),
        dtype=np.float32,
    )
    ranked_batches = []
    offset = 0
    for candidates in candidate_batches:
        scores = all_scores[offset : offset + len(candidates)]
        offset += len(candidates)
        order = np.argsort(-scores)[:top_k]
        ranked_batches.append([candidates[int(index)] for index in order])
    return ranked_batches


def evaluate_variant(
    variant: Variant,
    *,
    samples: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    embedding_model: SentenceTransformer,
    es_index: ElasticsearchIndex,
    reranker_model: CrossEncoder | None,
    asset_cache: dict[str, tuple[list[Chunk], VectorIndex]],
    rewrite_cache: dict[tuple[str, str], tuple[list[str], str, float]],
    embedding_cache_dir: Path,
    faithfulness_cache_dir: Path,
    corpus_hash: str,
    with_faithfulness: bool,
    faithfulness_workers: int,
    faithfulness_cache_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signature = corpus_signature(variant)
    cached_assets = asset_cache.get(signature)
    if cached_assets is None:
        chunks = build_chunks(corpus, variant)
        cache_path = embedding_cache_dir / f"{corpus_hash}-{signature}.npy"
        matrix = None
        if cache_path.exists():
            try:
                matrix = np.load(cache_path, allow_pickle=False)
                if len(matrix) != len(chunks):
                    matrix = None
            except (OSError, ValueError):
                matrix = None
        print(
            json.dumps(
                {
                    "stage": "prepare_vector_index",
                    "signature": signature,
                    "chunks": len(chunks),
                    "cache_hit": matrix is not None,
                    "cache_path": str(cache_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        vector_index = VectorIndex(embedding_model, chunks, matrix=matrix)
        if matrix is None:
            embedding_cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vector_index.matrix, allow_pickle=False)
        asset_cache[signature] = (chunks, vector_index)
    else:
        chunks, vector_index = cached_assets
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if variant.hybrid_es:
        es_index.rebuild(chunks, signature=signature)

    retrieved_batches = []
    for sample in samples:
        question = str(sample["question"])
        department = str(sample["department"])
        queries = [question]
        rewrite_ms = 0.0
        if variant.query_rewrite:
            rewrite_key = (question, department)
            queries, _, rewrite_ms = rewrite_cache[rewrite_key]

        started = perf_counter()
        dense_rankings = [
            vector_index.search(query, department=department, top_k=12)
            for query in queries
        ]
        dense_results = rrf_merge(dense_rankings, top_k=12)
        results = dense_results
        if variant.hybrid_es:
            keyword_rankings = [
                es_index.search(
                    query,
                    department=department,
                    chunks_by_id=chunks_by_id,
                    top_k=12,
                )
                for query in queries
            ]
            results = rrf_merge([*dense_rankings, *keyword_rankings], top_k=12)
        elapsed_ms = (perf_counter() - started) * 1000.0 + rewrite_ms
        retrieved_batches.append((sample, question, results, elapsed_ms))

    if variant.reranker:
        if reranker_model is None:
            raise RuntimeError(f"Reranker model is required for {variant.name}")
        rerank_started = perf_counter()
        ranked_batches = _rerank_batches(
            reranker_model,
            [item[1] for item in retrieved_batches],
            [item[2] for item in retrieved_batches],
            top_k=5,
        )
        rerank_average_ms = (
            (perf_counter() - rerank_started) * 1000.0 / max(1, len(samples))
        )
    else:
        ranked_batches = [
            item[2] if variant.parent_child else item[2][:5]
            for item in retrieved_batches
        ]
        rerank_average_ms = 0.0

    rows = []
    pending_faithfulness = []
    for (sample, question, _, elapsed_ms), results in zip(
        retrieved_batches, ranked_batches
    ):
        if variant.parent_child:
            results = dedupe_parent_results(results, top_k=5)
        metrics = metric_for_sample(sample, results)
        row = {
            "id": sample["id"],
            **metrics,
            "faithfulness": None,
            "retrieval_ms": elapsed_ms + rerank_average_ms,
            "answer": "",
            "retrieved": [
                {"source": chunk.source, "page": chunk.page} for chunk in results
            ],
            "_contexts": [chunk.text for chunk in results],
        }
        rows.append(row)
        if with_faithfulness:
            pending_faithfulness.append((row, question, results))

    if pending_faithfulness:
        add_faithfulness_scores(
            pending_faithfulness,
            workers=faithfulness_workers,
            cache_dir=faithfulness_cache_dir,
            cache_only=faithfulness_cache_only,
        )

    faithfulness_scores = [
        float(row["faithfulness"])
        for row in rows
        if row["faithfulness"] is not None
    ]

    summary = {
        "name": variant.name,
        "change": variant.change,
        "hit_at_1": mean(row["hit_at_1"] for row in rows) * 100.0,
        "recall_at_5": mean(row["recall_at_5"] for row in rows) * 100.0,
        "mrr": mean(row["mrr"] for row in rows),
        "faithfulness": mean(faithfulness_scores) if faithfulness_scores else None,
        "faithfulness_sample_count": len(faithfulness_scores),
        "average_retrieval_ms": mean(row["retrieval_ms"] for row in rows),
    }
    return summary, rows


def decide_parent_child(c5: dict[str, Any], c6: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        "hit_at_1_pp": c6["hit_at_1"] - c5["hit_at_1"],
        "recall_at_5_pp": c6["recall_at_5"] - c5["recall_at_5"],
        "mrr": c6["mrr"] - c5["mrr"],
        "average_retrieval_ms": c6["average_retrieval_ms"] - c5["average_retrieval_ms"],
    }
    meaningful_gain = (
        deltas["hit_at_1_pp"] >= 1.0
        or deltas["recall_at_5_pp"] >= 1.0
        or deltas["mrr"] >= 0.01
    )
    no_quality_drop = all(
        value >= -1e-9 for key, value in deltas.items() if key != "average_retrieval_ms"
    )
    retained = meaningful_gain and no_quality_drop
    return {
        "decision": "retain" if retained else "discard",
        "rule": "质量至少一项有明确提升且其他质量指标不下降；否则因链路复杂度舍弃",
        "deltas": deltas,
    }


def decide_system_tradeoff(cumulative: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain only cumulative candidates that do not trade away a RAG metric."""
    selected = cumulative[0]
    decisions = []
    quality_keys = ("hit_at_1", "recall_at_5", "mrr")
    for candidate in cumulative[1:]:
        deltas = {
            key: float(candidate[key]) - float(selected[key]) for key in quality_keys
        }
        no_quality_drop = all(value >= -1e-9 for value in deltas.values())
        quality_gain = any(value > 1e-9 for value in deltas.values())
        latency_delta = float(candidate["average_retrieval_ms"]) - float(
            selected["average_retrieval_ms"]
        )
        retained = no_quality_drop and (quality_gain or latency_delta < 0.0)
        decisions.append(
            {
                "variant": candidate["name"],
                "decision": "retain" if retained else "discard",
                "quality_deltas": deltas,
                "average_retrieval_ms_delta": latency_delta,
            }
        )
        if retained:
            selected = candidate
    return {
        "rule": "相对当前保留组合，三个RAG质量指标均不下降，且质量或平均检索耗时至少一项改善",
        "final_variant": selected["name"],
        "decisions": decisions,
    }


def _round_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(summary)
    for key in (
        "hit_at_1",
        "recall_at_5",
        "mrr",
        "faithfulness",
        "average_retrieval_ms",
    ):
        if isinstance(rounded.get(key), (int, float)):
            rounded[key] = round(float(rounded[key]), 4)
    return rounded


def select_codex_audit_ids(samples: list[dict[str, Any]]) -> list[str]:
    """Select 40 deterministic, stratified samples across all RAG categories."""
    if len(samples) < 40:
        return [str(sample["id"]) for sample in samples]
    quotas = {"single_hop": 14, "multi_hop": 13, "hard_retrieval": 13}
    selected = []
    for category, quota in quotas.items():
        candidates = [
            str(sample["id"])
            for sample in samples
            if sample.get("category") == category
        ]
        if len(candidates) < quota:
            raise ValueError(f"Not enough {category} samples for Codex audit")
        selected.extend(
            candidates[math.floor(index * len(candidates) / quota)]
            for index in range(quota)
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--es-url", default=ES_HOST)
    parser.add_argument("--knowledge-root", default=KNOWLEDGE_ROOT_DIR)
    parser.add_argument(
        "--embedding-cache-dir", default="/tmp/medagent-rag-vector-cache"
    )
    parser.add_argument(
        "--faithfulness-cache-dir", default="/tmp/medagent-faithfulness-cache"
    )
    parser.add_argument("--rewrite-cache-dir", default="/tmp/medagent-rewrite-cache")
    parser.add_argument(
        "--embedding-device",
        default="auto",
        choices=("auto", "cpu", "mps"),
    )
    parser.add_argument("--rewrite-workers", type=int, default=8)
    parser.add_argument("--with-faithfulness", action="store_true")
    parser.add_argument(
        "--faithfulness-cache-only",
        action="store_true",
        help="Read completed faithfulness scores without making model calls",
    )
    parser.add_argument("--faithfulness-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    samples = list(dataset["samples"])
    if args.limit:
        samples = samples[: args.limit]
    corpus = build_corpus(Path(args.knowledge_root))
    embedding_device = args.embedding_device
    if embedding_device == "auto":
        embedding_device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        json.dumps(
            {
                "stage": "corpus_ready",
                "pdf_pages": len(corpus),
                "embedding_device": embedding_device,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    embedding_model = SentenceTransformer(
        local_model_path(EMBEDDING_MODEL_NAME),
        device=embedding_device,
    )
    es_index = ElasticsearchIndex(args.es_url)
    es_index.assert_available()
    reranker_model = CrossEncoder(local_model_path(RERANKER_MODEL_NAME), device="cpu")
    corpus_hash = corpus_fingerprint(corpus, EMBEDDING_MODEL_NAME)

    independent = []
    cumulative = []
    per_variant: dict[str, list[dict[str, Any]]] = {}
    asset_cache: dict[str, tuple[list[Chunk], VectorIndex]] = {}
    rewrite_cache = build_rewrite_cache(
        samples,
        workers=args.rewrite_workers,
        cache_dir=Path(args.rewrite_cache_dir),
    )
    for variant in INDEPENDENT_VARIANTS:
        summary, rows = evaluate_variant(
            variant,
            samples=samples,
            corpus=corpus,
            embedding_model=embedding_model,
            es_index=es_index,
            reranker_model=reranker_model,
            asset_cache=asset_cache,
            rewrite_cache=rewrite_cache,
            embedding_cache_dir=Path(args.embedding_cache_dir),
            faithfulness_cache_dir=Path(args.faithfulness_cache_dir),
            corpus_hash=corpus_hash,
            with_faithfulness=(
                args.with_faithfulness or args.faithfulness_cache_only
            ),
            faithfulness_workers=args.faithfulness_workers,
            faithfulness_cache_only=args.faithfulness_cache_only,
        )
        independent.append(_round_summary(summary))
        per_variant[variant.name] = rows
        print(
            json.dumps(
                {"completed": variant.name, "summary": independent[-1]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    for variant in CUMULATIVE_VARIANTS:
        matching = next(
            (
                candidate
                for candidate in INDEPENDENT_VARIANTS
                if variant_configuration(candidate) == variant_configuration(variant)
            ),
            None,
        )
        if matching is not None:
            source = next(item for item in independent if item["name"] == matching.name)
            summary = {**source, "name": variant.name, "change": variant.change}
            rows = per_variant[matching.name]
        else:
            summary, rows = evaluate_variant(
                variant,
                samples=samples,
                corpus=corpus,
                embedding_model=embedding_model,
                es_index=es_index,
                reranker_model=reranker_model,
                asset_cache=asset_cache,
                rewrite_cache=rewrite_cache,
                embedding_cache_dir=Path(args.embedding_cache_dir),
                faithfulness_cache_dir=Path(args.faithfulness_cache_dir),
                corpus_hash=corpus_hash,
                with_faithfulness=(
                    args.with_faithfulness or args.faithfulness_cache_only
                ),
                faithfulness_workers=args.faithfulness_workers,
                faithfulness_cache_only=args.faithfulness_cache_only,
            )
        cumulative.append(_round_summary(summary))
        per_variant[variant.name] = rows
        print(
            json.dumps(
                {"completed": variant.name, "summary": cumulative[-1]},
                ensure_ascii=False,
            ),
            flush=True,
        )

    parent_child_decision = decide_parent_child(cumulative[-2], cumulative[-1])
    system_tradeoff = decide_system_tradeoff(cumulative)
    knowledge_root = Path(args.knowledge_root).resolve()
    try:
        corpus_source = str(knowledge_root.relative_to(PROJECT_ROOT))
    except ValueError:
        corpus_source = str(knowledge_root)
    result = {
        "metadata": {
            "dataset": dataset["metadata"]["name"],
            "samples": len(samples),
            "corpus_pdf_pages": len(corpus),
            "corpus_source": corpus_source,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker_model": RERANKER_MODEL_NAME,
            "elasticsearch": args.es_url,
            "faithfulness_enabled": args.with_faithfulness,
            "faithfulness_mode": (
                "live"
                if args.with_faithfulness
                else "cache_only"
                if args.faithfulness_cache_only
                else "disabled"
            ),
            "metric_definitions": {
                "hit_at_1": "首条结果命中任一 gold PDF 页的样本比例",
                "recall_at_5": "前5条召回的唯一 gold PDF 页数 / 该题全部 gold PDF 页数",
                "mrr": "第一条正确 gold PDF 页排名的倒数均值，前5条未命中记0",
                "faithfulness": "有证据支持的医学结论数 / 全部可核对医学结论数 × 100",
            },
        },
        "independent": independent,
        "cumulative": cumulative,
        "parent_child_tradeoff": parent_child_decision,
        "system_tradeoff": system_tradeoff,
        "per_variant": per_variant,
    }

    final_rows = per_variant[system_tradeoff["final_variant"]]
    samples_by_id = {str(sample["id"]): sample for sample in samples}
    rows_by_id = {str(row["id"]): row for row in final_rows}
    audit_rows = [
        {
            "id": sample_id,
            "question": samples_by_id[sample_id]["question"],
            "answer": row["answer"],
            "retrieved": row["retrieved"],
            "evidence": row["_contexts"],
            "model_faithfulness": row["faithfulness"],
            "codex_supported_claims": None,
            "codex_total_claims": None,
            "codex_faithfulness": None,
            "codex_note": "",
        }
        for sample_id in select_codex_audit_ids(samples)
        for row in [rows_by_id[sample_id]]
    ]
    audit_model_scored = sum(
        row["model_faithfulness"] is not None for row in audit_rows
    )
    for rows in per_variant.values():
        for row in rows:
            row.pop("_contexts", None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_output = Path(args.audit_output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(
            {
                "metadata": {
                    "sample_size": len(audit_rows),
                    "model_scored_samples": audit_model_scored,
                    "status": (
                        "pending_codex_review"
                        if audit_model_scored == len(audit_rows)
                        else "pending_model_scoring"
                    ),
                },
                "samples": audit_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "parent_child_decision": parent_child_decision,
                "system_tradeoff": system_tradeoff,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
