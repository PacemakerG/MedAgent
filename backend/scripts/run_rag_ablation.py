# isort: skip_file
"""Run a reproducible retrieval ablation over the 180-row evaluation dataset.

The experiment evaluates only the grounded single-hop and multi-hop rows. The
open-domain and negative rows remain available for full-workflow evaluation.

Variants add one change at a time:
1. fixed raw chunks + dense retrieval
2. semantic-boundary raw chunks + dense retrieval
3. parent-child deduplication
4. OCR/text cleanup
5. parallel Elasticsearch BM25 + dense retrieval with RRF
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from app.core.langsmith_service import (  # noqa: E402
    configure_langsmith,
    is_langsmith_enabled,
    langsmith_traceable,
)
from build_langsmith_eval_dataset import (  # noqa: E402
    DATASET_VERSION,
    DEFAULT_OUTPUT_PATH as DEFAULT_DATASET_PATH,
    DEFAULT_SOURCE_PATHS,
    load_jsonl,
)

DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "data" / "eval" / "rag_ablation_result_v2.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "docs" / "evaluation" / "rag_ablation_results_v2.csv"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ES_INDEX = "medigenius_rag_ablation_v2"


@dataclass(frozen=True)
class CorpusDocument:
    source_id: str
    department: str
    content: str
    source_book: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    department: str
    content: str
    parent_content: str


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    chunk_strategy: str
    parent_child: bool
    cleaned: bool
    hybrid_es: bool


VARIANTS = (
    Variant(
        "a0_fixed_raw_vector",
        "固定长度切分 + 向量召回",
        "fixed",
        False,
        False,
        False,
    ),
    Variant(
        "a1_semantic_raw_vector",
        "语义边界切分 + 向量召回",
        "semantic",
        False,
        False,
        False,
    ),
    Variant(
        "a2_parent_child_raw_vector",
        "语义切分 + 父子索引 + 向量召回",
        "semantic",
        True,
        False,
        False,
    ),
    Variant(
        "a3_parent_child_clean_vector",
        "数据清洗 + 语义切分 + 父子索引 + 向量召回",
        "semantic",
        True,
        True,
        False,
    ),
    Variant(
        "a4_parent_child_clean_es_rrf",
        "数据清洗 + 父子索引 + ES/向量并行召回 + RRF",
        "semantic",
        True,
        True,
        True,
    ),
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def clean_ocr_text(value: str) -> str:
    """Apply conservative normalization without changing medical meaning."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=\d)[oO](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[lI](?=\d)", "1", text)
    text = re.sub(r"[〜~]{2,}", "~", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fixed_chunks(text: str, *, size: int = 96, overlap: int = 24) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    step = max(1, size - overlap)
    return [
        normalized[start : start + size] for start in range(0, len(normalized), step)
    ]


def semantic_chunks(text: str, *, target_size: int = 160) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？；!?;])\s*", normalized)
        if item.strip()
    ]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > target_size * 2:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_chunks(sentence, size=target_size, overlap=20))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > target_size:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [normalized]


def load_corpus(source_paths: Sequence[Path]) -> List[CorpusDocument]:
    seen = set()
    corpus: List[CorpusDocument] = []
    for path in source_paths:
        for row in load_jsonl(path):
            source_id = str(row.get("id") or "").strip()
            if not source_id or source_id in seen:
                continue
            content = str(row.get("reference_answer") or "").strip()
            department = str(
                row.get("expected_department") or row.get("selected_department") or ""
            ).strip()
            if not content or not department:
                continue
            seen.add(source_id)
            corpus.append(
                CorpusDocument(
                    source_id=source_id,
                    department=department,
                    content=content,
                    source_book=str(
                        row.get("source_book") or row.get("expected_source_book") or ""
                    ),
                )
            )
    if not corpus:
        raise ValueError("No grounded corpus documents were found.")
    return corpus


def build_chunks(corpus: Sequence[CorpusDocument], variant: Variant) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in corpus:
        parent_content = (
            clean_ocr_text(document.content) if variant.cleaned else document.content
        )
        parts = (
            fixed_chunks(parent_content)
            if variant.chunk_strategy == "fixed"
            else semantic_chunks(parent_content)
        )
        for index, content in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}:{index:03d}",
                    source_id=document.source_id,
                    department=document.department,
                    content=content,
                    parent_content=parent_content,
                )
            )
    return chunks


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def unique_source_ranking(ranked_chunks: Iterable[Chunk], top_k: int) -> List[str]:
    ranking: List[str] = []
    seen = set()
    for chunk in ranked_chunks:
        if chunk.source_id in seen:
            continue
        seen.add(chunk.source_id)
        ranking.append(chunk.source_id)
        if len(ranking) >= top_k:
            break
    return ranking


def rrf_fuse(
    rankings: Sequence[Sequence[str]], *, top_k: int, rrf_k: int = 60
) -> List[str]:
    scores: Dict[str, float] = {}
    best_rank: Dict[str, int] = {}
    for ranking in rankings:
        seen = set()
        for rank, source_id in enumerate(ranking, start=1):
            if source_id in seen:
                continue
            seen.add(source_id)
            scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[source_id] = min(best_rank.get(source_id, rank), rank)
    return [
        source_id
        for source_id, _ in sorted(
            scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0])
        )[:top_k]
    ]


def gold_source_ids(sample: Dict[str, Any]) -> List[str]:
    ids = [
        str(source.get("source_id") or "").strip()
        for source in sample.get("expected_sources") or []
    ]
    return list(dict.fromkeys(source_id for source_id in ids if source_id))


def score_ranking(
    ranking: Sequence[str], gold_ids: Sequence[str], top_k: int
) -> Dict[str, float]:
    gold = set(gold_ids)
    top = list(ranking[:top_k])
    if not gold:
        return {"top1": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "complete_at_k": 0.0}
    first_rank = next(
        (idx for idx, source_id in enumerate(top, 1) if source_id in gold), None
    )
    hit_count = len(gold.intersection(top))
    return {
        "top1": float(bool(top and top[0] in gold)),
        "recall_at_k": hit_count / len(gold),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        "complete_at_k": float(hit_count == len(gold)),
    }


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray):
        self.chunks = list(chunks)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.by_department: Dict[str, np.ndarray] = {}
        departments = sorted({chunk.department for chunk in self.chunks})
        for department in departments:
            self.by_department[department] = np.asarray(
                [
                    idx
                    for idx, chunk in enumerate(self.chunks)
                    if chunk.department == department
                ],
                dtype=np.int64,
            )

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        department: str,
        top_k: int,
        parent_child: bool,
    ) -> List[str]:
        indices = self.by_department.get(department)
        if indices is None or not len(indices):
            return []
        scores = self.embeddings[indices] @ np.asarray(
            query_embedding, dtype=np.float32
        )
        order = indices[np.argsort(-scores)]
        chunks = [self.chunks[int(index)] for index in order]
        if parent_child:
            return unique_source_ranking(chunks, top_k)
        return [chunk.source_id for chunk in chunks[:top_k]]


class ElasticsearchIndex:
    def __init__(self, base_url: str, index_name: str):
        self.base_url = base_url.rstrip("/")
        self.index_name = index_name
        self.client = httpx.Client(base_url=self.base_url, timeout=20.0)

    def close(self) -> None:
        self.client.close()

    def check(self) -> None:
        response = self.client.get("/")
        response.raise_for_status()

    def rebuild(self, chunks: Sequence[Chunk]) -> float:
        started = perf_counter()
        self.client.delete(f"/{self.index_name}")
        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "cjk"},
                    "source_id": {"type": "keyword"},
                    "department": {"type": "keyword"},
                }
            },
        }
        response = self.client.put(f"/{self.index_name}", json=mapping)
        response.raise_for_status()
        lines: List[str] = []
        for chunk in chunks:
            lines.append(
                json.dumps(
                    {"index": {"_index": self.index_name, "_id": chunk.chunk_id}},
                    ensure_ascii=False,
                )
            )
            lines.append(
                json.dumps(
                    {
                        "content": chunk.content,
                        "source_id": chunk.source_id,
                        "department": chunk.department,
                    },
                    ensure_ascii=False,
                )
            )
        response = self.client.post(
            "/_bulk?refresh=true",
            content="\n".join(lines) + "\n",
            headers={"content-type": "application/x-ndjson"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError("Elasticsearch bulk indexing returned item errors")
        return (perf_counter() - started) * 1000.0

    def search(self, query: str, *, department: str, top_k: int) -> List[str]:
        payload = {
            "size": max(top_k * 4, 20),
            "query": {
                "bool": {
                    "must": [{"match": {"content": {"query": query}}}],
                    "filter": [{"term": {"department": department}}],
                }
            },
            "_source": ["source_id"],
        }
        response = self.client.post(f"/{self.index_name}/_search", json=payload)
        response.raise_for_status()
        hits = (response.json().get("hits") or {}).get("hits") or []
        ranking: List[str] = []
        seen = set()
        for hit in hits:
            source_id = str((hit.get("_source") or {}).get("source_id") or "")
            if not source_id or source_id in seen:
                continue
            seen.add(source_id)
            ranking.append(source_id)
            if len(ranking) >= top_k:
                break
        return ranking


def summarize_sample_results(
    items: Sequence[Dict[str, Any]], top_k: int
) -> Dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in items]
    return {
        "samples": len(items),
        "top1_accuracy": round(mean(item["top1"] for item in items), 4),
        f"recall@{top_k}": round(mean(item["recall_at_k"] for item in items), 4),
        "mrr": round(mean(item["mrr"] for item in items), 4),
        f"complete@{top_k}": round(mean(item["complete_at_k"] for item in items), 4),
        "latency_avg_ms": round(mean(latencies), 4),
        "latency_p50_ms": round(median(latencies), 4),
        "latency_p95_ms": round(percentile(latencies, 0.95), 4),
    }


@langsmith_traceable("eval.rag_ablation_variant", run_type="chain")
def evaluate_variant(
    variant: Variant,
    samples: Sequence[Dict[str, Any]],
    query_embeddings: np.ndarray,
    model: SentenceTransformer,
    *,
    top_k: int,
    es_url: str,
    es_index_name: str,
) -> Dict[str, Any]:
    chunks = build_chunks(load_corpus(DEFAULT_SOURCE_PATHS), variant)
    index_started = perf_counter()
    chunk_embeddings = model.encode(
        [chunk.content for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dense_index = DenseIndex(chunks, chunk_embeddings)
    dense_index_ms = (perf_counter() - index_started) * 1000.0
    es_index: Optional[ElasticsearchIndex] = None
    es_index_ms = 0.0
    if variant.hybrid_es:
        es_index = ElasticsearchIndex(es_url, es_index_name)
        es_index.check()
        es_index_ms = es_index.rebuild(chunks)

    results: List[Dict[str, Any]] = []
    try:
        for sample_index, (sample, query_embedding) in enumerate(
            zip(samples, query_embeddings)
        ):
            started = perf_counter()
            department = str(sample.get("expected_department") or "")
            if es_index:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(
                        dense_index.search,
                        query_embedding,
                        department=department,
                        top_k=max(top_k * 4, 20),
                        parent_child=True,
                    )
                    es_future = executor.submit(
                        es_index.search,
                        str(sample.get("question") or ""),
                        department=department,
                        top_k=max(top_k * 4, 20),
                    )
                    vector_ranking = vector_future.result()
                    keyword_ranking = es_future.result()
                ranking = rrf_fuse(
                    [vector_ranking, keyword_ranking], top_k=top_k, rrf_k=60
                )
            else:
                vector_ranking = dense_index.search(
                    query_embedding,
                    department=department,
                    top_k=top_k,
                    parent_child=variant.parent_child,
                )
                keyword_ranking = []
                ranking = vector_ranking
            latency_ms = (perf_counter() - started) * 1000.0
            metrics = score_ranking(ranking, gold_source_ids(sample), top_k)
            results.append(
                {
                    "id": sample.get("id", f"sample-{sample_index}"),
                    "category": sample.get("category", ""),
                    "department": department,
                    "gold_source_ids": gold_source_ids(sample),
                    "ranking": ranking,
                    "vector_ranking": vector_ranking[:top_k],
                    "keyword_ranking": keyword_ranking[:top_k],
                    "latency_ms": round(latency_ms, 4),
                    **metrics,
                }
            )
    finally:
        if es_index:
            es_index.close()

    return {
        "name": variant.name,
        "description": variant.description,
        "configuration": {
            "chunk_strategy": variant.chunk_strategy,
            "parent_child": variant.parent_child,
            "cleaned": variant.cleaned,
            "hybrid_es": variant.hybrid_es,
            "keyword_backend": (
                "elasticsearch_bm25" if variant.hybrid_es else "disabled"
            ),
            "fusion": "rrf_k_60" if variant.hybrid_es else "none",
            "chunk_count": len(chunks),
        },
        "indexing": {
            "dense_index_ms": round(dense_index_ms, 4),
            "elasticsearch_index_ms": round(es_index_ms, 4),
        },
        "summary": summarize_sample_results(results, top_k),
        "by_category": {
            category: summarize_sample_results(
                [item for item in results if item["category"] == category], top_k
            )
            for category in ("single_hop", "multi_hop")
        },
        "results": results,
    }


def write_csv(variants: Sequence[Dict[str, Any]], path: Path, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "variant",
        "description",
        "samples",
        "top1_accuracy",
        f"recall@{top_k}",
        "mrr",
        f"complete@{top_k}",
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "dense_index_ms",
        "elasticsearch_index_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "variant": variant["name"],
                    "description": variant["description"],
                    **variant["summary"],
                    **variant["indexing"],
                }
            )


def run_ablation(
    *,
    dataset_path: Path,
    output_path: Path,
    csv_path: Path,
    model_name: str,
    top_k: int,
    es_url: str,
    es_index_name: str,
) -> Dict[str, Any]:
    configure_langsmith()
    all_rows = load_jsonl(dataset_path)
    samples = [
        row
        for row in all_rows
        if row.get("category") in {"single_hop", "multi_hop"}
        and row.get("should_use_rag")
    ]
    if not samples:
        raise ValueError(f"No grounded samples in {dataset_path}")
    model_started = perf_counter()
    model = SentenceTransformer(model_name)
    model_load_ms = (perf_counter() - model_started) * 1000.0
    query_started = perf_counter()
    query_embeddings = model.encode(
        [str(sample.get("question") or "") for sample in samples],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding_total_ms = (perf_counter() - query_started) * 1000.0

    variants = [
        evaluate_variant(
            variant,
            samples,
            query_embeddings,
            model,
            top_k=top_k,
            es_url=es_url,
            es_index_name=es_index_name,
        )
        for variant in VARIANTS
    ]
    payload = {
        "experiment": "medigenius_rag_ablation_v2",
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_samples": len(all_rows),
        "retrieval_samples": len(samples),
        "excluded_from_retrieval": len(all_rows) - len(samples),
        "top_k": top_k,
        "model": model_name,
        "model_load_ms": round(model_load_ms, 4),
        "query_embedding_total_ms": round(query_embedding_total_ms, 4),
        "query_embedding_avg_ms": round(query_embedding_total_ms / len(samples), 4),
        "latency_scope": "retrieval only; query embeddings precomputed once",
        "langsmith_tracing_enabled": is_langsmith_enabled(),
        "elasticsearch_url": es_url,
        "elasticsearch_index": es_index_name,
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(variants, csv_path, top_k)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MediGenius RAG ablation.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--es-url", default="http://127.0.0.1:9200")
    parser.add_argument("--es-index", default=DEFAULT_ES_INDEX)
    args = parser.parse_args()
    payload = run_ablation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        csv_path=Path(args.csv),
        model_name=str(args.model),
        top_k=max(1, int(args.top_k)),
        es_url=str(args.es_url),
        es_index_name=str(args.es_index),
    )
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "retrieval_samples": payload["retrieval_samples"],
                "langsmith_tracing_enabled": payload["langsmith_tracing_enabled"],
                "variants": [
                    {"name": item["name"], **item["summary"]}
                    for item in payload["variants"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
