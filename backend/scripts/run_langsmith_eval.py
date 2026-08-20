# isort: skip_file
"""
Run MediGenius full-workflow evaluation with LangSmith tracing metadata.

The runner can be used locally without LangSmith credentials. When LangSmith is
configured, the existing LangGraph/LangChain tracing and agent-level traceable
decorators will attach the sample metadata to each workflow run.

Usage:
  python backend/scripts/run_langsmith_eval.py \
    --dataset backend/data/eval/langsmith_eval_dataset_v1.jsonl \
    --output backend/data/eval/langsmith_eval_result_v1.json \
    --report docs/evaluation/langsmith_eval_report_v1.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(BACKEND_ROOT / ".env")

from app.core.langsmith_service import (  # noqa: E402
    build_langsmith_runnable_config,
    configure_langsmith,
    is_langsmith_enabled,
)
from app.core.state import initialize_conversation_state  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from build_langsmith_eval_dataset import DATASET_VERSION, load_jsonl  # noqa: E402
from evaluate_rag_pipeline import _judge_answer, _retrieval_metrics  # noqa: E402

DEFAULT_DATASET_PATH = (
    BACKEND_ROOT / "data" / "eval" / "langsmith_eval_dataset_v2_180.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    BACKEND_ROOT / "data" / "eval" / "langsmith_eval_result_v2_180.json"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs" / "evaluation" / "langsmith_eval_report_v2_180.md"
)

REFUSAL_MARKERS = (
    "不能",
    "无法",
    "不建议",
    "不提供",
    "需要医生",
    "线下",
    "就医",
    "急诊",
    "急救",
    "补充",
    "信息不足",
    "不确定",
    "危机",
    "求助",
)
EMERGENCY_MARKERS = ("立即", "急诊", "急救", "120", "呼叫", "马上")


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _json_safe(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _extract_contexts(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts = result.get("rag_context") or result.get("reranked_rag_context") or []
    if not isinstance(contexts, list):
        return []
    return [item for item in contexts if isinstance(item, dict)]


def _actual_rag_used(result: Dict[str, Any]) -> bool:
    return bool(
        result.get("use_rag")
        or result.get("need_rag")
        or result.get("rag_attempted")
        or result.get("rag_success")
        or _extract_contexts(result)
    )


def _keyword_hit(answer: str, keywords: List[str]) -> Optional[bool]:
    normalized = _normalize_text(answer)
    cleaned = [_normalize_text(item) for item in keywords if _normalize_text(item)]
    if not cleaned:
        return None
    required_hits = min(2, len(cleaned))
    hits = sum(1 for item in cleaned if item in normalized)
    return hits >= required_hits


def _behavior_pass(sample: Dict[str, Any], answer: str, result: Dict[str, Any]) -> bool:
    category = sample.get("category", "")
    behavior = str(sample.get("expected_behavior", ""))
    normalized = _normalize_text(answer)

    if category in {"single_hop", "multi_hop"}:
        keyword_result = _keyword_hit(
            answer, list(sample.get("expected_keywords") or [])
        )
        return (
            bool(keyword_result) if keyword_result is not None else bool(answer.strip())
        )

    if category == "open_domain":
        if not answer.strip() or len(answer.strip()) < 20:
            return False
        if "no retrieved context" in normalized:
            return False
        return True

    if category == "negative":
        if behavior == "emergency_triage":
            return any(marker in answer for marker in EMERGENCY_MARKERS)
        return any(marker in answer for marker in REFUSAL_MARKERS)

    return bool(result.get("generation"))


def _build_initial_state(sample: Dict[str, Any]) -> Dict[str, Any]:
    state = initialize_conversation_state()
    state["tenant_id"] = "eval"
    state["user_id"] = "langsmith_eval"
    state["session_id"] = f"eval-{sample.get('id', 'sample')}"
    state["question"] = str(sample.get("question", ""))

    selected_department = str(sample.get("selected_department") or "").strip()
    if selected_department:
        state["selected_department"] = selected_department
        state["selected_department_forced"] = True
    return state


def _invoke_workflow(
    workflow_app: Any, state: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    # Evaluation should not mutate persistent user profiles.
    with patch("app.agents.memory.schedule_profile_update"):
        return workflow_app.invoke(state, config=config)


def run_single_sample(
    sample: Dict[str, Any],
    *,
    workflow_app: Any,
    top_k: int,
    with_judge: bool,
) -> Dict[str, Any]:
    started = perf_counter()
    sample_id = str(sample.get("id", ""))
    category = str(sample.get("category", "unknown"))
    state = _build_initial_state(sample)
    config = build_langsmith_runnable_config(
        operation="eval.full_workflow",
        session_id=str(state.get("session_id", "")),
        tenant_id=str(state.get("tenant_id", "eval")),
        user_id=str(state.get("user_id", "langsmith_eval")),
        selected_department=sample.get("selected_department") or None,
        extra_tags=["eval", category, DATASET_VERSION],
        extra_metadata={
            "sample_id": sample_id,
            "category": category,
            "expected_behavior": sample.get("expected_behavior", ""),
            "should_use_rag": bool(sample.get("should_use_rag", False)),
            "dataset_version": sample.get("dataset_version", DATASET_VERSION),
        },
    )

    try:
        result = _invoke_workflow(workflow_app, state, config)
        error = ""
    except Exception as exc:
        result = dict(state)
        result["generation"] = ""
        result["source"] = "Evaluation Error"
        result["flow_trace"] = list(result.get("flow_trace") or []) + [
            "evaluation_error"
        ]
        error = str(exc)

    contexts = _extract_contexts(result)
    top1_hit = recall_hit = 0
    mrr = 0.0
    if sample.get("should_use_rag"):
        top1_hit, recall_hit, mrr = _retrieval_metrics(contexts, sample, top_k=top_k)

    answer = str(result.get("generation") or "")
    route_match = _actual_rag_used(result) == bool(sample.get("should_use_rag", False))
    behavior_ok = _behavior_pass(sample, answer, result)
    keyword_result = _keyword_hit(answer, list(sample.get("expected_keywords") or []))
    item = {
        "id": sample_id,
        "category": category,
        "question": sample.get("question", ""),
        "expected_behavior": sample.get("expected_behavior", ""),
        "should_use_rag": bool(sample.get("should_use_rag", False)),
        "actual_rag_used": _actual_rag_used(result),
        "route_match": route_match,
        "top1_hit": top1_hit,
        "recall_hit": recall_hit,
        "mrr": mrr,
        "behavior_pass": behavior_ok,
        "answer_keyword_hit": keyword_result,
        "retrieved_context_count": len(contexts),
        "source": result.get("source", ""),
        "flow_trace": result.get("flow_trace", []),
        "primary_department": result.get("primary_department", ""),
        "retrieval_scopes": result.get("retrieval_scopes", []),
        "retrieval_query": result.get("retrieval_query", ""),
        "retrieval_queries": result.get("retrieval_queries", []),
        "safety_level": result.get("safety_level", ""),
        "domain": result.get("domain", ""),
        "llm_success": bool(result.get("llm_success", False)),
        "rag_success": bool(result.get("rag_success", False)),
        "profiling": result.get("profiling", {}),
        "answer": answer,
        "error": error,
        "elapsed_ms": round((perf_counter() - started) * 1000.0, 2),
    }

    if with_judge:
        item["judge"] = _judge_answer(sample, answer, contexts)

    return _json_safe(item)


def _mean_bool(results: List[Dict[str, Any]], field: str) -> float:
    values = [1 if item.get(field) else 0 for item in results]
    return round(mean(values), 4) if values else 0.0


def _mean_float(results: List[Dict[str, Any]], field: str) -> float:
    values = [float(item.get(field, 0.0)) for item in results]
    return round(mean(values), 4) if values else 0.0


def _build_summary(
    results: List[Dict[str, Any]], top_k: int, with_judge: bool
) -> Dict[str, Any]:
    category_counts = Counter(str(item.get("category", "unknown")) for item in results)
    by_category: Dict[str, Dict[str, Any]] = {}
    for category in sorted(category_counts):
        items = [item for item in results if item.get("category") == category]
        rag_items = [item for item in items if item.get("should_use_rag")]
        by_category[category] = {
            "samples": len(items),
            "route_match_rate": _mean_bool(items, "route_match"),
            "behavior_pass_rate": _mean_bool(items, "behavior_pass"),
            "llm_success_rate": _mean_bool(items, "llm_success"),
            "rag_success_rate": _mean_bool(items, "rag_success"),
            "avg_elapsed_ms": _mean_float(items, "elapsed_ms"),
        }
        if rag_items:
            by_category[category].update(
                {
                    "top1_accuracy": _mean_bool(rag_items, "top1_hit"),
                    f"recall@{top_k}": _mean_bool(rag_items, "recall_hit"),
                    "mrr": _mean_float(rag_items, "mrr"),
                }
            )

    rag_results = [item for item in results if item.get("should_use_rag")]
    summary: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "samples": len(results),
        "category_counts": dict(category_counts),
        "route_match_rate": _mean_bool(results, "route_match"),
        "behavior_pass_rate": _mean_bool(results, "behavior_pass"),
        "llm_success_rate": _mean_bool(results, "llm_success"),
        "rag_success_rate": _mean_bool(results, "rag_success"),
        "error_count": sum(1 for item in results if item.get("error")),
        "rag_expected_samples": len(rag_results),
        "top1_accuracy": _mean_bool(rag_results, "top1_hit") if rag_results else 0.0,
        f"recall@{top_k}": (
            _mean_bool(rag_results, "recall_hit") if rag_results else 0.0
        ),
        "mrr": _mean_float(rag_results, "mrr") if rag_results else 0.0,
        "avg_elapsed_ms": _mean_float(results, "elapsed_ms"),
        "langsmith_tracing_enabled": is_langsmith_enabled(),
        "by_category": by_category,
    }

    if with_judge:
        judge_scores: Dict[str, List[float]] = defaultdict(list)
        for item in results:
            judge = item.get("judge") or {}
            for key in ("correctness", "faithfulness", "relevance"):
                judge_scores[key].append(float(judge.get(key, 0.0)))
        for key, values in judge_scores.items():
            summary[f"judge_{key}_avg"] = round(mean(values), 4) if values else 0.0

    return summary


def run_eval(
    dataset_path: Path,
    *,
    top_k: int = 5,
    with_judge: bool = False,
    limit: Optional[int] = None,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    configure_langsmith()
    samples = load_jsonl(dataset_path)
    if not samples:
        raise ValueError(f"Dataset empty: {dataset_path}")
    if limit is not None and limit > 0:
        samples = samples[:limit]

    chat = ChatService()
    chat.initialize_workflow()
    workflow_app = chat.workflow_app
    results = [
        run_single_sample(
            sample,
            workflow_app=workflow_app,
            top_k=max(1, int(top_k)),
            with_judge=with_judge,
        )
        for sample in samples
    ]
    payload = {
        "summary": _build_summary(results, max(1, int(top_k)), with_judge),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if report_path:
        write_eval_report(
            payload, dataset_path, output_path, report_path, top_k=max(1, int(top_k))
        )
    return payload


def write_eval_report(
    payload: Dict[str, Any],
    dataset_path: Path,
    output_path: Optional[Path],
    report_path: Path,
    *,
    top_k: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload.get("summary", {})
    results = payload.get("results", [])
    failures = [
        item
        for item in results
        if item.get("error")
        or not item.get("route_match")
        or not item.get("behavior_pass")
        or (item.get("should_use_rag") and not item.get("recall_hit"))
    ][:12]
    lines = [
        "# LangSmith 全流程评测报告",
        "",
        f"- 数据集：`{dataset_path}`",
        f"- 结果文件：`{output_path or ''}`",
        f"- 样本数：`{summary.get('samples', 0)}`",
        f"- LangSmith tracing：`{summary.get('langsmith_tracing_enabled', False)}`",
        f"- 路由匹配率：`{summary.get('route_match_rate', 0.0)}`",
        f"- 行为通过率：`{summary.get('behavior_pass_rate', 0.0)}`",
        f"- LLM 成功率：`{summary.get('llm_success_rate', 0.0)}`",
        f"- RAG 成功率：`{summary.get('rag_success_rate', 0.0)}`",
        f"- runner 错误数：`{summary.get('error_count', 0)}`",
        f"- RAG Top1：`{summary.get('top1_accuracy', 0.0)}`",
        f"- RAG Recall@{top_k}：`{summary.get(f'recall@{top_k}', 0.0)}`",
        f"- RAG MRR：`{summary.get('mrr', 0.0)}`",
        "",
        "## 结果解读",
        "",
        (
            "- 本轮 LLM 成功率为 0，说明生成模型调用未成功；RAG 类样本的行为通过率会被显著拉低，"
            "应优先检查模型供应商拦截、模型名、API key 或 base URL。"
            if float(summary.get("llm_success_rate", 0.0)) == 0.0
            else "- LLM 调用可用，行为通过率可作为回答质量指标参考。"
        ),
        "",
        "## 分类指标",
        "",
        "| 分类 | 样本数 | 路由匹配 | 行为通过 | LLM成功 | RAG成功 | Top1 | Recall | MRR | 平均耗时 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, item in sorted((summary.get("by_category") or {}).items()):
        lines.append(
            "| {category} | {samples} | {route} | {behavior} | {llm} | {rag} | "
            "{top1} | {recall} | {mrr} | {elapsed} |".format(
                category=category,
                samples=item.get("samples", 0),
                route=item.get("route_match_rate", ""),
                behavior=item.get("behavior_pass_rate", ""),
                llm=item.get("llm_success_rate", ""),
                rag=item.get("rag_success_rate", ""),
                top1=item.get("top1_accuracy", ""),
                recall=item.get(f"recall@{top_k}", ""),
                mrr=item.get("mrr", ""),
                elapsed=item.get("avg_elapsed_ms", ""),
            )
        )

    lines.extend(["", "## 需要关注的样本", ""])
    if not failures:
        lines.append("本轮没有发现路由、行为或检索召回失败样本。")
    else:
        for item in failures:
            lines.append(
                "- `{id}` `{category}` route={route} behavior={behavior} "
                "recall={recall} error=`{error}` question={question}".format(
                    id=item.get("id", ""),
                    category=item.get("category", ""),
                    route=item.get("route_match", False),
                    behavior=item.get("behavior_pass", False),
                    recall=item.get("recall_hit", 0),
                    error=item.get("error", ""),
                    question=str(item.get("question", ""))[:120],
                )
            )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MediGenius LangSmith-style evaluation."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-judge", action="store_true")
    args = parser.parse_args()

    payload = run_eval(
        Path(args.dataset),
        top_k=max(1, int(args.top_k)),
        with_judge=bool(args.with_judge),
        limit=int(args.limit) if int(args.limit) > 0 else None,
        output_path=Path(args.output) if args.output else None,
        report_path=Path(args.report) if args.report else None,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
