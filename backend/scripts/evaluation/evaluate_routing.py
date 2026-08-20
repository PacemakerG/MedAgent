"""Evaluate the 50 routing samples through the compiled LangGraph workflow."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import KNOWLEDGE_ROOT_DIR  # noqa: E402
from app.core.langgraph_workflow import create_workflow  # noqa: E402
from app.core.langsmith_service import build_langsmith_runnable_config  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.tools.es_client import (  # noqa: E402
    bulk_index_documents,
    ensure_es_index,
    es_enabled,
)
from app.tools.llm_client import get_light_llm, get_llm  # noqa: E402
from app.tools.pdf_loader import process_knowledge_library  # noqa: E402
from app.tools.tavily_search import get_tavily_search  # noqa: E402
from app.tools.vector_store import get_or_create_vectorstore  # noqa: E402

DEFAULT_DATASET = BACKEND_ROOT / "data" / "eval" / "routing" / "dataset_50.json"
DEFAULT_OUTPUT = BACKEND_ROOT / "data" / "eval" / "routing" / "result.json"
EVAL_VECTOR_STORE = "/tmp/medagent-chroma-eval"


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("samples") or []) != 50:
        raise ValueError("Routing dataset must contain exactly 50 samples")
    return payload


def preflight() -> None:
    missing = []
    if get_llm(user_id="routing-eval") is None:
        missing.append("main LLM")
    if get_light_llm(user_id="routing-eval") is None:
        missing.append("light routing LLM")
    if get_tavily_search() is None:
        missing.append("Tavily")
    if not es_enabled() or not ensure_es_index():
        missing.append("Elasticsearch")
    if missing:
        raise RuntimeError(
            "Full-chain routing evaluation requires: " + ", ".join(missing)
        )


def prepare_retrieval() -> None:
    documents = process_knowledge_library(KNOWLEDGE_ROOT_DIR)
    vectorstore = get_or_create_vectorstore(
        documents=documents,
        persist_dir=EVAL_VECTOR_STORE,
    )
    if vectorstore is None:
        raise RuntimeError("Vector store initialization failed")
    if not bulk_index_documents(documents):
        raise RuntimeError("Elasticsearch knowledge index initialization failed")


def actual_route(result: dict[str, Any]) -> str:
    if result.get("domain") != "medical":
        return "non_medical"
    tool_calls = result.get("tool_calls") or []
    if any(item.get("tool") == "web_search" for item in tool_calls):
        return "web_search"
    return "local_rag"


def evaluate_sample(workflow, sample: dict[str, Any]) -> dict[str, Any]:
    state = initialize_conversation_state()
    state.update(
        {
            "session_id": f"routing-eval-{sample['id']}",
            "user_id": "routing-eval",
            "question": sample["question"],
        }
    )
    config = build_langsmith_runnable_config(
        operation="evaluation.routing.full_chain",
        session_id=state["session_id"],
        user_id=state["user_id"],
        extra_tags=["evaluation", "routing", "v1"],
        extra_metadata={"sample_id": sample["id"]},
    )
    result = workflow.invoke(state, config=config)
    predicted_route = actual_route(result)
    predicted_department = result.get("primary_department")
    return {
        "id": sample["id"],
        "expected_route": sample["expected_route"],
        "actual_route": predicted_route,
        "route_correct": predicted_route == sample["expected_route"],
        "expected_department": sample.get("expected_department"),
        "actual_department": predicted_department,
        "department_correct": (
            predicted_department == sample.get("expected_department")
            if sample.get("expected_department")
            else None
        ),
        "flow_trace": result.get("flow_trace") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    preflight()
    prepare_retrieval()
    workflow = create_workflow()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        rows = list(
            pool.map(
                lambda sample: evaluate_sample(workflow, sample),
                dataset["samples"],
            )
        )
    medical = [row for row in rows if row["expected_department"]]
    result = {
        "metadata": {
            "dataset": dataset["metadata"]["name"],
            "samples": 50,
            "execution": "compiled LangGraph + live LLM + local RAG + live Elasticsearch + live Tavily",
            "metric_definitions": {
                "route_accuracy": "本地RAG、联网搜索、非医疗三种最终路径判断正确的样本数 / 50",
                "department_accuracy": "40条医疗问题中主科室判断正确的样本数 / 40",
            },
        },
        "metrics": {
            "route_accuracy": round(
                mean(row["route_correct"] for row in rows) * 100, 4
            ),
            "department_accuracy": round(
                mean(row["department_correct"] for row in medical) * 100,
                4,
            ),
        },
        "samples": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
