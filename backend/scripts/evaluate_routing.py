# isort: skip_file
"""Evaluate deterministic routing decisions on the dedicated routing split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence
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

import app.services.chat_service  # noqa: E402,F401
from app.agents.executor import _decide_web_search  # noqa: E402
from app.agents.planner import HealthConciergeAgent  # noqa: E402
from app.core.langsmith_service import (  # noqa: E402
    configure_langsmith,
    is_langsmith_enabled,
    langsmith_traceable,
)
from app.core.state import initialize_conversation_state  # noqa: E402
from build_langsmith_eval_dataset import (  # noqa: E402
    DATASET_VERSION,
    DEFAULT_OUTPUT_PATH as DEFAULT_DATASET_PATH,
    load_jsonl,
)

DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "data" / "eval" / "routing_eval_result_v2.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "docs" / "evaluation" / "routing_eval_results_v2.csv"


@langsmith_traceable("eval.routing", run_type="chain")
def evaluate_route(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate routing without calling an LLM or executing an external web search."""
    state = initialize_conversation_state()
    state.update(
        {
            "tenant_id": "eval",
            "user_id": "routing_eval",
            "session_id": f"routing-{sample.get('id', '')}",
            "question": str(sample.get("question") or ""),
        }
    )
    with patch("app.agents.planner.get_light_llm", return_value=None):
        result = HealthConciergeAgent(state)

    actual_safety = str(result.get("safety_level") or "")
    if actual_safety == "SAFE":
        actual_web, _ = _decide_web_search(result)
    else:
        actual_web = False

    expected_domain = str(sample.get("expected_domain") or "")
    expected_use_rag = bool(sample.get("expected_use_rag"))
    expected_web = bool(sample.get("expected_web_search"))
    expected_safety = str(sample.get("expected_safety_level") or "")
    domain_match = str(result.get("domain") or "") == expected_domain
    rag_match = bool(result.get("use_rag")) == expected_use_rag
    web_match = bool(actual_web) == expected_web
    safety_match = actual_safety == expected_safety

    return {
        "id": sample.get("id", ""),
        "routing_type": sample.get("routing_type", ""),
        "question": sample.get("question", ""),
        "expected_domain": expected_domain,
        "actual_domain": result.get("domain", ""),
        "domain_match": domain_match,
        "expected_use_rag": expected_use_rag,
        "actual_use_rag": bool(result.get("use_rag")),
        "rag_match": rag_match,
        "expected_web_search": expected_web,
        "actual_web_search": bool(actual_web),
        "web_match": web_match,
        "expected_safety_level": expected_safety,
        "actual_safety_level": actual_safety,
        "safety_match": safety_match,
        "strict_match": domain_match and rag_match and web_match and safety_match,
        "current_tool": result.get("current_tool", ""),
        "routing_reason": result.get("routing_reason", ""),
    }


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "samples": 0,
            "domain_accuracy": 0.0,
            "rag_accuracy": 0.0,
            "web_accuracy": 0.0,
            "safety_accuracy": 0.0,
            "strict_route_accuracy": 0.0,
        }
    return {
        "samples": len(rows),
        "domain_accuracy": round(mean(bool(row["domain_match"]) for row in rows), 4),
        "rag_accuracy": round(mean(bool(row["rag_match"]) for row in rows), 4),
        "web_accuracy": round(mean(bool(row["web_match"]) for row in rows), 4),
        "safety_accuracy": round(mean(bool(row["safety_match"]) for row in rows), 4),
        "strict_route_accuracy": round(
            mean(bool(row["strict_match"]) for row in rows), 4
        ),
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "routing_type",
        "question",
        "expected_domain",
        "actual_domain",
        "domain_match",
        "expected_use_rag",
        "actual_use_rag",
        "rag_match",
        "expected_web_search",
        "actual_web_search",
        "web_match",
        "expected_safety_level",
        "actual_safety_level",
        "safety_match",
        "strict_match",
        "current_tool",
        "routing_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def run_routing_eval(
    dataset_path: Path,
    *,
    output_path: Path,
    csv_path: Path,
) -> Dict[str, Any]:
    configure_langsmith()
    samples = [
        row for row in load_jsonl(dataset_path) if row.get("category") == "routing"
    ]
    if not samples:
        raise ValueError(f"No routing samples in {dataset_path}")

    results = [evaluate_route(sample) for sample in samples]
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_type[str(row.get("routing_type") or "unknown")].append(row)

    payload = {
        "experiment": "medigenius_routing_eval_v2",
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "mode": "deterministic planner + web-decision; no LLM and no external search",
        "langsmith_tracing_enabled": is_langsmith_enabled(),
        "summary": _summary(results),
        "by_routing_type": {
            route_type: _summary(items) for route_type, items in sorted(by_type.items())
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(results, csv_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MediGenius route accuracy.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    args = parser.parse_args()
    payload = run_routing_eval(
        Path(args.dataset),
        output_path=Path(args.output),
        csv_path=Path(args.csv),
    )
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "by_routing_type": payload["by_routing_type"],
                "langsmith_tracing_enabled": payload["langsmith_tracing_enabled"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
