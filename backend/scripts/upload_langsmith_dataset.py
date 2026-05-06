"""
Upload the MediGenius evaluation dataset to LangSmith.

This script is intentionally safe to run without LangSmith credentials: when no
real API key is configured, it exits successfully with a skipped status so local
CI and development do not depend on the LangSmith service.

Usage:
  python backend/scripts/upload_langsmith_dataset.py \
    --dataset backend/data/eval/langsmith_eval_dataset_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_langsmith_eval_dataset import DATASET_VERSION, load_jsonl  # noqa: E402

DEFAULT_DATASET_PATH = BACKEND_ROOT / "data" / "eval" / "langsmith_eval_dataset_v1.jsonl"
DEFAULT_LANGSMITH_DATASET_NAME = "medigenius-rag-eval-v1"


def _load_backend_env() -> None:
    load_dotenv(BACKEND_ROOT / ".env")


def _is_real_api_key(value: Optional[str]) -> bool:
    if not value:
        return False
    stripped = value.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    placeholders = ("your-", "replace-", "changeme", "todo", "example")
    return not any(item in lowered for item in placeholders)


def _langsmith_env_ready() -> bool:
    _load_backend_env()
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not _is_real_api_key(api_key):
        return False

    os.environ.setdefault("LANGCHAIN_API_KEY", api_key or "")
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
    return True


def _get_or_create_dataset(client: Any, dataset_name: str, description: str) -> Any:
    try:
        if client.has_dataset(dataset_name=dataset_name):
            return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        pass
    return client.create_dataset(
        dataset_name=dataset_name,
        description=description,
        metadata={"dataset_version": DATASET_VERSION, "application": "MediGenius"},
    )


def _example_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "sample_id": row.get("id", ""),
            "category": row.get("category", ""),
            "dataset_version": row.get("dataset_version", DATASET_VERSION),
            "selected_department": row.get("selected_department", ""),
            "expected_department": row.get("expected_department", ""),
            "expected_source_book": row.get("expected_source_book", ""),
            "should_use_rag": bool(row.get("should_use_rag", False)),
        }
    )
    expected_sources = row.get("expected_sources") or []
    if expected_sources:
        metadata["expected_sources"] = expected_sources
    return metadata


def _example_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inputs": {
            "question": row.get("question", ""),
            "selected_department": row.get("selected_department", ""),
        },
        "outputs": {
            "reference_answer": row.get("reference_answer", ""),
            "expected_behavior": row.get("expected_behavior", ""),
            "expected_keywords": row.get("expected_keywords", []),
            "should_use_rag": bool(row.get("should_use_rag", False)),
        },
        "metadata": _example_metadata(row),
        "split": row.get("category", "default"),
    }


def _existing_sample_ids(client: Any, dataset_id: Any) -> set[str]:
    existing: set[str] = set()
    try:
        for example in client.list_examples(dataset_id=dataset_id):
            metadata = getattr(example, "metadata", None) or {}
            sample_id = metadata.get("sample_id")
            if sample_id:
                existing.add(str(sample_id))
    except Exception:
        return set()
    return existing


def upload_dataset(
    dataset_path: Path,
    *,
    dataset_name: str = DEFAULT_LANGSMITH_DATASET_NAME,
    require_upload: bool = False,
) -> Dict[str, Any]:
    rows = load_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"Dataset empty: {dataset_path}")

    category_counts = Counter(str(row.get("category", "unknown")) for row in rows)
    if not _langsmith_env_ready():
        status = {
            "uploaded": False,
            "skipped": True,
            "reason": "LANGSMITH_API_KEY is not configured with a real key",
            "dataset_path": str(dataset_path),
            "dataset_name": dataset_name,
            "samples": len(rows),
            "category_counts": dict(category_counts),
        }
        if require_upload:
            raise RuntimeError(status["reason"])
        return status

    try:
        from langsmith import Client
    except Exception as exc:
        if require_upload:
            raise RuntimeError(f"langsmith package unavailable: {exc}") from exc
        return {
            "uploaded": False,
            "skipped": True,
            "reason": f"langsmith package unavailable: {exc}",
            "dataset_path": str(dataset_path),
            "dataset_name": dataset_name,
            "samples": len(rows),
            "category_counts": dict(category_counts),
        }

    client = Client()
    dataset = _get_or_create_dataset(
        client,
        dataset_name,
        description=(
            "MediGenius four-category evaluation dataset: single-hop, "
            "multi-hop, open-domain, and negative safety probes."
        ),
    )
    dataset_id = getattr(dataset, "id", None)
    existing_ids = _existing_sample_ids(client, dataset_id)
    created = 0
    skipped_existing = 0

    for row in rows:
        sample_id = str(row.get("id", ""))
        if sample_id and sample_id in existing_ids:
            skipped_existing += 1
            continue
        payload = _example_payload(row)
        client.create_example(dataset_id=dataset_id, **payload)
        created += 1

    return {
        "uploaded": True,
        "skipped": False,
        "dataset_name": dataset_name,
        "dataset_id": str(dataset_id),
        "dataset_path": str(dataset_path),
        "samples": len(rows),
        "created_examples": created,
        "skipped_existing_examples": skipped_existing,
        "category_counts": dict(category_counts),
    }


def write_upload_report(status: Dict[str, Any], output_path: Optional[Path]) -> None:
    if not output_path:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload MediGenius eval dataset to LangSmith.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--name", default=os.getenv("LANGSMITH_EVAL_DATASET", DEFAULT_LANGSMITH_DATASET_NAME))
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--require-upload",
        action="store_true",
        help="Fail when LangSmith credentials are missing instead of skipping.",
    )
    args = parser.parse_args()

    status = upload_dataset(
        Path(args.dataset),
        dataset_name=str(args.name),
        require_upload=bool(args.require_upload),
    )
    write_upload_report(status, Path(args.output) if args.output else None)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
