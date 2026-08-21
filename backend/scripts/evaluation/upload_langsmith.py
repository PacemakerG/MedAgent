"""Upload the RAG, routing, and Redis datasets as three LangSmith datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
EVAL_ROOT = BACKEND_ROOT / "data" / "eval"

DATASET_PATHS = {
    "rag": EVAL_ROOT / "rag" / "dataset_150.json",
    "routing": EVAL_ROOT / "routing" / "dataset_50.json",
    "redis": EVAL_ROOT / "redis" / "dataset_50.json",
}


def _real_key(value: str | None) -> bool:
    text = (value or "").strip().lower()
    return bool(text) and not any(
        marker in text for marker in ("your-", "replace-", "changeme", "example")
    )


def configure_environment() -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not _real_key(key):
        raise RuntimeError("A real LANGSMITH_API_KEY is required")
    os.environ["LANGSMITH_API_KEY"] = key or ""
    os.environ.setdefault("LANGCHAIN_API_KEY", key or "")


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("metadata") or not payload.get("samples"):
        raise ValueError(f"Invalid dataset: {path}")
    return payload


def example_payload(kind: str, sample: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "sample_id": sample["id"],
        "category": sample.get("category", ""),
        "dataset_kind": kind,
    }
    if kind == "rag":
        return {
            "inputs": {
                "question": sample["question"],
                "department": sample["department"],
            },
            "outputs": {
                "gold_evidence": sample["gold_evidence"],
                "reference_answer": sample["reference_answer"],
            },
            "metadata": metadata,
        }
    if kind == "routing":
        return {
            "inputs": {"question": sample["question"]},
            "outputs": {
                "expected_route": sample["expected_route"],
                "expected_department": sample.get("expected_department"),
            },
            "metadata": metadata,
        }
    return {
        "inputs": {
            "cached_question": sample["cached_question"],
            "probe_question": sample["probe_question"],
        },
        "outputs": {
            "cached_answer": sample["cached_answer"],
            "expected_hit": sample["expected_hit"],
        },
        "metadata": metadata,
    }


def get_or_create_dataset(client, payload: dict[str, Any]):
    name = payload["metadata"]["name"]
    if client.has_dataset(dataset_name=name):
        return client.read_dataset(dataset_name=name)
    return client.create_dataset(
        dataset_name=name,
        description=f"MediGenius {name} reproducible evaluation dataset",
        metadata=payload["metadata"],
    )


def upload_one(client, kind: str, path: Path) -> dict[str, Any]:
    payload = load_dataset(path)
    dataset = get_or_create_dataset(client, payload)
    existing = {}
    for example in client.list_examples(dataset_id=dataset.id):
        metadata = getattr(example, "metadata", None) or {}
        sample_id = metadata.get("sample_id")
        if sample_id:
            existing[str(sample_id)] = example

    created = 0
    updated = 0
    for sample in payload["samples"]:
        row = example_payload(kind, sample)
        current = existing.get(sample["id"])
        if current is None:
            client.create_example(dataset_id=dataset.id, **row)
            created += 1
        else:
            client.update_example(
                current.id,
                inputs=row["inputs"],
                outputs=row["outputs"],
                metadata=row["metadata"],
            )
            updated += 1
    return {
        "kind": kind,
        "name": payload["metadata"]["name"],
        "dataset_id": str(dataset.id),
        "samples": len(payload["samples"]),
        "created": created,
        "updated": updated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("rag", "routing", "redis"),
        help="Upload one dataset instead of all three",
    )
    args = parser.parse_args()

    configure_environment()
    from langsmith import Client

    client = Client()
    selected = {args.only: DATASET_PATHS[args.only]} if args.only else DATASET_PATHS
    statuses = [upload_one(client, kind, path) for kind, path in selected.items()]
    print(json.dumps({"uploaded": True, "datasets": statuses}, ensure_ascii=False))


if __name__ == "__main__":
    main()
