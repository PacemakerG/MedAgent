import json
from pathlib import Path

from scripts.build_langsmith_eval_dataset import (
    DATASET_VERSION,
    build_dataset,
    load_jsonl,
    validate_dataset,
    write_jsonl,
)
from scripts.evaluate_routing import evaluate_route
from scripts.run_langsmith_eval import run_single_sample
from scripts.upload_langsmith_dataset import upload_dataset


def _source_row(idx: int, department: str = "dermatology") -> dict:
    return {
        "id": f"{department}_{idx:03d}",
        "question": f"{department} 测试问题 {idx} 怎么处理？",
        "selected_department": department,
        "expected_department": department,
        "expected_source_book": "皮肤性病学",
        "expected_anchor_text": f"测试锚点 {idx}",
        "expected_keywords": [f"关键词{idx}", "头孢曲松"],
        "reference_answer": f"参考答案 {idx}，包含 关键词{idx} 和头孢曲松。",
        "source_path": f"/tmp/source-{idx}.epub",
        "source_book": "皮肤性病学",
        "page": idx,
        "department_display_name": "皮肤科",
        "dataset_source": "unit_test",
    }


def test_build_langsmith_eval_dataset_balances_categories(tmp_path: Path):
    source_path = tmp_path / "source.jsonl"
    rows = [_source_row(1), _source_row(2), _source_row(3), _source_row(4)]
    write_jsonl(rows, source_path)

    dataset = build_dataset(
        [source_path],
        single_hop_count=3,
        multi_hop_count=2,
        open_domain_count=2,
        negative_count=2,
        routing_count=3,
    )

    categories = [item["category"] for item in dataset]
    assert categories.count("single_hop") == 3
    assert categories.count("multi_hop") == 2
    assert categories.count("open_domain") == 2
    assert categories.count("negative") == 2
    assert categories.count("routing") == 3
    assert all(item["question"] for item in dataset)
    assert all("expected_behavior" in item for item in dataset)
    assert all("metadata" in item for item in dataset)

    multi_hop = [item for item in dataset if item["category"] == "multi_hop"][0]
    assert multi_hop["should_use_rag"] is True
    assert len(multi_hop["expected_sources"]) == 2


def test_build_180_row_dataset_passes_quality_gate(tmp_path: Path):
    source_path = tmp_path / "source.jsonl"
    departments = ("dermatology", "ent", "neurology", "ophthalmology")
    rows = [
        _source_row(idx, department=departments[(idx - 1) % len(departments)])
        for idx in range(1, 49)
    ]
    write_jsonl(rows, source_path)

    dataset = build_dataset(
        [source_path],
        single_hop_count=48,
        multi_hop_count=42,
        open_domain_count=30,
        negative_count=30,
        routing_count=30,
    )
    quality = validate_dataset(
        dataset,
        expected_counts={
            "single_hop": 48,
            "multi_hop": 42,
            "open_domain": 30,
            "negative": 30,
            "routing": 30,
        },
    )

    assert len(dataset) == 180
    assert quality["passed"] is True
    assert quality["unique_id_count"] == 180
    assert quality["unique_question_count"] == 180


def test_langsmith_upload_skips_without_real_api_key(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(
        [
            {
                "id": "sample-1",
                "category": "open_domain",
                "question": "BMI 是什么？",
                "selected_department": "",
                "reference_answer": "BMI 是体重指数。",
                "expected_behavior": "answer_general_safely",
                "expected_keywords": ["BMI"],
                "should_use_rag": False,
                "metadata": {},
            }
        ],
        dataset_path,
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "your-langsmith-key")

    status = upload_dataset(dataset_path, dataset_name="unit-test-dataset")

    assert status["uploaded"] is False
    assert status["skipped"] is True
    assert status["samples"] == 1


class _FakeWorkflow:
    def invoke(self, state, config=None):
        assert config["metadata"]["sample_id"] == "ls_single_hop_001"
        result = dict(state)
        result.update(
            {
                "generation": "根据资料，淋菌性咽炎可涉及头孢曲松和左氧氟沙星方案。你希望我下一步重点帮你看哪一部分？",
                "source": "Dermatology Knowledge Base",
                "use_rag": True,
                "rag_attempted": True,
                "rag_success": True,
                "llm_success": True,
                "domain": "medical",
                "safety_level": "SAFE",
                "flow_trace": [
                    "memory_read",
                    "health_concierge",
                    "query_rewriter",
                    "rag",
                    "reranker",
                    "executor",
                    "memory_write_async",
                ],
                "rag_context": [
                    {
                        "content": "淋菌性咽炎 头孢曲松 左氧氟沙星 测试锚点",
                        "metadata": {
                            "department": "dermatology",
                            "source_book": "皮肤性病学",
                            "page": 12,
                        },
                    }
                ],
            }
        )
        return result


def test_run_single_sample_scores_fake_workflow():
    sample = {
        "id": "ls_single_hop_001",
        "category": "single_hop",
        "question": "淋菌性咽炎常用哪些治疗方案？",
        "selected_department": "dermatology",
        "expected_department": "dermatology",
        "expected_source_book": "皮肤性病学",
        "expected_anchor_text": "测试锚点",
        "expected_keywords": ["淋菌性咽炎", "头孢曲松", "左氧氟沙星"],
        "reference_answer": "参考答案",
        "expected_behavior": "grounded_medical_answer",
        "should_use_rag": True,
        "dataset_version": DATASET_VERSION,
        "metadata": {},
    }

    item = run_single_sample(
        sample,
        workflow_app=_FakeWorkflow(),
        top_k=5,
        with_judge=False,
    )

    assert item["route_match"] is True
    assert item["top1_hit"] == 1
    assert item["recall_hit"] == 1
    assert item["behavior_pass"] is True
    assert item["retrieved_context_count"] == 1


def test_evaluate_route_scores_non_medical_probe_without_llm():
    item = evaluate_route(
        {
            "id": "ls_routing_test",
            "routing_type": "non_medical",
            "question": "请用 Python 写一个冒泡排序示例。",
            "expected_domain": "general",
            "expected_use_rag": False,
            "expected_web_search": False,
            "expected_safety_level": "SAFE",
        }
    )

    assert item["actual_domain"] == "general"
    assert item["strict_match"] is True


def test_load_jsonl_rejects_invalid_json(tmp_path: Path):
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text(json.dumps({"ok": True}) + "\n{bad", encoding="utf-8")

    try:
        load_jsonl(dataset_path)
    except ValueError as exc:
        assert "Invalid JSONL" in str(exc)
    else:
        raise AssertionError("load_jsonl should reject malformed JSONL")
