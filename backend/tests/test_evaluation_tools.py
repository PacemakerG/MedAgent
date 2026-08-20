import pytest

from scripts.evaluation.build_datasets import DEPARTMENTS, build_all
from scripts.evaluation.evaluate_rag import (
    Chunk,
    add_faithfulness_scores,
    decide_parent_child,
    decide_system_tradeoff,
    metric_for_sample,
    rrf_merge,
    select_codex_audit_ids,
)
from scripts.evaluation.evaluate_routing import actual_route
from scripts.evaluation.upload_langsmith import example_payload


@pytest.fixture(scope="module")
def datasets():
    return build_all(validate_only=True)


def test_three_datasets_have_exact_250_unique_samples(datasets):
    assert {name: len(item["samples"]) for name, item in datasets.items()} == {
        "rag": 150,
        "routing": 50,
        "redis": 50,
    }
    ids = [
        sample["id"] for dataset in datasets.values() for sample in dataset["samples"]
    ]
    assert len(ids) == len(set(ids)) == 250


def test_rag_queries_are_all_english(datasets):
    samples = datasets["rag"]["samples"]
    assert all(sample["language"] == "en" for sample in samples)
    assert all(
        not any("\u4e00" <= char <= "\u9fff" for char in sample["question"])
        for sample in samples
    )


def test_codex_audit_selection_is_stratified(datasets):
    samples = datasets["rag"]["samples"]
    selected = set(select_codex_audit_ids(samples))
    assert len(selected) == 40
    counts = {
        category: sum(
            sample["id"] in selected and sample["category"] == category
            for sample in samples
        )
        for category in ("single_hop", "multi_hop", "hard_retrieval")
    }
    assert counts == {"single_hop": 14, "multi_hop": 13, "hard_retrieval": 13}


def test_faithfulness_cache_only_does_not_call_model(tmp_path):
    row = {"id": "rag_single_001", "faithfulness": None, "answer": ""}
    context = Chunk("1", "a.pdf", 1, "ent", "evidence", "evidence")

    add_faithfulness_scores(
        [(row, "question", [context])],
        workers=1,
        cache_dir=tmp_path,
        cache_only=True,
    )

    assert row["faithfulness"] is None
    assert row["answer"] == ""


def test_routing_dataset_has_five_medical_questions_per_department(datasets):
    dataset = datasets["routing"]
    medical = [
        sample
        for sample in dataset["samples"]
        if sample["expected_route"] != "non_medical"
    ]
    assert {
        department: sum(
            sample["expected_department"] == department for sample in medical
        )
        for department in DEPARTMENTS
    } == {department: 5 for department in DEPARTMENTS}


def test_rag_metrics_use_unique_gold_pdf_pages():
    sample = {
        "gold_evidence": [
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 2},
        ]
    }
    results = [
        Chunk("1", "a.pdf", 1, "ent", "a", "a"),
        Chunk("2", "x.pdf", 8, "ent", "x", "x"),
        Chunk("3", "b.pdf", 2, "ent", "b", "b"),
    ]
    metrics = metric_for_sample(sample, results)
    assert metrics == {"hit_at_1": 1.0, "recall_at_5": 1.0, "mrr": 1.0}


def test_rrf_merges_dense_and_keyword_rankings():
    a = Chunk("a", "a.pdf", 1, "ent", "a", "a")
    b = Chunk("b", "b.pdf", 2, "ent", "b", "b")
    c = Chunk("c", "c.pdf", 3, "ent", "c", "c")
    merged = rrf_merge([[a, b], [b, c]], top_k=3)
    assert [item.chunk_id for item in merged] == ["b", "a", "c"]


def test_parent_child_is_discarded_without_meaningful_quality_gain():
    c5 = {
        "hit_at_1": 80.0,
        "recall_at_5": 95.0,
        "mrr": 0.88,
        "average_retrieval_ms": 20.0,
    }
    c6 = {
        "hit_at_1": 80.0,
        "recall_at_5": 95.0,
        "mrr": 0.881,
        "average_retrieval_ms": 27.0,
    }
    assert decide_parent_child(c5, c6)["decision"] == "discard"


def test_system_tradeoff_rejects_recall_gain_that_hurts_ranking():
    cumulative = [
        {
            "name": "C0",
            "hit_at_1": 15.0,
            "recall_at_5": 35.0,
            "mrr": 0.24,
            "average_retrieval_ms": 20.0,
        },
        {
            "name": "C1",
            "hit_at_1": 30.0,
            "recall_at_5": 49.0,
            "mrr": 0.41,
            "average_retrieval_ms": 1000.0,
        },
        {
            "name": "C2",
            "hit_at_1": 27.0,
            "recall_at_5": 55.0,
            "mrr": 0.40,
            "average_retrieval_ms": 7000.0,
        },
    ]
    decision = decide_system_tradeoff(cumulative)
    assert decision["final_variant"] == "C1"
    assert decision["decisions"][-1]["decision"] == "discard"


def test_full_chain_route_is_derived_from_final_state():
    assert actual_route({"domain": "general"}) == "non_medical"
    assert actual_route({"domain": "medical", "tool_calls": []}) == "local_rag"
    assert (
        actual_route({"domain": "medical", "tool_calls": [{"tool": "web_search"}]})
        == "web_search"
    )


def test_langsmith_payloads_keep_each_dataset_contract():
    rag = example_payload(
        "rag",
        {
            "id": "rag_1",
            "category": "single_hop",
            "question": "q",
            "department": "ent",
            "gold_evidence": [{"source": "a.pdf", "page": 1, "text": "e"}],
            "reference_answer": "a",
        },
    )
    cache = example_payload(
        "redis",
        {
            "id": "cache_1",
            "category": "semantic_equivalent",
            "cached_question": "q1",
            "probe_question": "q2",
            "cached_answer": "a",
            "expected_hit": True,
        },
    )
    assert rag["outputs"]["gold_evidence"][0]["page"] == 1
    assert cache["outputs"]["expected_hit"] is True
