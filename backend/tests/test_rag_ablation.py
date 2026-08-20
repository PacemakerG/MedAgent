from scripts.run_rag_ablation import (
    Chunk,
    clean_ocr_text,
    fixed_chunks,
    rrf_fuse,
    score_ranking,
    semantic_chunks,
    unique_source_ranking,
)


def test_chunking_and_ocr_cleanup_are_deterministic():
    raw = "患 者 体 温 39.O 度。需要观察呼吸；必要时就医。"

    assert "患者体温" in clean_ocr_text(raw)
    assert fixed_chunks(raw) == fixed_chunks(raw)
    assert semantic_chunks(raw) == ["患 者 体 温 39.O 度。 需要观察呼吸； 必要时就医。"]


def test_parent_child_ranking_removes_duplicate_sources():
    chunks = [
        Chunk("a:1", "a", "d", "one", "parent a"),
        Chunk("a:2", "a", "d", "two", "parent a"),
        Chunk("b:1", "b", "d", "three", "parent b"),
    ]

    assert unique_source_ranking(chunks, top_k=2) == ["a", "b"]


def test_rrf_uses_rank_not_backend_score():
    ranking = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], top_k=4, rrf_k=60)

    assert ranking[:2] == ["b", "c"]
    assert set(ranking) == {"a", "b", "c", "d"}


def test_multi_hop_metrics_use_fractional_source_recall():
    metrics = score_ranking(["source-a", "other"], ["source-a", "source-b"], top_k=5)

    assert metrics["top1"] == 1.0
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr"] == 1.0
    assert metrics["complete_at_k"] == 0.0
