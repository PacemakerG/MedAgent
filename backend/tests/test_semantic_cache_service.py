from app.services.semantic_cache_service import semantic_cache_service


def test_semantic_cache_uses_structured_fingerprint_not_raw_query():
    lookup = semantic_cache_service.build_lookup(
        query="高血压能不能喝咖啡？",
        tenant_id="default",
    )

    assert lookup.eligible is True
    assert "高血压" not in lookup.answer_key
    assert "咖啡" not in lookup.answer_key
    assert len(lookup.fingerprint) == 64


def test_semantically_equivalent_yes_no_queries_share_cache_key():
    first = semantic_cache_service.build_lookup(
        query="高血压能不能喝咖啡？",
        tenant_id="default",
    )
    second = semantic_cache_service.build_lookup(
        query="血压高的人可以喝咖啡吗？",
        tenant_id="default",
    )

    assert first.eligible is True
    assert second.eligible is True
    assert first.metadata["negation"] is False
    assert second.metadata["negation"] is False
    assert first.answer_key == second.answer_key


def test_true_negative_query_uses_different_cache_key():
    positive = semantic_cache_service.build_lookup(
        query="高血压能不能喝咖啡？",
        tenant_id="default",
    )
    negative = semantic_cache_service.build_lookup(
        query="高血压不能喝咖啡吗？",
        tenant_id="default",
    )

    assert positive.eligible is True
    assert negative.eligible is True
    assert positive.metadata["negation"] is False
    assert negative.metadata["negation"] is True
    assert positive.answer_key != negative.answer_key


def test_semantic_cache_rejects_high_risk_query():
    lookup = semantic_cache_service.build_lookup(
        query="胸痛伴呼吸困难怎么办？",
        tenant_id="default",
    )

    assert lookup.eligible is False
    assert lookup.reason == "risk_not_low"


def test_semantic_cache_round_trip():
    lookup = semantic_cache_service.build_lookup(
        query="血压高的人可以喝咖啡吗？",
        tenant_id="default",
    )

    semantic_cache_service.store_answer(
        lookup,
        answer="可以少量饮用，但要观察血压变化。",
        source="Test",
        flow_trace=["semantic_cache_test"],
    )
    cached = semantic_cache_service.get_answer(lookup)

    assert cached["answer"] == "可以少量饮用，但要观察血压变化。"
    assert cached["cache_hit"] is True
