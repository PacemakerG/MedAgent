"""Tests for all agents — Deep Modular Architecture"""

import os
import sys
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.executor import ExecutorAgent  # noqa: E402
from app.agents.medical_router import MedicalRouterAgent  # noqa: E402
from app.agents.memory import MemoryAgent  # noqa: E402
from app.agents.planner import PlannerAgent  # noqa: E402
from app.agents.query_rewriter import QueryRewriterAgent  # noqa: E402
from app.agents.reranker import RerankerAgent  # noqa: E402
from app.agents.retriever import RetrieverAgent  # noqa: E402
from app.core.state import initialize_conversation_state  # noqa: E402
from app.tools.pdf_loader import split_documents  # noqa: E402


# --- Planner Agent Tests ---
def test_planner_agent_medical():
    state = initialize_conversation_state()
    state["question"] = "我有症状，发烧了，还想问用药问题"
    new_state = PlannerAgent(state)
    assert new_state["current_tool"] == "medical_router"
    assert new_state["domain"] == "medical"
    assert new_state["use_rag"] is True
    assert "safety_level" not in new_state
    assert new_state["flow_trace"] == ["keyword_router"]


def test_planner_agent_general():
    state = initialize_conversation_state()
    state["question"] = "Hello there"
    new_state = PlannerAgent(state)
    assert new_state["current_tool"] == "judge_need_rag"
    assert new_state["domain"] == "general"
    assert new_state["use_rag"] is False


def test_planner_agent_medical_keyword_is_not_safety_shortcut():
    state = initialize_conversation_state()
    state["question"] = "偏头痛通常需要做什么检查？"

    new_state = PlannerAgent(state)

    assert new_state["domain"] == "medical"
    assert new_state["current_tool"] == "medical_router"
    assert "safety_level" not in new_state


def test_planner_agent_recognizes_specialty_medical_entities():
    for question in ("登革热怎么传播？", "青光眼为什么要复查？", "疥疮怎么检测？"):
        state = initialize_conversation_state()
        state["question"] = question

        new_state = PlannerAgent(state)

        assert new_state["domain"] == "medical"
        assert new_state["current_tool"] == "medical_router"


def test_planner_agent_manual_department_override():
    state = initialize_conversation_state()
    state["question"] = "我想了解干眼症平时如何护理"
    state["selected_department"] = "neurology"
    state["selected_department_forced"] = True
    new_state = PlannerAgent(state)
    assert new_state["domain"] == "medical"
    assert new_state["use_rag"] is True
    assert new_state["primary_department"] == "neurology"
    assert new_state["current_tool"] == "query_rewriter"


# --- Retriever Agent Tests ---
def test_retriever_agent_success():
    state = initialize_conversation_state()
    state["question"] = "fever"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "infectious_disease"
    state["department_candidates"] = [{"name": "infectious_disease", "score": 0.9}]
    state["retrieval_query"] = "发热 感染"
    state["department_queries"] = {"infectious_disease": "感染 发热"}

    with patch("app.agents.retriever.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="Fever details " * 10)
        ]
        mock_get_retriever.return_value = mock_retriever

        new_state = RetrieverAgent(state)
        assert new_state["rag_success"] is True
        assert len(new_state["documents"]) > 0
        assert "infectious_disease" in new_state["retrieval_results_by_scope"]


def test_retriever_agent_manual_scope_only():
    state = initialize_conversation_state()
    state["question"] = "头晕怎么处理"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["selected_department"] = "neurology"
    state["selected_department_forced"] = True

    with patch("app.agents.retriever.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="神经系统症状评估 " * 10)
        ]
        mock_get_retriever.return_value = mock_retriever
        new_state = RetrieverAgent(state)

    assert new_state["retrieval_scopes"] == ["neurology"]
    assert new_state["rag_success"] is True
    assert mock_get_retriever.call_args.kwargs["search_kwargs"]["filter"] == {
        "department": "neurology"
    }
    assert new_state["profiling"]["retrieval"]["search_all_departments"] is False


def test_retriever_agent_forced_general_searches_all_medical_departments():
    state = initialize_conversation_state()
    state["question"] = "通用健康建议"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["selected_department"] = "general_medical"
    state["selected_department_forced"] = True
    state["retrieval_queries"] = ["通用健康建议"]

    with (
        patch("app.agents.retriever.KEYWORD_BACKEND", "memory"),
        patch("app.agents.retriever.HYBRID_RETRIEVAL_ENABLED", True),
        patch(
            "app.agents.retriever.keyword_search", return_value=[]
        ) as mock_keyword_search,
        patch("app.agents.retriever.get_retriever") as mock_get_retriever,
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="通用医疗知识 " * 10)
        ]
        mock_get_retriever.return_value = mock_retriever
        new_state = RetrieverAgent(state)

    assert new_state["retrieval_scopes"] == ["general_medical"]
    assert mock_get_retriever.call_args.kwargs["search_kwargs"]["filter"] == {
        "domain": "medical"
    }
    assert mock_keyword_search.call_args.kwargs["search_all_departments"] is True
    assert new_state["profiling"]["retrieval"]["search_all_departments"] is True


def test_retriever_agent_automatic_general_scope_searches_all_medical_pdfs():
    state = initialize_conversation_state()
    state["question"] = "通用健康建议"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.9}]

    with patch("app.agents.retriever.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="通用医疗知识 " * 10)
        ]
        mock_get_retriever.return_value = mock_retriever
        new_state = RetrieverAgent(state)

    assert mock_get_retriever.call_args.kwargs["search_kwargs"]["filter"] == {
        "domain": "medical"
    }
    assert new_state["profiling"]["retrieval"]["search_all_departments"] is True


def test_retriever_agent_automatic_specialty_only_searches_primary_department():
    state = initialize_conversation_state()
    state["question"] = "头晕和视物模糊应该如何判断"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "neurology"
    state["department_candidates"] = [
        {"name": "neurology", "score": 0.9},
        {"name": "ophthalmology", "score": 0.6},
        {"name": "general_medical", "score": 0.4},
    ]

    with patch("app.agents.retriever.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="神经系统症状评估 " * 10)
        ]
        mock_get_retriever.return_value = mock_retriever
        new_state = RetrieverAgent(state)

    assert new_state["retrieval_scopes"] == ["neurology"]
    assert mock_get_retriever.call_args.kwargs["search_kwargs"]["filter"] == {
        "department": "neurology"
    }
    assert new_state["profiling"]["retrieval"]["search_all_departments"] is False


def test_retriever_agent_failure():
    state = initialize_conversation_state()
    state["question"] = "unknown"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.9}]
    state["retrieval_query"] = "贫血"
    state["department_queries"] = {"general_medical": "贫血"}
    with patch("app.agents.retriever.get_retriever") as mock_get:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get.return_value = mock_retriever

        new_state = RetrieverAgent(state)
        assert new_state["rag_success"] is False


def test_retriever_agent_with_elasticsearch_keyword_backend():
    state = initialize_conversation_state()
    state["question"] = "发热是不是感染"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "infectious_disease"
    state["department_candidates"] = [{"name": "infectious_disease", "score": 0.9}]
    state["retrieval_query"] = "发热 感染"
    state["retrieval_queries"] = ["发热 感染"]
    state["department_queries"] = {"infectious_disease": "感染 发热"}

    vector_doc = Document(
        page_content="向量召回结果 " * 12,
        metadata={"chunk_id": "vec-1", "department": "infectious_disease"},
    )
    keyword_doc = Document(
        page_content="关键词召回结果 " * 12,
        metadata={"chunk_id": "kw-1", "department": "infectious_disease"},
    )

    with (
        patch("app.agents.retriever.KEYWORD_BACKEND", "elasticsearch"),
        patch("app.agents.retriever.keyword_backend_available", return_value=True),
        patch("app.agents.retriever.get_retriever") as mock_get_retriever,
        patch("app.agents.retriever.keyword_search_es", return_value=[keyword_doc]),
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [vector_doc]
        mock_get_retriever.return_value = mock_retriever

        new_state = RetrieverAgent(state)

    methods = {item["retrieval_method"] for item in new_state["merged_rag_context"]}
    assert new_state["rag_success"] is True
    assert {"vector", "keyword"} <= methods
    assert new_state["profiling"]["retrieval"]["keyword_backend"] == "elasticsearch"


def test_retriever_agent_no_tool():
    state = initialize_conversation_state()
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.9}]
    with patch("app.agents.retriever.get_retriever", return_value=None):
        new_state = RetrieverAgent(state)
        assert new_state["rag_success"] is False


def test_split_documents_assigns_stable_chunk_ids():
    docs = [
        Document(
            page_content="第一章 总论\n这里是测试内容。" * 40,
            metadata={
                "source": "demo.epub",
                "source_path": "demo.epub",
                "page": 1,
                "section": "chapter-1",
            },
        )
    ]

    chunks_a = split_documents(docs)
    chunks_b = split_documents(docs)

    ids_a = [chunk.metadata.get("chunk_id") for chunk in chunks_a]
    ids_b = [chunk.metadata.get("chunk_id") for chunk in chunks_b]
    assert ids_a
    assert ids_a == ids_b


def test_medical_router_agent_fallback():
    state = initialize_conversation_state()
    state["question"] = "贫血通常需要做哪些检查"
    state["domain"] = "medical"
    state["use_rag"] = True
    new_state = MedicalRouterAgent(state)
    assert new_state["primary_department"] == "general_medical"
    assert new_state["department_candidates"]


def test_medical_router_explicit_specialty_overrides_llm_general_fallback():
    state = initialize_conversation_state()
    state["question"] = "肾结石复发应该怎样预防？"
    state["domain"] = "medical"
    state["use_rag"] = True
    response = MagicMock()
    response.content = (
        '{"primary_department":"general_medical",'
        '"department_candidates":[{"name":"general_medical","score":0.9}],'
        '"routing_reason":"general"}'
    )
    llm = MagicMock()
    llm.invoke.return_value = response

    with patch("app.agents.medical_router.get_light_llm", return_value=llm):
        new_state = MedicalRouterAgent(state)

    assert new_state["primary_department"] == "general_surgery"


def test_query_rewriter_agent_fallback():
    state = initialize_conversation_state()
    state["question"] = "贫血通常需要做哪些检查"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.9}]
    new_state = QueryRewriterAgent(state)
    assert new_state["retrieval_query"]
    assert "general_medical" in new_state["department_queries"]


def test_query_rewriter_disabled_is_exact_pass_through(monkeypatch):
    monkeypatch.setattr("app.agents.query_rewriter.QUERY_REWRITER_ENABLED", False)
    state = initialize_conversation_state()
    state["question"] = "What evidence supports CKD screening?"
    state["domain"] = "medical"
    state["use_rag"] = True
    state["primary_department"] = "general_medical"
    state["department_candidates"] = [{"name": "general_medical", "score": 0.9}]

    new_state = QueryRewriterAgent(state)

    assert new_state["retrieval_query"] == state["question"]
    assert new_state["retrieval_queries"] == [state["question"]]
    assert new_state["department_multi_queries"] == {
        "general_medical": [state["question"]]
    }


def test_reranker_agent():
    state = initialize_conversation_state()
    state["question"] = "偏头痛会不会导致头晕"
    state["retrieval_query"] = "偏头痛 头晕"
    state["primary_department"] = "neurology"
    state["retrieval_scopes"] = ["neurology", "general_medical"]
    state["merged_rag_context"] = [
        {
            "content": "偏头痛常见症状包括头痛和头晕。",
            "metadata": {"department": "neurology"},
            "scope": "neurology",
            "raw_rank": 0,
        },
        {
            "content": "胃病也可能导致不适。",
            "metadata": {"department": "general_medical"},
            "scope": "general_medical",
            "raw_rank": 1,
        },
    ]
    new_state = RerankerAgent(state)
    assert new_state["rag_context"][0]["scope"] == "neurology"


# --- Memory Agent Tests ---
def test_memory_agent():
    state = initialize_conversation_state()
    state["conversation_history"] = [
        {"role": "user", "content": str(i)} for i in range(25)
    ]

    new_state = MemoryAgent(state)

    assert len(new_state["conversation_history"]) == 20
    assert new_state["conversation_history"][-1]["content"] == "24"


def test_memory_agent_loads_user_preferences():
    state = initialize_conversation_state()
    state["session_id"] = "sess-pref"

    with (
        patch("app.agents.memory.load_profile") as mock_load,
        patch("app.agents.memory.render_profile_as_text") as mock_render,
    ):
        mock_load.return_value = {
            "preferences": {
                "preferred_name": "王女士",
                "communication_style": "concise",
                "detail_level": "brief",
            }
        }
        mock_render.return_value = "mock profile context"
        new_state = MemoryAgent(state)

    assert new_state["memory_context"] == "mock profile context"
    assert new_state["user_preferences"]["preferred_name"] == "王女士"
    assert new_state["user_preferences"]["detail_level"] == "brief"


# --- Executor Agent Tests ---
def test_executor_agent_with_docs():
    state = initialize_conversation_state()
    state["question"] = "What is X?"
    state["rag_context"] = [{"content": "X is Y."}]

    with (
        patch("app.agents.executor.get_llm") as mock_get_llm,
        patch("app.agents.executor._decide_web_search", return_value=(False, "")),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "根据资料，X 大概率与 Y 相关。你最近还有别的症状吗？"
        )
        mock_get_llm.return_value = mock_llm

        new_state = ExecutorAgent(state)

        assert "大概率与 y 相关" in new_state["generation"].lower()
        assert len(new_state["conversation_history"]) == 2  # user + assistant


def test_executor_agent_no_llm():
    state = initialize_conversation_state()
    state["question"] = "test"
    with patch("app.agents.executor.get_llm", return_value=None):
        new_state = ExecutorAgent(state)
        assert "暂时不可用" in new_state["generation"]
        assert "你希望我下一步" in new_state["generation"]


def test_executor_agent_no_llm_with_preferred_name():
    state = initialize_conversation_state()
    state["question"] = "test"
    state["user_preferences"] = {"preferred_name": "王女士"}
    with patch("app.agents.executor.get_llm", return_value=None):
        new_state = ExecutorAgent(state)
        assert "王女士，你希望我下一步" in new_state["generation"]


def test_executor_agent_llm_fail():
    state = initialize_conversation_state()
    state["question"] = "test"
    state["documents"] = [Document(page_content="some content")]
    with patch("app.agents.executor.get_llm") as mock_get:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("error")
        mock_get.return_value = mock_llm
        new_state = ExecutorAgent(state)
        assert "咨询线下医生" in new_state["generation"]
        assert "你希望我下一步" in new_state["generation"]


def test_executor_agent_includes_personalization_guidance_in_prompt():
    state = initialize_conversation_state()
    state["question"] = "最近头痛怎么办"
    state["user_preferences"] = {
        "preferred_name": "李先生",
        "communication_style": "professional",
        "detail_level": "detailed",
        "language": "en-US",
    }

    with (
        patch("app.agents.executor.get_llm") as mock_get_llm,
        patch("app.agents.executor._decide_web_search", return_value=(False, "")),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "请先观察一周并记录症状变化。你最近有发热吗？"
        )
        mock_get_llm.return_value = mock_llm

        ExecutorAgent(state)
        prompt = mock_llm.invoke.call_args[0][0]

    assert "偏好称呼：优先称呼用户为“李先生”" in prompt
    assert "表达风格：更偏专业与严谨" in prompt
    assert "详略偏好：适度展开机制解释" in prompt
    assert "主体仍用简体中文" in prompt


def test_executor_ecg_skill_shortcut():
    state = initialize_conversation_state()
    state["session_id"] = "session-ecg-1"
    state["question"] = (
        "请根据以下ECG数据生成报告：```json"
        '{"patient_info":{"age":24,"gender":"female"},'
        '"features":{"heart_rate":74}}```'
    )
    with patch("app.agents.executor._maybe_run_ecg_skill") as mock_skill:
        mock_skill.return_value = MagicMock(
            report="**心电图诊断报告**\\n\\n**建议**\\n1. 复查",
            risk_level="low",
            disclaimer="仅供参考",
        )
        new_state = ExecutorAgent(state)
        mock_skill.assert_called_once_with(
            state["question"],
            "session-ecg-1",
            user_id="anonymous",
        )
        assert "心电图诊断报告" in new_state["generation"]
        assert new_state["source"] == "ECG Report Skill"
