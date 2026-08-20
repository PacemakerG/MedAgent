"""
MediGenius — agents/planner.py
KeywordRouterAgent: binary medical-intent routing before department routing.
"""

from app.core.langsmith_service import langsmith_traceable
from app.core.logging_config import logger
from app.core.medical_taxonomy import (
    DEPARTMENT_TAXONOMY,
    GENERAL_MEDICAL_DEPARTMENT,
    department_display_name,
)
from app.core.state import AgentState, append_flow_trace, profile_node

CORE_MEDICAL_KEYWORDS = (
    "医学",
    "医疗",
    "疾病",
    "症状",
    "诊断",
    "治疗",
    "用药",
    "药物",
    "处方",
    "剂量",
    "副作用",
    "检查",
    "化验",
    "手术",
    "康复",
    "预防",
    "疫苗",
    "传染",
    "患者",
    "医生",
    "医院",
    "发烧",
    "发热",
    "疼痛",
    "头痛",
    "咳嗽",
    "流感",
    "感染",
    "失眠",
    "medical",
    "disease",
    "symptom",
    "diagnosis",
    "treatment",
    "medication",
    "medicine",
    "prescription",
    "dosage",
    "side effect",
    "surgery",
    "doctor",
    "hospital",
)

TAXONOMY_MEDICAL_KEYWORDS = tuple(
    dict.fromkeys(
        str(keyword).lower()
        for info in DEPARTMENT_TAXONOMY.values()
        for keyword in info.get("keywords", [])
    )
)


def _is_medical_question(question: str) -> bool:
    normalized = (question or "").strip().lower()
    return any(keyword in normalized for keyword in CORE_MEDICAL_KEYWORDS) or any(
        keyword in normalized for keyword in TAXONOMY_MEDICAL_KEYWORDS
    )


@langsmith_traceable("keyword_router")
def KeywordRouterAgent(state: AgentState) -> AgentState:
    """Classify the query as medical or non-medical using deterministic keywords."""
    append_flow_trace(state, "keyword_router")
    with profile_node(state, "keyword_router"):
        state["keyword_hit"] = False
        state["domain"] = "general"
        state["use_rag"] = False
        state["need_rag"] = False
        state["current_tool"] = "judge_need_rag"
        state["retry_count"] = 0
        state["primary_department"] = None
        state["department_candidates"] = []
        state["department_queries"] = {}
        state["retrieval_scopes"] = []
        state["routing_reason"] = "non-medical keyword route"
        state["rewrite_reason"] = ""

        selected_department = state.get("selected_department")
        if state.get("selected_department_forced") and selected_department:
            state["keyword_hit"] = True
            state["domain"] = "medical"
            state["use_rag"] = True
            state["primary_department"] = selected_department
            state["department_candidates"] = [
                {
                    "name": selected_department,
                    "score": 1.0,
                    "display_name": department_display_name(selected_department),
                }
            ]
            state["routing_reason"] = (
                "manual department override"
                if selected_department != GENERAL_MEDICAL_DEPARTMENT
                else "manual general-medical override"
            )
            state["current_tool"] = "query_rewriter"
            return state

        is_medical = _is_medical_question(state.get("question") or "")
        state["keyword_hit"] = is_medical
        state["domain"] = "medical" if is_medical else "general"
        state["use_rag"] = is_medical
        state["current_tool"] = "medical_router" if is_medical else "judge_need_rag"
        state["routing_reason"] = (
            "medical keyword route" if is_medical else "non-medical keyword route"
        )
        logger.info(
            "KeywordRouter: is_medical=%s, next=%s",
            is_medical,
            state["current_tool"],
        )
    return state


PlannerAgent = KeywordRouterAgent
