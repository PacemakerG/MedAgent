"""
MediGenius — core/medical_taxonomy.py
Medical department taxonomy, folder naming, and heuristic helpers.
"""

from __future__ import annotations

import re
from typing import Dict, List

GENERAL_MEDICAL_DEPARTMENT = "general_medical"

DEPARTMENT_TAXONOMY: Dict[str, Dict[str, object]] = {
    GENERAL_MEDICAL_DEPARTMENT: {
        "zh": "通用医疗",
        "aliases": [
            "general_medical",
            "general-medical",
            "generalmedical",
            "通用医疗",
            "hematology",
            "blood",
            "血液科",
            "cardiology",
            "cardio",
            "心内科",
            "respiratory",
            "pulmonary",
            "呼吸内科",
            "gastroenterology",
            "gastro",
            "消化内科",
            "endocrinology",
            "endo",
            "内分泌科",
            "nephrology",
            "renal",
            "肾内科",
            "rheumatology",
            "rheum",
            "风湿免疫科",
            "gynecology",
            "gyn",
            "妇科",
            "obstetrics",
            "ob",
            "产科",
            "oncology",
            "onco",
            "肿瘤科",
        ],
        "keywords": [
            "症状",
            "检查",
            "治疗",
            "用药",
            "化验",
            "看什么科",
            "贫血",
            "血红蛋白",
            "白细胞",
            "血小板",
            "淋巴",
            "骨髓",
            "凝血",
            "胸痛",
            "心慌",
            "心悸",
            "心率",
            "心电",
            "胸闷",
            "高血压",
            "咳嗽",
            "咳痰",
            "气短",
            "呼吸困难",
            "肺",
            "哮喘",
            "发绀",
            "腹痛",
            "腹泻",
            "恶心",
            "呕吐",
            "胃",
            "肠",
            "便血",
            "血糖",
            "糖尿病",
            "甲状腺",
            "激素",
            "多饮",
            "体重下降",
            "水肿",
            "尿蛋白",
            "肌酐",
            "肾",
            "尿少",
            "血尿",
            "关节痛",
            "免疫",
            "风湿",
            "晨僵",
            "狼疮",
            "月经",
            "白带",
            "阴道",
            "盆腔",
            "子宫",
            "卵巢",
            "怀孕",
            "孕",
            "胎动",
            "产检",
            "宫缩",
            "分娩",
            "肿瘤",
            "癌",
            "化疗",
            "放疗",
            "结节",
            "肿块",
            "慢性病",
            "慢性呼吸",
            "心血管",
            "癌症筛查",
        ],
    },
    "general_surgery": {
        "zh": "普外科",
        "aliases": [
            "general_surgery",
            "surgery",
            "普外科",
            "orthopedics",
            "ortho",
            "骨科",
            "urology",
            "uro",
            "泌尿外科",
            "emergency",
            "er",
            "急诊科",
        ],
        "keywords": [
            "阑尾",
            "疝气",
            "外伤",
            "伤口",
            "包块",
            "手术",
            "骨折",
            "腰痛",
            "关节",
            "膝盖",
            "颈椎",
            "扭伤",
            "尿频",
            "尿急",
            "排尿",
            "前列腺",
            "肾结石",
            "尿痛",
            "昏迷",
            "晕厥",
            "剧痛",
            "大出血",
            "休克",
            "急救",
        ],
    },
    "pediatrics": {
        "zh": "儿科",
        "aliases": ["pediatrics", "peds", "儿科"],
        "keywords": ["儿童", "宝宝", "小孩", "婴儿", "生长发育", "奶量"],
    },
    "neurology": {
        "zh": "神经内科",
        "aliases": ["neurology", "neuro", "神经内科"],
        "keywords": [
            "头晕",
            "头痛",
            "麻木",
            "抽搐",
            "意识",
            "中风",
            "偏瘫",
            "癫痫",
            "神经系统",
        ],
    },
    "infectious_disease": {
        "zh": "感染科",
        "aliases": ["infectious_disease", "infection", "感染科"],
        "keywords": [
            "感染",
            "发热",
            "病毒",
            "细菌",
            "传染",
            "乙肝",
            "肝炎",
            "hepatitis",
            "登革热",
            "霍乱",
            "疟疾",
            "疫情",
            "暴发",
        ],
    },
    "ent": {
        "zh": "耳鼻喉科",
        "aliases": ["ent", "otolaryngology", "耳鼻喉科"],
        "keywords": [
            "咽痛",
            "鼻塞",
            "耳鸣",
            "扁桃体",
            "眩晕",
            "流鼻血",
            "听力",
            "耳机",
            "耳蜗",
            "耳聋",
        ],
    },
    "ophthalmology": {
        "zh": "眼科",
        "aliases": ["ophthalmology", "ophthal", "眼科"],
        "keywords": [
            "视力",
            "眼痛",
            "红眼",
            "飞蚊",
            "流泪",
            "畏光",
            "青光眼",
            "近视",
            "屈光",
            "白内障",
        ],
    },
    "dermatology": {
        "zh": "皮肤科",
        "aliases": ["dermatology", "derm", "皮肤科"],
        "keywords": [
            "皮疹",
            "瘙痒",
            "红斑",
            "湿疹",
            "痤疮",
            "脱发",
            "疥疮",
            "疥螨",
            "皮肤",
        ],
    },
}


def department_display_name(code: str) -> str:
    info = DEPARTMENT_TAXONOMY.get(code, {})
    return str(info.get("zh") or code)


def department_folder_name(code: str) -> str:
    return f"{code}_{department_display_name(code)}"


def list_department_codes() -> List[str]:
    return list(DEPARTMENT_TAXONOMY.keys())


def normalize_department_code(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if normalized in DEPARTMENT_TAXONOMY:
        return normalized

    for code in DEPARTMENT_TAXONOMY:
        if normalized.startswith(f"{code}_"):
            return code

    parts = [part for part in re.split(r"[_/]", normalized) if part]
    if parts and parts[0] in DEPARTMENT_TAXONOMY:
        return parts[0]

    for code, info in DEPARTMENT_TAXONOMY.items():
        aliases = {str(alias).lower() for alias in info.get("aliases", [])}
        aliases.add(code)
        if normalized in aliases:
            return code
        if any(part in aliases for part in parts):
            return code
    return None


def infer_department_candidates(question: str, top_k: int = 3) -> List[dict]:
    q = (question or "").lower()
    scored: List[tuple[str, int]] = []
    for code, info in DEPARTMENT_TAXONOMY.items():
        if code == GENERAL_MEDICAL_DEPARTMENT:
            continue
        score = sum(
            1 for keyword in info.get("keywords", []) if str(keyword).lower() in q
        )
        if score > 0:
            scored.append((code, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    candidates = [
        {
            "name": code,
            "score": min(0.99, 0.45 + score * 0.12),
            "display_name": department_display_name(code),
        }
        for code, score in scored[:top_k]
    ]
    if not candidates:
        candidates.append(
            {
                "name": GENERAL_MEDICAL_DEPARTMENT,
                "score": 0.4,
                "display_name": department_display_name(GENERAL_MEDICAL_DEPARTMENT),
            }
        )
    return candidates


def extract_query_terms(text: str) -> List[str]:
    if not text:
        return []
    terms: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text.lower()):
        cleaned = token.strip()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        terms.append(cleaned)
    return terms
