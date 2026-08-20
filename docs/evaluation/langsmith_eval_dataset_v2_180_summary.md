# LangSmith 评测数据集摘要

- 数据集版本：`langsmith_eval_v2_180`
- 样本总数：`180`
- 输出文件：`/Users/elon2ge/workspace/HardWare-Medicial/backend/data/eval/langsmith_eval_dataset_v2_180.jsonl`
- 样本粒度：每行一个 Query
- 构造方式：从 48 条医学证据反向构造检索题，并补充开放域、负样本与路由专项题
- 分类分布：
  - `multi_hop`: 42（23%）
  - `negative`: 30（17%）
  - `open_domain`: 30（17%）
  - `routing`: 30（17%）
  - `single_hop`: 48（27%）

- 医学科室覆盖：
  - `dermatology`: 12
  - `ent`: 12
  - `general_medical`: 11
  - `general_surgery`: 11
  - `infectious_disease`: 11
  - `neurology`: 11
  - `ophthalmology`: 11
  - `pediatrics`: 11

## 字段说明

- `question`：评测输入问题
- `category`：`single_hop` / `multi_hop` / `open_domain` / `negative` / `routing`
- `should_use_rag`：期望路由是否进入 RAG
- `expected_behavior`：回答行为预期，用于安全和开放域评估
- `expected_keywords`：期望答案或证据中应覆盖的关键词
- `expected_sources`：由原始 RAG 数据集反推的来源信息
- 路由专项字段：`expected_domain` / `expected_use_rag` / `expected_web_search` / `expected_safety_level`

## 使用边界

- 单跳/多跳问题由证据反向构造，适合版本回归与消融，不代表真实线上问题分布。
- 开放域和负样本用于全链路路由/安全评测，不纳入纯检索 Recall@K 与 MRR。
- 30 条路由专项题单独评估本地 RAG、时效性 Web 搜索和非医疗分流，不纳入 RAG 消融。
- 数据集发布前仍建议由医学背景人员复核问题自然度和答案时效性。
