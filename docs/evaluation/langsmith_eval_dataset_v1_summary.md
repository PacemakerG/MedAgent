# LangSmith 评测数据集摘要

- 数据集版本：`langsmith_eval_v1`
- 样本总数：`50`
- 输出文件：`backend/data/eval/langsmith_eval_dataset_v1.jsonl`
- 分类分布：
  - `multi_hop`: 14
  - `negative`: 10
  - `open_domain`: 10
  - `single_hop`: 16

- 医学科室覆盖：
  - `dermatology`: 5
  - `ent`: 4
  - `general_medical`: 4
  - `general_surgery`: 2
  - `infectious_disease`: 4
  - `neurology`: 4
  - `ophthalmology`: 4
  - `pediatrics`: 3

## 字段说明

- `question`：评测输入问题
- `category`：`single_hop` / `multi_hop` / `open_domain` / `negative`
- `should_use_rag`：期望路由是否进入 RAG
- `expected_behavior`：回答行为预期，用于安全和开放域评估
- `expected_keywords`：期望答案或证据中应覆盖的关键词
- `expected_sources`：由原始 RAG 数据集反推的来源信息
