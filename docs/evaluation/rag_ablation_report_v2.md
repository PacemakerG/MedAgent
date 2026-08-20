# RAG 消融评测报告 v2

## 结论

在 90 条有本地证据的问题上，最终方案“数据清洗 + 语义切分 + 父子索引 + Elasticsearch BM25/向量并行召回 + RRF”将 Recall@5 从 **75.00%** 提升到 **100.00%**，Top1 从 **50.00%** 提升到 **77.78%**。最终方案检索 p95 为 **6.75 ms**；查询向量编码平均 **2.71 ms**，未计入检索延迟。

| 版本 | 单一变量变化 | Top1 | Recall@5 | MRR | Complete@5 | 检索 p95 |
|---|---|---:|---:|---:|---:|---:|
| A0 | 固定长度切分 + 向量召回 | 50.00% | 75.00% | 0.6328 | 65.56% | 0.02 ms |
| A1 | 改为语义边界切分 | 51.11% | 82.22% | 0.6535 | 73.33% | 0.01 ms |
| A2 | 增加父子索引去重 | 51.11% | 89.44% | 0.6678 | 83.33% | 0.01 ms |
| A3 | 增加 OCR/文本清洗 | 52.22% | 92.78% | 0.6783 | 88.89% | 0.01 ms |
| A4 | 增加 ES/向量并行召回与 RRF | 77.78% | 100.00% | 0.8833 | 100.00% | 6.75 ms |

逐步 Recall@5 增量为：语义切分 **+7.22pp**、父子索引 **+7.22pp**、数据清洗 **+3.34pp**、ES 混合检索 **+7.22pp**。

## 实验设置

- 总数据集：180 条；检索消融只使用 48 条单跳和 42 条多跳，共 90 条。
- 向量模型：`sentence-transformers/all-MiniLM-L6-v2`。
- 关键词检索：本地真实 Elasticsearch `8.17.3`，`cjk` analyzer + BM25。
- 融合：ES 和向量检索通过两个 worker 并行执行，使用 `RRF(k=60)` 合并。
- LangSmith：追踪已开启；180 条数据集已上传为 `medigenius-rag-eval-v2-180`。
- 指标：多跳 Recall@5 按命中证据源比例计分；Complete@5 要求所有证据源均命中；MRR 取第一个相关源的倒数排名。

## 分类结果

| 版本 | 单跳 Recall@5 | 多跳 Recall@5 | 单跳 Complete@5 | 多跳 Complete@5 |
|---|---:|---:|---:|---:|
| A0 | 79.17% | 70.24% | 79.17% | 50.00% |
| A1 | 87.50% | 76.19% | 87.50% | 57.14% |
| A2 | 91.67% | 86.90% | 91.67% | 73.81% |
| A3 | 93.75% | 91.67% | 93.75% | 83.33% |
| A4 | 100.00% | 100.00% | 100.00% | 100.00% |

## 不能据此宣称的结论

当前数据不支持“数据清洗带来最大提升”：它在本次消融中的 Recall@5 增量只有 3.34pp。Codex Judge 还发现自动生成的多跳题语义质量较差，因此 A4 的 100% 更适合作为版本内回归结果，不能当作线上真实准确率。要验证“清洗贡献最大”，需要保留原始 OCR 快照、清洗后文本和人工复核 Query，重新做严格对照。

原始结果见 [`rag_ablation_result_v2.json`](../../backend/data/eval/rag_ablation_result_v2.json)，汇总表见 [`rag_ablation_results_v2.csv`](./rag_ablation_results_v2.csv)，数据质量抽审见 [`codex_judge_review_v2.md`](./codex_judge_review_v2.md)。

## 复现

```bash
cd backend
uv sync --extra dev
uv run python scripts/build_langsmith_eval_dataset.py
uv run python scripts/run_rag_ablation.py --es-url http://127.0.0.1:9200
```

运行消融前需自行启动 Elasticsearch；实验下载的临时 ES 二进制未提交仓库。
