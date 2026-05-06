# LangSmith 全流程评测报告

- 数据集：`backend/data/eval/langsmith_eval_dataset_v1.jsonl`
- 结果文件：`backend/data/eval/langsmith_eval_result_v1.json`
- 样本数：`50`
- LangSmith tracing：`False`
- 路由匹配率：`0.84`
- 行为通过率：`0.4`
- LLM 成功率：`0`
- RAG 成功率：`0.64`
- runner 错误数：`0`
- RAG Top1：`0.1`
- RAG Recall@5：`0.1667`
- RAG MRR：`0.1333`

## 结果解读

- 本轮 LLM 成功率为 0，说明生成模型调用未成功；RAG 类样本的行为通过率会被显著拉低，应优先检查模型供应商拦截、模型名、API key 或 base URL。

## 分类指标

| 分类 | 样本数 | 路由匹配 | 行为通过 | LLM成功 | RAG成功 | Top1 | Recall | MRR | 平均耗时 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_hop | 14 | 0.9286 | 0 | 0 | 0.9286 | 0.0714 | 0.1429 | 0.1071 | 646.4264 |
| negative | 10 | 0.7 | 1 | 0 | 0.3 |  |  |  | 338.191 |
| open_domain | 10 | 0.6 | 1 | 0 | 0 |  |  |  | 276.177 |
| single_hop | 16 | 1 | 0 | 0 | 1 | 0.125 | 0.1875 | 0.1562 | 2545.7288 |

## 需要关注的样本

- `ls_single_hop_001` `single_hop` route=True behavior=False recall=0 error=`` question=淋菌性咽炎常用哪些治疗方案？
- `ls_single_hop_002` `single_hop` route=True behavior=False recall=0 error=`` question=二期梅毒一般在什么时候发生？有哪些典型皮肤黏膜表现？
- `ls_single_hop_003` `single_hop` route=True behavior=False recall=0 error=`` question=二期梅毒常见的斑疹性梅毒疹和黏膜损害有哪些特点？
- `ls_single_hop_004` `single_hop` route=True behavior=False recall=0 error=`` question=慢性外耳道炎常用哪些滴耳药物？怎么使用？
- `ls_single_hop_005` `single_hop` route=True behavior=False recall=0 error=`` question=变应性鼻炎应用莫米松鼻喷雾剂有什么作用和用法？
- `ls_single_hop_006` `single_hop` route=True behavior=False recall=0 error=`` question=甲型流感为什么容易引起大流行？它有哪些流行特点？
- `ls_single_hop_007` `single_hop` route=True behavior=False recall=0 error=`` question=某些传染病进入多尿期后可能出现哪些并发症？需要注意哪些电解质问题？
- `ls_single_hop_008` `single_hop` route=True behavior=False recall=1 error=`` question=TIA患者在什么情况下建议住院治疗？ABCD2评分怎么用？
- `ls_single_hop_009` `single_hop` route=True behavior=False recall=0 error=`` question=额叶癫痫发作持续时间和发作形式有哪些典型特点？
- `ls_single_hop_010` `single_hop` route=True behavior=False recall=1 error=`` question=ETDRS视力检查法有什么特点，临床上怎么使用？
- `ls_single_hop_011` `single_hop` route=True behavior=False recall=0 error=`` question=角膜炎的病理变化通常分哪几个阶段？浸润期有什么表现？
- `ls_single_hop_012` `single_hop` route=True behavior=False recall=0 error=`` question=色觉检查常用哪些方法？假同色图检查有什么优缺点？
