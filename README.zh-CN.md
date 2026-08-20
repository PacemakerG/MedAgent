# 医枢智疗 (HardWare-Medicial)

[English](./README.md) | [简体中文](./README.zh-CN.md)

医枢智疗是一个面向真实医疗场景的生产级 AI 助手系统，将多 Agent 协作、RAG 检索、流式交互、长期记忆和医疗报告交付整合为一个端到端工作台。

两条核心管线：

1. **多科室医疗问答** -- 医学意图二值识别 -> 科室路由 -> 可选查询改写 -> 混合检索（ChromaDB + 关键词）-> 重排序 -> 个性化回答生成，可选联网搜索
2. **ECG 报告生成** -- 云端抓取或合成正常模式 -> 结构化参数分析 -> 专业中文报告 + PDF 输出

适用场景：诊前分诊与症状引导、慢病随访上下文连续、可穿戴/监护仪 ECG 解读辅助。

## 核心特性

- **9 节点 LangGraph 工作流**，采用医学意图二值路由和单执行器汇聚模式
- **科室级 RAG** 覆盖 8 个临床科室，支持可选查询改写与范围检索，避免跨科室污染
- **混合检索** -- ChromaDB 向量检索 + Elasticsearch BM25 并行召回，使用 RRF 融合
- **两阶段重排序** -- 基于规则打分 + BGE 交叉编码器模型重排序
- **真 SSE 流式** -- token 级增量更新
- **用户/会话隔离** -- `user_id + session_id` 两级隔离，PBKDF2 密码认证 + HMAC 签名令牌
- **Redis 语义缓存** -- 医学实体归一化与过滤 + 512 维向量 + Redis Stack HNSW 检索
- **LangSmith 可观测性** -- 分布式追踪 + RAG、路由、Redis 三套独立评测管线
- **ECG 全流程** -- 云端抓取 -> 信号解析 -> 结构化报告 -> 含波形渲染的 PDF 输出

## 技术栈

| 层级 | 技术 |
|---|---|
| 前端 | React 19 + Vite + Tailwind CSS 4 + daisyUI 5 |
| 后端 | FastAPI + LangGraph |
| 大模型 | OpenAI 兼容 API（模型可配置） |
| 检索 | ChromaDB（向量）+ Elasticsearch（BM25）+ RRF + BGE Reranker |
| 存储 | SQLite（聊天/用户）+ Redis Stack（语义缓存）+ JSON（画像）+ 文件系统（PDF/向量库） |
| 测试 | pytest + vitest |

## 系统架构

```
前端 (React/Vite)
   ├─ POST /api/v1/chat/stream (SSE)
   ├─ POST /api/v1/auth/login
   ├─ POST /api/v1/ecg/monitor/start
   └─ GET  /api/v1/ecg/monitor/{task_id}/events

后端 (FastAPI) ─── LangGraph 工作流 (9 节点):

  semantic_cache（语义缓存）
      ├── 命中 ───────────────────────────────────────────────────► 返回响应
      └── 未命中
              │
              ▼
  memory_read（记忆读取）
      │
      ▼
  keyword_router（医学 / 非医学二值意图识别）
      │
      ├── 手动锁定科室 ────► query_rewriter → rag → reranker → executor
      │
      ├── medical ──► medical_router → query_rewriter → rag → reranker → executor
      │
      └── non-medical ──► judge_need_rag → (need_rag) query_rewriter → rag → reranker → executor
                                          └─ (!need_rag) executor
                                                                           │
                                                                           ▼
                                                                    memory_write_async（记忆写入）
                                                                           │
                                                                           ▼
                                                              semantic_cache 写入 → 返回响应
```

核心服务：ChatService, AuthService, DatabaseService, ProfileService, RedisService, RateLimitService, SemanticCacheService, TaskQueueService, ECGReportService, ECGMonitorService, ECGPdfService

## 快速开始

### 1) 环境准备

```bash
conda activate medigenius
```

### 2) 安装依赖

```bash
cd backend
pip install -r requirements.txt

cd ../frontend
npm install
```

### 3) 配置环境变量

```bash
cp backend/.env.example backend/.env
```

必配变量：
- `OPENAI_BASE_URL` -- LLM API 地址
- `OPENAI_API_KEY` -- LLM API 密钥
- `LLM_MODEL` / `LIGHT_LLM_MODEL` -- 主模型和轻量模型名称
- `OPENAI_WIRE_API` -- `chat` 或 `responses`
- `SESSION_SECRET_KEY` / `AUTH_TOKEN_SECRET` -- 会话和令牌签名的随机密钥

推荐配置：
- `RAG_ENABLED`, `EMBEDDING_MODEL_NAME` -- RAG 配置
- `QUERY_REWRITER_ENABLED`, `HYBRID_RETRIEVAL_ENABLED`, `RERANKER_MODEL_ENABLED` -- 检索质量
- `TAVILY_API_KEY` -- 联网搜索回退
- `ECG_SITE_URL` / `ECG_SITE_USER` / `ECG_SITE_PASS` -- ECG 云端抓取
- `SEMANTIC_CACHE_ENABLED`, `REDIS_ENABLED` -- 性能优化

### 4) 一键启动

```bash
python run.py
```

默认端口：后端 `8000`，前端 `5173`（端口占用自动递增）。

### 5) 准备本地医学知识库

医学知识库 PDF 体积较大且受各发布机构使用条款约束，因此只保留在本地，不提交到 GitHub。请按照[官方来源清单](backend/data/knowledge/医学知识库官方下载来源.md)下载文件，并保持清单中的目录和文件名；下载完成后重建向量库。

## API 接口

### 认证
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/auth/me` | 当前登录状态和身份信息 |
| POST | `/api/v1/auth/login` | 密码登录，返回 Bearer 令牌 |
| POST | `/api/v1/auth/logout` | 清除会话身份 |

### 聊天与会话
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 非流式聊天 |
| POST | `/api/v1/chat/stream` | SSE 流式聊天 |
| POST | `/api/v1/chat/jobs` | 排队异步聊天任务 |
| GET | `/api/v1/jobs/{job_id}` | 轮询异步任务状态 |
| GET | `/api/v1/sessions` | 列出会话（按用户范围） |
| GET | `/api/v1/session/{session_id}` | 加载会话详情 |
| DELETE | `/api/v1/session/{session_id}` | 删除会话 |
| GET | `/api/v1/history` | 当前会话聊天历史 |
| POST | `/api/v1/new-chat` | 创建新会话 |
| POST | `/api/v1/clear` | 清除对话状态 |
| POST | `/api/v1/welcome` | 生成主动问候语 |

### ECG
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/ecg/report` | 生成 ECG 报告 |
| GET | `/api/v1/ecg/report/{report_id}` | 按 ID 查询报告 |
| GET | `/api/v1/ecg/report/{report_id}/pdf` | 下载 PDF 报告 |
| POST | `/api/v1/ecg/monitor/start` | 启动 ECG 监控任务 |
| GET | `/api/v1/ecg/monitor/{task_id}` | 查询任务状态 |

### 健康检查
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/healthz` | 存活探针 |
| GET | `/api/v1/readyz` | 就绪探针 |

## 测试

```bash
# 后端
cd backend
pytest -v                           # 全部测试
pytest tests/test_agents.py -v      # 单个测试文件
pytest --cov=app --cov-report=html  # HTML 覆盖率报告

# 前端
cd frontend
npm test            # vitest
npm run build       # 生产构建
npm run lint        # ESLint
```

## 评测

项目使用三套互不混算的数据集，共 250 个样本：RAG 150 条、全链路路由 50 条、Redis 语义缓存 50 对。完整口径见[评测方案](docs/evaluation/评测方案.md)，逐项结果见[评测结果](docs/evaluation/评测结果.md)。

已完成的主要结果：

| 实验 | 结果 |
| --- | --- |
| 最终 RAG 组合 C2 | Hit@1 30.00%、Recall@5 48.67%、MRR 0.4050、平均检索耗时 1067.76 ms |
| 优化前路由诊断 | 路由准确率 76.00%、科室准确率 65.00%；修正后正式重跑待模型额度恢复 |
| Redis 语义缓存 | 命中判断准确率 100.00%（50/50），平均耗时由 10676.83 ms 降至 6.70 ms |

最终 RAG 默认采用固定分块、向量/Elasticsearch 并行召回、RRF 和 BGE Reranker。Query Rewrite 因平均增加约 6.3 秒且降低 Hit@1，默认关闭；父子索引因质量下降和复杂度增加而舍弃。部分答案忠实度及修正后路由重跑需要调用模型，当前因额度耗尽暂缓，README 不把它们记为已完成。

```bash
cd backend
uv run python scripts/evaluation/build_datasets.py
uv run python scripts/evaluation/upload_langsmith.py
uv run python scripts/evaluation/evaluate_rag.py --with-faithfulness
uv run python scripts/evaluation/evaluate_routing.py
uv run python scripts/evaluation/evaluate_redis_cache.py
```

## ECG 使用流程

1. 登录系统
2. 点击 ECG 报告按钮，打开引导弹窗
3. 填写患者信息（姓名、年龄、性别、身高、体重）
4. 确认 ECG 数据已上传至云端站点
5. 系统自动抓取最新记录，解析信号，生成含风险分层的 PDF 报告

## 项目结构

```
HardWare-Medicial/
├── backend/
│   ├── app/
│   │   ├── agents/          # 9 个 LangGraph Agent
│   │   ├── api/v1/endpoints/ # REST API 端点
│   │   ├── core/            # State、Config、Workflow、LangSmith
│   │   ├── models/          # SQLAlchemy 模型
│   │   ├── schemas/         # Pydantic 模式
│   │   ├── services/        # 业务逻辑（chat、auth、ecg、cache 等）
│   │   └── tools/           # LLM 客户端、向量库、搜索、重排序器
│   ├── data/knowledge/      # 本地医学 PDF（Git 忽略）与官方来源清单
│   ├── data/eval/           # LangSmith 评测数据集与结果
│   ├── scripts/evaluation/  # 五个统一评测脚本
│   ├── storage/             # SQLite 数据库、ChromaDB、用户画像、ECG PDF
│   └── tests/               # 20 个测试文件 (pytest)
├── frontend/
│   └── src/                 # React 19 单页应用
├── hardware/                # ECG 数据管线脚本
├── docs/
│   ├── engineering/         # 工程方案与实施报告
│   └── evaluation/          # 评测报告与方法论
└── run.py                   # 一键启动脚本
```

## 致谢

本项目的早期灵感来自 **Md. Emon Hasan** 的 MediGenius 原型：

- https://github.com/Md-Emon-Hasan/MediGenius

在此原型基础上，本项目进行了大幅二次开发与系统重构，主要改进包括：8 科室路由与范围化 RAG 检索、采用医学意图二值路由的 9 节点 Agent 工作流、SSE 真流式交互、混合检索与两阶段重排序、ECG 全流程（云端抓取 -> 信号解析 -> 报告生成 -> PDF 交付）、用户级认证体系、语义缓存、速率限制，以及 LangSmith 可观测性与评测管线。

## 创作者

- **ElonGe** -- [GitHub](https://github.com/PacemakerG)
- **xhforever** -- [GitHub](https://github.com/xhforever)
- **项目地址** -- [HardWare-Medicial](https://github.com/PacemakerG/HardWare-Medicial)

## 免责声明

本系统用于医疗辅助与科研演示，不替代执业医师诊断。若出现急性高风险症状（胸痛、呼吸困难、意识改变等），请立即线下就医或呼叫急救。
