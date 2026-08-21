# MedAgent

[English](./README.md) | [简体中文](./README.zh-CN.md)

MedAgent is a production-oriented healthcare AI assistant that combines multi-agent orchestration, RAG retrieval, streaming interaction, long-term memory, and medical report delivery into one end-to-end system.

Two core pipelines:

1. **Multi-department medical Q&A** -- medical-intent classification -> department routing -> optional query rewriting -> hybrid retrieval (ChromaDB + keyword) -> reranking -> personalized answer generation, with optional web search
2. **ECG report generation** -- cloud fetch or synthetic-normal mode -> structured parameter analysis -> professional Chinese report with PDF output

Designed for practical scenarios: pre-consult triage, chronic-care follow-up, and wearable/monitor ECG interpretation support.

## Key Highlights

- **9-node LangGraph workflow** with binary medical-intent routing and a single executor sink
- **Department-level RAG** across 8 medical departments with optional query rewriting and scoped retrieval
- **Hybrid retrieval** -- parallel ChromaDB vector search + Elasticsearch BM25 with RRF fusion
- **Two-stage reranking** -- rule-based scoring + BGE cross-encoder reranking
- **Real-time SSE streaming** with token-level delta updates
- **User/session isolation** via `user_id + session_id` with PBKDF2 password auth and HMAC-signed tokens
- **Redis semantic cache** -- normalized medical entities and filtering + 512-d vectors + Redis Stack HNSW search
- **LangSmith observability** -- tracing plus separate RAG, routing, and Redis evaluation pipelines
- **ECG end-to-end** -- cloud fetch -> signal parsing -> structured report -> PDF with waveform rendering

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + Vite + Tailwind CSS 4 + daisyUI 5 |
| Backend | FastAPI + LangGraph |
| LLM | OpenAI-compatible API (configurable model) |
| Retrieval | ChromaDB (vector) + Elasticsearch (BM25) + RRF + BGE Reranker |
| Storage | SQLite (chat/users) + Redis Stack (semantic cache) + JSON (profiles) + filesystem (PDF/vectors) |
| Testing | pytest + vitest |

## Architecture

```
Frontend (React/Vite)
   ├─ POST /api/v1/chat/stream (SSE)
   ├─ POST /api/v1/auth/login
   ├─ POST /api/v1/ecg/monitor/start
   └─ GET  /api/v1/ecg/monitor/{task_id}/events

Backend (FastAPI) ─── LangGraph Workflow (9 nodes):

  semantic_cache
      ├── cache hit ───────────────────────────────────────────────► response
      └── cache miss
              │
              ▼
  memory_read
      │
      ▼
  keyword_router  ←── binary medical-intent classification
      │
      ├── department forced ────► query_rewriter → rag → reranker → executor
      │
      ├── medical ──► medical_router → query_rewriter → rag → reranker → executor
      │
      └── non-medical ──► judge_need_rag → (need_rag) query_rewriter → rag → reranker → executor
                                          └─ (!need_rag) executor
                                                                           │
                                                                           ▼
                                                                    memory_write_async
                                                                           │
                                                                           ▼
                                                              semantic_cache store → response
```

Key services: ChatService, AuthService, DatabaseService, ProfileService, RedisService, RateLimitService, SemanticCacheService, TaskQueueService, ECGReportService, ECGMonitorService, ECGPdfService

## Quick Start

### 1) Environment

```bash
conda activate medigenius
```

### 2) Install Dependencies

```bash
cd backend
pip install -r requirements.txt

cd ../frontend
npm install
```

### 3) Configure Environment

```bash
cp backend/.env.example backend/.env
```

Required variables:
- `OPENAI_BASE_URL` -- LLM API base URL
- `OPENAI_API_KEY` -- LLM API key
- `LLM_MODEL` / `LIGHT_LLM_MODEL` -- main and lightweight model names
- `OPENAI_WIRE_API` -- `chat` or `responses`
- `SESSION_SECRET_KEY` / `AUTH_TOKEN_SECRET` -- random secrets for session and token signing

Optional but recommended:
- `RAG_ENABLED`, `EMBEDDING_MODEL_NAME` -- RAG configuration
- `QUERY_REWRITER_ENABLED`, `HYBRID_RETRIEVAL_ENABLED`, `RERANKER_MODEL_ENABLED` -- retrieval quality
- `TAVILY_API_KEY` -- web search fallback
- `ECG_SITE_URL` / `ECG_SITE_USER` / `ECG_SITE_PASS` -- ECG cloud fetch
- `SEMANTIC_CACHE_ENABLED`, `REDIS_ENABLED` -- performance

### 4) Run

```bash
python run.py
```

Default ports: backend `8000`, frontend `5173` (auto-increments if occupied).

### 5) Prepare the Local Medical Knowledge Base

The medical PDFs are large and remain subject to their publishers' terms, so they are kept locally and are not committed to GitHub. Download them from the [official-source manifest](backend/data/knowledge/医学知识库官方下载来源.md), preserve the listed paths and filenames, and then rebuild the vector store.

## API Endpoints

### Auth
| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/auth/me` | Current login status and identity |
| POST | `/api/v1/auth/login` | Password login, returns Bearer token |
| POST | `/api/v1/auth/logout` | Clear session identity |

### Chat & Sessions
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/chat` | Non-streaming chat |
| POST | `/api/v1/chat/stream` | SSE streaming chat |
| POST | `/api/v1/chat/jobs` | Queue async chat task |
| GET | `/api/v1/jobs/{job_id}` | Poll async job status |
| GET | `/api/v1/sessions` | List sessions (scoped by user) |
| GET | `/api/v1/session/{session_id}` | Load session details |
| DELETE | `/api/v1/session/{session_id}` | Delete session |
| GET | `/api/v1/history` | Current session chat history |
| POST | `/api/v1/new-chat` | Create new session |
| POST | `/api/v1/clear` | Clear conversation state |
| POST | `/api/v1/welcome` | Generate proactive greeting |

### ECG
| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/ecg/report` | Generate ECG report |
| GET | `/api/v1/ecg/report/{report_id}` | Query report by ID |
| GET | `/api/v1/ecg/report/{report_id}/pdf` | Download PDF report |
| POST | `/api/v1/ecg/monitor/start` | Start ECG monitoring task |
| GET | `/api/v1/ecg/monitor/{task_id}` | Query task status |

### Health
| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/healthz` | Liveness probe |
| GET | `/api/v1/readyz` | Readiness probe |

## Testing

```bash
# Backend
cd backend
pytest -v                           # all tests
pytest tests/test_agents.py -v      # single file
pytest --cov=app --cov-report=html  # with HTML coverage

# Frontend
cd frontend
npm test            # vitest
npm run build       # production build
npm run lint        # ESLint
```

## Evaluation

The project uses three independent datasets with 250 samples in total: 150 for RAG, 50 for end-to-end routing, and 50 pairs for Redis semantic caching. See the [evaluation plan](docs/evaluation/评测方案.md) for definitions and the [evaluation report](docs/evaluation/评测结果.md) for itemized results.

Key completed results:

| Experiment | Result |
| --- | --- |
| Final RAG combination C2 | Hit@1 30.00%, Recall@5 48.67%, MRR 0.4050, mean retrieval latency 1067.76 ms |
| Pre-fix routing diagnostic | Route accuracy 76.00%, department accuracy 65.00%; the post-fix formal rerun awaits model quota |
| Redis semantic cache | 100.00% hit-decision accuracy (50/50), with mean latency reduced from 10676.83 ms to 6.70 ms |

The default RAG path uses fixed chunks, parallel vector/Elasticsearch retrieval, RRF, and the BGE Reranker. Query Rewrite is disabled by default because it added about 6.3 seconds on average while reducing Hit@1; parent-child indexing was dropped because it reduced quality and increased complexity. Some answer-faithfulness scoring and the post-fix routing rerun require model calls and remain pending because the model quota is exhausted; they are not presented as completed results.

```bash
cd backend
uv run python scripts/evaluation/build_datasets.py
uv run python scripts/evaluation/upload_langsmith.py
uv run python scripts/evaluation/evaluate_rag.py --with-faithfulness
uv run python scripts/evaluation/evaluate_routing.py
uv run python scripts/evaluation/evaluate_redis_cache.py
```

## ECG Usage Flow

1. Log into the system
2. Click the ECG report button to open the guide modal
3. Fill in patient info (name, age, gender, height, weight)
4. Ensure ECG data has been uploaded to the cloud site
5. System fetches the latest record, analyzes signals, and generates a PDF report with risk stratification

## Project Structure

```
MedAgent/
├── backend/
│   ├── app/
│   │   ├── agents/          # 9 LangGraph agents
│   │   ├── api/v1/endpoints/ # REST API endpoints
│   │   ├── core/            # State, config, workflow, langsmith
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic (chat, auth, ecg, cache, etc.)
│   │   └── tools/           # LLM client, vector store, search, reranker
│   ├── data/knowledge/      # Local medical PDFs (Git-ignored) and source manifest
│   ├── data/eval/           # LangSmith eval datasets and results
│   ├── scripts/evaluation/  # Five consolidated evaluation scripts
│   ├── storage/             # SQLite DB, ChromaDB, profiles, ECG PDFs
│   └── tests/               # 20 test files (pytest)
├── frontend/
│   └── src/                 # React 19 single-page app
├── hardware/                # ECG data pipeline scripts
├── docs/
│   ├── engineering/         # Engineering plans and reports
│   └── evaluation/          # Eval reports and methodology
└── run.py                   # One-click launcher
```

## Acknowledgement

This project was inspired by the original MediGenius prototype by **Md. Emon Hasan**:

- https://github.com/Md-Emon-Hasan/MediGenius

On top of that prototype, this repository introduces substantial re-engineering: 8-department routing with scoped RAG, a 9-node agent workflow with binary medical-intent routing, SSE streaming, hybrid retrieval with reranking, ECG end-to-end pipeline with PDF output, user-scoped authentication, semantic caching, and LangSmith observability.

## Creators

- **ElonGe** -- [GitHub](https://github.com/PacemakerG)
- **xhforever** -- [GitHub](https://github.com/xhforever)
- **Project** -- [MedAgent](https://github.com/PacemakerG/MedAgent)

## Disclaimer

This system is for medical assistance and research demonstration only. It does not replace licensed clinical diagnosis. If you experience acute high-risk symptoms, seek immediate in-person medical care or call emergency services.
