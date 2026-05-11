# MediGenius (HardWare-Medicial)

[English](./README.md) | [简体中文](./README.zh-CN.md)

MediGenius is a production-oriented healthcare AI assistant that combines multi-agent orchestration, RAG retrieval, streaming interaction, long-term memory, and medical report delivery into one end-to-end system.

Two core pipelines:

1. **Multi-department medical Q&A** -- safety triage -> domain/department routing -> query rewriting -> hybrid retrieval (ChromaDB + keyword) -> reranking -> personalized answer generation, with optional web search
2. **ECG report generation** -- cloud fetch or synthetic-normal mode -> structured parameter analysis -> professional Chinese report with PDF output

Designed for practical scenarios: pre-consult triage, chronic-care follow-up, and wearable/monitor ECG interpretation support.

## Key Highlights

- **9-node LangGraph workflow** with safety triage bypass and single executor sink
- **Department-level RAG** across 8 medical departments with per-department query rewriting and scoped retrieval
- **Hybrid retrieval** -- parallel ChromaDB vector search + BM25 keyword search (memory or Elasticsearch backend)
- **Two-stage reranking** -- rule-based scoring + optional cross-encoder model reranker
- **Real-time SSE streaming** with token-level delta updates
- **Multi-tenant isolation** via `tenant_id + user_id + session_id` with PBKDF2 password auth and HMAC-signed tokens
- **Infrastructure** -- optional Redis with in-memory fallback, semantic cache, rate limiting, async task queue
- **LangSmith observability** -- tracing, evaluation pipeline with route/retrieval/behavior metrics
- **ECG end-to-end** -- cloud fetch -> signal parsing -> structured report -> PDF with waveform rendering

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 + Vite + Tailwind CSS 4 + daisyUI 5 |
| Backend | FastAPI + LangGraph |
| LLM | OpenAI-compatible API (configurable model) |
| Retrieval | ChromaDB (vector) + BM25 (keyword, in-memory or Elasticsearch) |
| Storage | SQLite (chat/users) + JSON (profiles) + filesystem (PDF/vectors) |
| Testing | pytest + vitest |

## Architecture

```
Frontend (React/Vite)
   ├─ POST /api/v1/chat/stream (SSE)
   ├─ POST /api/v1/auth/login
   ├─ POST /api/v1/ecg/monitor/start
   └─ GET  /api/v1/ecg/monitor/{task_id}/events

Backend (FastAPI) ─── LangGraph Workflow (9 nodes):

  memory_read
      │
      ▼
  health_concierge  ←── safety triage + domain classification
      │
      ├── EMERGENCY/CLARIFY ────► executor (safety bypass)
      │
      ├── department forced ────► query_rewriter → rag → reranker → executor
      │
      ├── medical ──► medical_router → query_rewriter → rag → reranker → executor
      │
      ├── nutrition/fitness/sleep ──► query_rewriter → rag → reranker → executor
      │
      └── general ──► judge_need_rag → (need_rag) query_rewriter → rag → reranker → executor
                                      └─ (!need_rag) executor
                                                                           │
                                                                           ▼
                                                                    memory_write_async
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
| GET | `/api/v1/sessions` | List sessions (scoped by tenant/user) |
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

## ECG Usage Flow

1. Log into the system
2. Click the ECG report button to open the guide modal
3. Fill in patient info (name, age, gender, height, weight)
4. Ensure ECG data has been uploaded to the cloud site
5. System fetches the latest record, analyzes signals, and generates a PDF report with risk stratification

## Project Structure

```
HardWare-Medicial/
├── backend/
│   ├── app/
│   │   ├── agents/          # 9 LangGraph agents
│   │   ├── api/v1/endpoints/ # REST API endpoints
│   │   ├── core/            # State, config, workflow, langsmith
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic schemas
│   │   ├── services/        # Business logic (chat, auth, ecg, cache, etc.)
│   │   └── tools/           # LLM client, vector store, search, reranker
│   ├── data/knowledge/      # Medical textbook EPUBs by department
│   ├── data/eval/           # LangSmith eval datasets and results
│   ├── scripts/             # Eval pipeline, RAG profiling
│   ├── storage/             # SQLite DB, ChromaDB, profiles, ECG PDFs
│   └── tests/               # 16 test files (pytest)
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

On top of that prototype, this repository introduces substantial re-engineering: multi-department routing with per-department RAG, 9-node agent workflow with safety triage, SSE streaming, hybrid retrieval with reranking, ECG end-to-end pipeline with PDF output, multi-tenant auth, semantic caching, and LangSmith observability.

## Creators

- **ElonGe** -- [GitHub](https://github.com/PacemakerG)
- **xhforever** -- [GitHub](https://github.com/xhforever)
- **Project** -- [HardWare-Medicial](https://github.com/PacemakerG/HardWare-Medicial)

## Disclaimer

This system is for medical assistance and research demonstration only. It does not replace licensed clinical diagnosis. If you experience acute high-risk symptoms, seek immediate in-person medical care or call emergency services.
