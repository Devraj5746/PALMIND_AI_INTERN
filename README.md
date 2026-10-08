# RAG Backend

Document ingestion and conversational RAG backend built with FastAPI, Qdrant, Redis, and SQLAlchemy. Supports LLM-driven interview booking through natural conversation.

## Features

- **Document Ingestion** — Upload `.pdf` / `.txt`, choose chunking strategy (fixed-size or recursive), generate embeddings, store vectors in Qdrant, persist metadata in SQL
- **Conversational RAG** — Custom retrieval pipeline (no RetrievalQAChain), Redis-backed chat memory, multi-turn context
- **Interview Booking** — LLM extracts name, email, date, time from conversation; multi-turn slot-filling; bookings stored in SQL
- **Fully Offline Test Suite** — 314 tests run without API keys using mock LLM + hash embeddings + in-memory Qdrant + fakeredis + SQLite

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI + Uvicorn |
| Vector store | Qdrant |
| Chat memory | Redis |
| Metadata DB | PostgreSQL (SQLAlchemy async) / SQLite for dev |
| LLM | OpenAI (configurable) |
| Embeddings | OpenAI / Sentence Transformers / Hash (offline) |
| Document parsing | pypdf, plain-text with encoding detection |
| Chunking | Fixed-size, Recursive (via LangChain text splitters) |

## Project Structure

```
app/
├── api/v1/           # Route handlers (ingestion, chat, booking, system)
├── core/             # Config, errors, middleware, logging
├── db/               # SQLAlchemy models + repositories
├── schemas/          # Pydantic request/response models
└── services/
    ├── chunking/     # Fixed + recursive chunking strategies
    ├── embeddings/   # OpenAI, sentence-transformers, hash (offline)
    ├── extraction/   # PDF + TXT text extraction
    ├── llm/          # LLM factory, mock model, booking rules
    ├── memory/       # Redis chat memory + booking draft store
    ├── rag/          # Custom RAG engine, retriever, prompts
    └── vector_store/ # Qdrant adapter
tests/
├── integration/      # Full API endpoint tests
└── unit/             # Per-module unit tests
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Install

```bash
make install
cp .env.example .env
# Edit .env — set OPENAI_API_KEY or switch to offline mode
```

### Offline Mode (no API keys)

Set these in `.env`:

```
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hash
DATABASE_URL=sqlite+aiosqlite:///./rag.db
QDRANT_URL=:memory:
REDIS_URL=redis://localhost:6379/0
```

### Run

```bash
make run              # uvicorn on port 8000, auto-reload
```

### Docker

```bash
docker compose up -d  # starts API + Qdrant + Redis + PostgreSQL
curl http://localhost:8000/api/v1/health
```

### Tests

```bash
make test             # 314 tests, no network/API keys needed
make test-cov         # with coverage report
make check            # lint + typecheck + tests
```

## API Endpoints

### Document Ingestion (`/api/v1/documents`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents` | Upload PDF/TXT with chunking strategy selection |
| `GET` | `/documents` | List documents (paginated, filterable by status) |
| `GET` | `/documents/{id}` | Get document with chunks |
| `DELETE` | `/documents/{id}` | Delete document + vectors |
| `POST` | `/documents/search` | Semantic search over indexed chunks |

**Upload example:**

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@report.pdf" \
  -F "chunking_strategy=recursive" \
  -F "chunk_size=1000" \
  -F "chunk_overlap=150"
```

### Conversational RAG (`/api/v1/chat`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send message (RAG or booking, auto-routed) |
| `GET` | `/chat/{session_id}/history` | Get conversation history |
| `DELETE` | `/chat/{session_id}` | Clear session |

**Chat example:**

```bash
# RAG query
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the document say about pricing?", "session_id": "user-123"}'

# Booking via conversation
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to book an interview for tomorrow at 2pm", "session_id": "user-123"}'
```

### Bookings (`/api/v1/bookings`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/bookings` | Create booking directly (REST) |
| `GET` | `/bookings` | List bookings (filter by email/session) |
| `GET` | `/bookings/{id}` | Get booking |
| `DELETE` | `/bookings/{id}` | Delete booking |

### System (`/api/v1`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (DB, Qdrant, Redis, embeddings) |
| `GET` | `/stats` | Corpus statistics |

Interactive docs at `http://localhost:8000/docs`.

## Chunking Strategies

- **`recursive`** (default) — Splits on paragraph, sentence, then word boundaries. Preserves natural text structure. Best for most documents.
- **`fixed`** — Uniform character windows with overlap. Predictable chunk sizes. Good for structured/tabular content.

Both are selectable per upload via the `chunking_strategy` form field.

## Configuration

All settings via environment variables. See [`.env.example`](.env.example) for the full list with descriptions.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `mock` |
| `EMBEDDING_PROVIDER` | `openai` | `openai`, `hash`, or `sentence_transformers` |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant connection (`:memory:` for ephemeral) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `DATABASE_URL` | PostgreSQL | Any SQLAlchemy async URL |
| `DEFAULT_CHUNKING_STRATEGY` | `recursive` | `recursive` or `fixed` |

## Design Decisions

- **Custom RAG pipeline** — No `RetrievalQAChain`. Four explicit steps: condense follow-ups, retrieve chunks, assemble context, generate answer. Full control over prompts and context window.
- **Qdrant only** — No FAISS/Chroma as per requirements.
- **Redis for chat memory** — Sliding-window transcript per session with TTL. Also stores in-progress booking drafts.
- **Container pattern** — Single `Container` dataclass wires all dependencies at startup. Tests inject offline providers without touching application code.
- **Mock LLM** — Deterministic rule-based model enables full offline testing of the booking flow, including multi-turn slot-filling.
