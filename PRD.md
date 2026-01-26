# Product Requirements Document (PRD): Docling RAG (GPU-Accelerated, Multi‑Agent)

## 1. Executive Summary

Docling RAG is a full‑stack application for turning PDF documents (including complex layouts, tables, images, and handwritten notes) into a searchable knowledge base and a chat experience backed by retrieval‑augmented generation (RAG). Documents are parsed and chunked with Docling, indexed into Azure AI Search for hybrid + vector retrieval, and answered using Azure OpenAI models.

The product’s differentiators are (1) robust document understanding via OCR + table extraction (optionally GPU‑accelerated), (2) a multi‑agent LangGraph pipeline that evaluates retrieved chunks in parallel and synthesizes an answer with explicit source attribution, and (3) a real‑time streaming UI that visualizes the pipeline (Retrieve → Distribute → Analyze → Synthesize).

**MVP goal:** Enable a user to upload PDFs, see ingestion status, query their document set via chat, and receive answers grounded in retrieved chunks with clear citations (document + page/section metadata when available).

## 2. Mission

**Mission statement:** Make complex PDFs instantly searchable and answerable with trustworthy, citation‑backed responses—fast enough for iterative analysis, and transparent enough to verify.

**Core principles**
- Grounded answers first: prioritize source attribution and “I don’t know” over hallucination.
- Fast ingestion & iteration: optimize for short upload→index→query loops; GPU acceleration when available.
- Transparent pipeline: expose progress, retrieval counts, relevance decisions, and used sources.
- Safe by default: keep secrets out of the client; minimize data exposure; least‑privilege where feasible.
- Operable in real environments: support corporate TLS/cert bundles, constrained networks, and container deployment.

## 3. Target Users

**Primary personas**
- **Security/Compliance Analyst:** uploads advisories, audit reports, policies; asks for summaries and evidence with citations.
- **Engineer / SRE:** uploads runbooks and postmortems; asks procedural “how-to” questions and comparisons.
- **Knowledge Worker / PM:** uploads specs and meeting PDFs; asks “what changed” and “where is X defined”.

**Technical comfort**
- Moderate: comfortable with a web UI; may run Docker locally; not expected to edit code.
- Advanced (secondary): developers/operators deploying to internal environments.

**Key needs & pain points**
- PDFs are hard to search; tables/handwriting break standard extractors.
- Users need confidence: “Where did this come from?” with page/section context.
- Slow ingestion and unclear status cause friction.
- Multi‑document questions require relevant retrieval and synthesis.

## 4. MVP Scope

### Core Functionality
- ✅ Upload PDFs via UI and API; start background processing job
- ✅ Parse PDFs with Docling (OCR + table extraction) and hierarchical chunking
- ✅ Generate embeddings with Azure OpenAI; index chunks into Azure AI Search
- ✅ List indexed documents and delete a document (and its chunks)
- ✅ Ask questions via chat UI; retrieve and answer using RAG
- ✅ Multi‑agent RAG query endpoint (LangGraph orchestration) returning synthesized response + sources used
- ✅ Streaming RAG endpoint (SSE) emitting pipeline events for real‑time UI visualization

### Technical
- ✅ GPU acceleration when available (NVIDIA runtime; torch CUDA detection; EasyOCR GPU toggle)
- ✅ Automatic OCR quality assessment and optional VLM fallback for PDFs (Azure OpenAI vision-capable model)
- ✅ Chunk metadata extraction (page numbers, section/hierarchy path, chunk type, timestamps)
- ✅ Basic operational endpoints (`/health`, streaming health)

### Integration
- ✅ Azure AI Search index creation (vector + metadata fields) and hybrid search
- ✅ Azure OpenAI embeddings and chat completions
- ❌ Non‑Azure providers (OpenSearch, Pinecone, OpenAI non‑Azure)

### Deployment
- ✅ Docker Compose for frontend + backend + LangGraph state stores (Redis/Postgres)
- ✅ Environment‑based configuration via `.env`
- ❌ Multi-tenant hosting, SSO, or enterprise IAM integration

## 5. User Stories

1) As a **user**, I want to **upload a PDF and see processing progress**, so that **I know when it’s ready to query**.  
Example: Upload a scanned report; see “processing” then “completed: 143 chunks indexed”.

2) As a **user**, I want to **ask a question and get an answer grounded in my documents**, so that **I can act on it confidently**.  
Example: “What are the required controls for vendor access?” returns a summary and indicates which documents were used.

3) As a **user**, I want to **see which sources were used (document + page/section when available)**, so that **I can verify the claims quickly**.  
Example: Answer includes references to “Policy.pdf p. 12 § Access Control”.

4) As a **user**, I want to **delete a document and all its indexed chunks**, so that **outdated or sensitive content is removed**.  
Example: Delete “Draft-Contract.pdf” and confirm it no longer appears in the document list.

5) As a **user**, I want to **toggle streaming mode and watch pipeline progress**, so that **long queries feel responsive and debuggable**.  
Example: See “Retrieve (20 chunks) → Distribute (20 analysts) → Analyze → Synthesize”.

6) As an **operator**, I want to **configure Azure endpoints and model deployments via environment variables**, so that **deployments are reproducible across environments**.  
Example: Set `AZURE_OPENAI_ENDPOINT` and deployment names in `.env` for staging vs prod.

7) As an **operator**, I want **GPU acceleration to be optional**, so that **the app runs on CPU-only hosts without changes**.  
Example: Run without NVIDIA runtime; service logs “CUDA not available, falling back to CPU”.

8) As a **developer**, I want **structured analysis outputs with metadata mapping**, so that **citation formatting remains consistent even when upstream fields vary**.  
Example: Map `created_date`/`indexed_at` from index to `SourceMetadata.date_created/date_indexed`.

## 6. Core Architecture & Patterns

**High-level architecture**

```
Frontend (React/Vite/Nginx)
  └─ talks to ─► Backend (FastAPI)
                   ├─ Ingest: Docling → chunk+metadata → embeddings (Azure OpenAI) → index (Azure AI Search)
                   ├─ Query (simple): retrieve (AI Search) → generate (Azure OpenAI)
                   └─ Query (multi-agent): LangGraph retrieve → parallel analyze → synthesize (+ citations)
```

**Directory structure (current)**
- `backend/app/routers/*`: HTTP API surface (documents, search, chat, multi-agent rag, SSE streaming)
- `backend/app/services/*`: external integrations and processing (Docling parsing, Azure Search, Azure OpenAI)
- `backend/app/rag_agent/*`: LangGraph orchestration (state reducers, nodes, prompts, schemas, tools)
- `frontend/src/*`: React UI, API client, SSE streaming client, agent pipeline visualization components

**Key patterns**
- Router/service separation in FastAPI for clearer ownership and testability.
- Typed state with reducer functions for parallel fan‑out/fan‑in (LangGraph `Send()` + `Annotated[..., add]`).
- Metadata-first indexing to support citations and future filters.
- SSE event stream with a client-side reducer to maintain a single pipeline state model.

## 7. Tools/Features

### Document Ingestion
- Inputs: PDF upload (multipart form upload)
- Processing:
  - Standard pipeline: Docling PDF pipeline with OCR + table structure extraction + hierarchical chunker
  - GPU usage: auto-detect CUDA; enable EasyOCR GPU and Docling accelerator device when available
  - Quality assessment: confidence, density, garbage ratio, chunks/page; optional automatic fallback to VLM pipeline
  - VLM pipeline: convert PDF pages as images via Azure OpenAI vision model to markdown-like text
- Outputs:
  - Chunk text + metadata: page numbers, section title, hierarchy path, chunk type, timestamps
  - Embeddings and Azure AI Search documents

### Retrieval & Answering
- **Simple chat**: embed query → search → generate answer from retrieved chunk text.
- **Multi-agent RAG**:
  - Retrieve N chunks from Azure AI Search (with metadata filters)
  - Distribute to N analyst agents (parallel) to judge relevance and extract key points
  - Synthesize final response with citations and a shared “collaborative document”

### Streaming & Observability (User-Facing)
- Server-Sent Events endpoint for pipeline events:
  - `pipeline_started`, `step_started`, `step_completed`, `analyst_spawned`, `analyst_completed`, `final_response`, `error`
- UI visualization:
  - Step progress bar (4 steps)
  - Analyst grid showing per-chunk outcomes (relevance score, summary, confidence)
  - Metrics panel (retrieved count, relevant count, timings)
  - Shared document viewer (analyst contributions)

## 8. Technology Stack

### Backend
- Python `>=3.11,<3.14`
- FastAPI `0.115.0`, Uvicorn `0.30.6`
- Docling `~2.12` (PDF processing + chunking)
- EasyOCR `>=1.7.0`
- PyTorch + TorchVision (CUDA-enabled wheel index for `cu121`)
- LangGraph `>=0.6.0` + LangChain `>=0.3.0` family
- Azure SDKs: `azure-search-documents==11.5.2`, `azure-identity==1.17.1`
- Postgres (LangGraph checkpointing) + Redis (LangGraph queue/pubsub) via Docker Compose

### Frontend
- React `18.3`, Vite `5.4`, TypeScript `5.6`
- Tailwind CSS `3.4`
- SSE client logic + state reducer (`frontend/src/api/ragStream.ts`)
- Markdown rendering (`react-markdown`, `remark-gfm`)

### Optional / Deployment
- NVIDIA Container Toolkit (for GPU access)
- Nginx (static frontend hosting and reverse proxy behavior for SSE)

### Third-party integrations
- Azure AI Search (indexing + hybrid/vector retrieval)
- Azure OpenAI (embeddings, chat, optional vision model for VLM pipeline)

## 9. Security & Configuration

**Authentication/Authorization**
- MVP: ❌ No end-user authentication; endpoints are public (intended for trusted environments).
- Post‑MVP (recommended): ✅ Add auth (e.g., Azure AD / OAuth2) and authorization scopes for document access.

**Configuration management**
- Environment variables loaded via `.env` for backend; frontend uses `VITE_API_BASE_URL` (default `/api`).
- Secrets (Azure keys) must remain server-side; do not expose in the browser bundle.

**Security scope**
- In scope:
  - Secret handling best practices (no logging of keys; configuration via env vars)
  - CORS restrictions for local frontend origins
  - Deleting indexed data by `doc_id`
- Out of scope (MVP):
  - Tenant isolation / per-user document permissions
  - At-rest encryption strategy beyond Azure-managed defaults
  - DLP/PII redaction and advanced compliance controls

**Deployment considerations**
- Corporate TLS/cert bundles: support custom CA bundle inside containers.
- Long-running ingestion: disable keep-alive timeout for document processing.
- SSE: disable proxy buffering (e.g., `X-Accel-Buffering: no`).

## 10. API Specification

### Documents
- `POST /api/documents/upload` (multipart)  
  Query: `use_vlm` (optional boolean)  
  Response: `{ "job_id": "...", "filename": "...", "status": "pending" }`

- `GET /api/documents/status/{job_id}`  
  Response: `{ "job_id": "...", "status": "processing|completed|failed", "chunks_count": 123, "error": null }`

- `GET /api/documents`  
  Response: `[ { "doc_id": "...", "filename": "...", "chunks_count": 123 } ]`

- `DELETE /api/documents/{doc_id}`  
  Response: `{ "deleted_chunks": 123 }`

### Search & Chat
- `POST /api/search`  
  Body: `{ "query": "…", "top_k": 5 }`  
  Response: `{ "results": [ { "chunk_id": "…", "content": "…", "score": 1.23, "filename": "…", "page_numbers": [1] } ] }`

- `POST /api/chat`  
  Body: `{ "message": "…", "top_k": 5 }`  
  Response: `{ "answer": "…", "sources": [ ...SearchResult ] }`

### Multi-agent RAG (non-streaming)
- `POST /rag/query`  
  Body: `{ "query": "…", "filters": { "document_type": "report" } }`  
  Response: `{ "query": "…", "final_response": "…", "retrieval_count": 20, "relevant_count": 7, "sources_used": ["a.pdf"], "shared_document": "…" }`

### Multi-agent RAG (streaming)
- `POST /api/rag/stream` (SSE)  
  Body: `{ "query": "…", "filters": { ... } }`  
  Response: `text/event-stream` with `data:` payloads shaped like:

```json
{
  "type": "analyst_completed",
  "timestamp": "2026-01-01T00:00:00Z",
  "data": {
    "chunk_id": "chunk_3",
    "relevance_score": 8,
    "is_relevant": true,
    "summary": "…",
    "confidence": "medium",
    "completed_analysts": 4
  }
}
```

## 11. Success Criteria

**MVP success definition**
- Users can upload PDFs, query them, and receive grounded answers with usable source attribution, without needing to inspect logs or restart services.

**Functional requirements**
- ✅ Uploading a PDF creates a job and completes indexing (or fails with a surfaced error)
- ✅ Deleting a document removes its chunks from the index
- ✅ Search returns top‑K chunks with scores and metadata fields (when present)
- ✅ Chat answers are generated from retrieved context; when context is insufficient, the system says so
- ✅ Multi-agent RAG returns a final response and a list of sources used
- ✅ Streaming endpoint drives the UI with step-by-step progress and final response

**Quality indicators (measurable)**
- Ingestion: 95% of typical PDFs (internal corpus) complete without manual intervention.
- Citation utility: ≥80% of answers include at least one source; when metadata exists, include page numbers in citations.
- Latency targets (initial):
  - Search: p95 < 2s (excluding Azure outages)
  - Chat (non-streaming): p95 < 10s for typical queries
  - Streaming: first event emitted < 1s after request accepted

**User experience goals**
- Clear progress feedback for uploads and long-running queries.
- Errors are actionable (e.g., missing Azure config, index schema mismatch).

## 12. Implementation Phases

### Phase 1 (1–2 weeks): MVP ingestion + basic RAG
- Goal: reliable upload→index→search/chat loop.
- Deliverables
  - ✅ Document upload + background processing + status polling
  - ✅ Azure AI Search index creation + chunk metadata storage
  - ✅ Basic `/api/search` + `/api/chat` wired to Azure OpenAI
  - ✅ Frontend: upload + document list + chat
- Validation
  - Upload a set of PDFs and confirm searchable content; delete and confirm removal.

### Phase 2 (1–2 weeks): Multi-agent RAG (non-streaming)
- Goal: higher-quality answers via parallel chunk analysis and synthesis with citations.
- Deliverables
  - ✅ LangGraph multi-agent pipeline (retrieve → distribute → analyze → synthesize)
  - ✅ Structured analyst outputs with source metadata mapping
  - ✅ `/rag/query` endpoint for orchestrated responses
- Validation
  - Run representative queries; confirm sources_used and shared_document are populated.

### Phase 3 (1–2 weeks): Streaming UX + pipeline transparency
- Goal: real-time pipeline visualization and improved user trust.
- Deliverables
  - ✅ `/api/rag/stream` SSE endpoint with robust event schema
  - ✅ Frontend state reducer + AgentStateViewer UI
  - ✅ Cancellation handling and disconnect safety
- Validation
  - Verify event ordering and UI consistency; confirm final response matches non-streaming output.

### Phase 4 (1–2 weeks): Hardening & deployment readiness
- Goal: reduce operational risk and improve security posture.
- Deliverables
  - ✅ Config validation + health diagnostics for Azure dependencies
  - ✅ Persistence strategy for job tracking (replace in-memory job map)
  - ✅ Optional auth (feature-flagged) and rate limiting
- Validation
  - Restart backend mid-processing and verify recovery behavior; run basic threat review.

## 13. Future Considerations

- Authentication/SSO (Azure AD) + per-user document access controls.
- More metadata extraction: author, modified date, tags, document type classification.
- Query understanding / rewriting node to improve retrieval for ambiguous queries.
- Batch ingestion and folder watchers; ingestion of DOCX/HTML alongside PDFs.
- Improved citation formatting in UI (click-to-open page preview, snippet highlighting).
- Evaluation harness: golden Q&A sets, retrieval metrics, hallucination checks.
- Cost controls: caching embeddings, adaptive top‑K, model selection (mini vs full) per step.

## 14. Risks & Mitigations

1) **OCR and layout extraction quality varies by PDF type**  
Mitigation: quality heuristics + VLM fallback; expose per-document parse diagnostics.

2) **Azure cost and rate limits (embeddings + multi-agent calls)**  
Mitigation: cap retrieval/analyst counts; use smaller models for analysts; add retries/backoff and observability.

3) **No authentication in MVP exposes data if deployed broadly**  
Mitigation: restrict deployment to trusted networks; add auth as an early post‑MVP deliverable.

4) **GPU dependency complexity in deployment**  
Mitigation: make GPU optional; detect CUDA at runtime; document NVIDIA runtime requirements; keep CPU path functional.

5) **Index schema drift breaks metadata/citations**  
Mitigation: enforce `ensure_index_exists` on startup; version schema; provide a safe “recreate index” admin workflow in non-prod.

## 15. Appendix

**Repository structure**
- `README.md`: quick start, endpoint overview, and deployment notes
- `docker-compose.yml`: services for frontend/backend + Redis/Postgres; optional GPU access
- `backend/pyproject.toml`: backend dependencies (Docling, LangGraph, Azure SDKs, torch CUDA index)
- `frontend/package.json`: frontend stack (React/Vite/Tailwind)

**Key configuration (backend)**
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KEY`, `AZURE_SEARCH_INDEX_NAME`
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_MODEL`, `AZURE_OPENAI_EMBEDDINGS`
- `USE_VLM_PIPELINE`, `AZURE_OPENAI_VLM_MODEL`, `AZURE_OPENAI_VLM_MAX_TOKENS`
