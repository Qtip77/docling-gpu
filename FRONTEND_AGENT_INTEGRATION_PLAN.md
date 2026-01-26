# Frontend Agent State Integration Plan

## Overview

Integrate the frontend with the multi-agent RAG pipeline to display real-time agent state progression through each step: **Retrieve → Distribute → Analyze (parallel) → Synthesize**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (React + Vite)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  AgentStateViewer Component                                         │    │
│  │  ├── Pipeline Progress Bar                                          │    │
│  │  ├── Step Cards (Retrieve, Analyze, Synthesize)                     │    │
│  │  ├── Analyst Grid (parallel agent status)                           │    │
│  │  └── Live Metrics (chunks, relevance, timing)                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│                            WebSocket / SSE                                   │
│                                   ▼                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                           BACKEND (FastAPI)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  /api/rag/stream endpoint (Server-Sent Events)                      │    │
│  │  ├── Emits: step_started, step_completed, analyst_update            │    │
│  │  ├── Emits: chunk_analyzed, synthesis_progress                      │    │
│  │  └── Emits: final_response                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                   │                                          │
│                          LangGraph Callbacks                                 │
│                                   ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  RAG Graph with State Streaming                                     │    │
│  │  retrieve → distribute → analyze_chunk (×N) → synthesize            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Backend Streaming Endpoint

**Files to create/modify:**

#### 1.1 Create `backend/app/routers/rag_stream.py`

New SSE streaming endpoint for real-time agent state updates.

```python
# Key features:
# - SSE endpoint: POST /api/rag/stream
# - Custom LangGraph callback handler to emit events
# - Event types: step_started, step_completed, analyst_update, final_response
```

**Event Schema:**

```typescript
interface AgentEvent {
  type: 'pipeline_started' | 'step_started' | 'step_completed' | 
        'analyst_spawned' | 'analyst_completed' | 'synthesis_started' | 
        'final_response' | 'error';
  timestamp: string;
  data: {
    step?: 'retrieve' | 'distribute' | 'analyze' | 'synthesize';
    analyst_id?: string;
    chunk_id?: string;
    relevance_score?: number;
    is_relevant?: boolean;
    summary?: string;
    total_chunks?: number;
    completed_analysts?: number;
    retrieval_count?: number;
    relevant_count?: number;
    final_response?: string;
    sources_used?: string[];
    shared_document?: string;
    duration_ms?: number;
  };
}
```

#### 1.2 Create `backend/app/rag_agent/callbacks.py`

LangGraph callback handler for streaming state updates.

```python
# Implements:
# - on_chain_start: Emit step_started
# - on_chain_end: Emit step_completed with results
# - Custom hooks for analyst completion tracking
```

#### 1.3 Modify `backend/app/rag_agent/graph.py`

Add streaming support to graph execution.

```python
# Changes:
# - Add astream_events() wrapper function
# - Pass callback handler to graph invocation
# - Emit intermediate state after each node
```

#### 1.4 Modify `backend/app/main.py`

Register the new streaming router.

```python
from app.routers import documents, search, chat, rag, rag_stream

app.include_router(rag_stream.router)
```

---

### Phase 2: Frontend State Management

**Files to create/modify:**

#### 2.1 Create `frontend/src/api/ragStream.ts`

SSE client for consuming agent state stream.

```typescript
// Key features:
// - EventSource wrapper with reconnection
// - Type-safe event parsing
// - State accumulation for UI updates
// - AbortController for cancellation

export interface PipelineState {
  status: 'idle' | 'retrieving' | 'distributing' | 'analyzing' | 'synthesizing' | 'complete' | 'error';
  currentStep: number; // 1-4
  totalSteps: 4;
  
  // Retrieval phase
  retrieval: {
    status: 'pending' | 'active' | 'complete';
    chunksRetrieved: number;
    durationMs?: number;
  };
  
  // Analysis phase
  analysis: {
    status: 'pending' | 'active' | 'complete';
    totalAnalysts: number;
    completedAnalysts: number;
    analysts: AnalystState[];
    durationMs?: number;
  };
  
  // Synthesis phase
  synthesis: {
    status: 'pending' | 'active' | 'complete';
    durationMs?: number;
  };
  
  // Results
  result?: {
    finalResponse: string;
    retrievalCount: number;
    relevantCount: number;
    sourcesUsed: string[];
    sharedDocument: string;
  };
  
  error?: string;
  totalDurationMs?: number;
}

export interface AnalystState {
  id: string;
  chunkId: string;
  status: 'pending' | 'analyzing' | 'complete';
  relevanceScore?: number;
  isRelevant?: boolean;
  summary?: string;
  confidence?: 'high' | 'medium' | 'low';
}
```

#### 2.2 Create `frontend/src/hooks/useAgentStream.ts`

React hook for managing agent state stream.

```typescript
// Key features:
// - Manages SSE connection lifecycle
// - Accumulates state updates
// - Provides abort capability
// - Error handling and reconnection

export function useAgentStream() {
  const [state, setState] = useState<PipelineState>(initialState);
  const [isStreaming, setIsStreaming] = useState(false);
  
  const startQuery = async (query: string, filters?: Record<string, any>) => { ... };
  const abort = () => { ... };
  
  return { state, isStreaming, startQuery, abort };
}
```

#### 2.3 Create `frontend/src/components/AgentStateViewer.tsx`

Main component for visualizing agent pipeline state.

```tsx
// Sub-components:
// - PipelineProgress: Horizontal step indicator (4 steps)
// - StepCard: Individual step status with timing
// - AnalystGrid: Grid of analyst agents with status indicators
// - MetricsPanel: Live metrics (chunks, relevance ratio, timing)
// - SharedDocumentViewer: Expandable analyst contributions
```

**Component Structure:**

```
AgentStateViewer
├── PipelineProgress
│   ├── StepIndicator (Retrieve) 
│   ├── StepIndicator (Distribute)
│   ├── StepIndicator (Analyze)
│   └── StepIndicator (Synthesize)
├── CurrentStepDetails
│   ├── RetrievalStep (when active)
│   ├── AnalysisStep (when active)
│   │   └── AnalystGrid
│   │       └── AnalystCard × N
│   └── SynthesisStep (when active)
├── MetricsPanel
│   ├── ChunksRetrieved
│   ├── RelevantChunks
│   ├── AnalystsComplete
│   └── TotalDuration
└── SharedDocumentViewer (expandable)
```

#### 2.4 Modify `frontend/src/components/ChatInterface.tsx`

Integrate AgentStateViewer into chat flow.

```tsx
// Changes:
// - Add useAgentStream hook
// - Show AgentStateViewer during query processing
// - Toggle between simple loading and detailed view
// - Display final response with full source attribution
```

#### 2.5 Update `frontend/src/api/client.ts`

Add streaming endpoint types.

```typescript
// Add new exports for streaming types and helpers
```

---

### Phase 3: UI Components

**Files to create:**

#### 3.1 `frontend/src/components/agent/PipelineProgress.tsx`

```tsx
// Horizontal progress indicator with 4 steps
// Active step pulses, completed steps show checkmark
// Shows step timing on completion
```

#### 3.2 `frontend/src/components/agent/AnalystGrid.tsx`

```tsx
// Grid of analyst cards (4 columns on desktop)
// Each card shows:
// - Chunk ID
// - Status (spinner/check/x)
// - Relevance score bar (0-10)
// - Confidence badge
// - Expandable summary
```

#### 3.3 `frontend/src/components/agent/MetricsPanel.tsx`

```tsx
// Real-time metrics:
// - Chunks retrieved / analyzed
// - Relevant vs non-relevant ratio (pie chart)
// - Elapsed time
// - Estimated completion
```

#### 3.4 `frontend/src/components/agent/SharedDocumentViewer.tsx`

```tsx
// Expandable panel showing analyst contributions
// Markdown formatted
// Collapsible sections per analyst
```

---

### Phase 4: Docker Integration

**Files to modify:**

#### 4.1 `docker-compose.yml`

```yaml
services:
  # ... existing services ...

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
    env_file:
      - .env
    depends_on:
      langgraph-redis:
        condition: service_healthy
      langgraph-postgres:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - REDIS_URI=redis://langgraph-redis:6379
      - DATABASE_URI=postgres://langgraph:langgraph@langgraph-postgres:5432/langgraph?sslmode=disable
      - HF_HUB_ENABLE_HF_TRANSFER=0
      - HF_HUB_DISABLE_XET=1
      - SSL_CERT_FILE=/etc/ssl/certs/bundle.pem
      - REQUESTS_CA_BUNDLE=/etc/ssl/certs/bundle.pem
      - CURL_CA_BUNDLE=/etc/ssl/certs/bundle.pem
      # NEW: Enable SSE streaming
      - STREAM_TIMEOUT=300
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      # NEW: Configure API endpoint for SSE
      - VITE_API_BASE_URL=http://backend:8000
```

#### 4.2 `frontend/nginx.conf`

Update nginx config to properly proxy SSE connections:

```nginx
# Add/modify upstream and location blocks for SSE:
location /api/rag/stream {
    proxy_pass http://backend:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_set_header Cache-Control 'no-cache';
    proxy_set_header Content-Type 'text/event-stream';
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    chunked_transfer_encoding off;
}
```

#### 4.3 `frontend/Dockerfile`

Ensure build args are passed for API configuration:

```dockerfile
# Add ARG and ENV for runtime configuration
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
```

---

## File Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `backend/app/routers/rag_stream.py` | SSE streaming endpoint |
| `backend/app/rag_agent/callbacks.py` | LangGraph callback handler |
| `frontend/src/api/ragStream.ts` | SSE client utilities |
| `frontend/src/hooks/useAgentStream.ts` | React hook for stream state |
| `frontend/src/components/AgentStateViewer.tsx` | Main state viewer component |
| `frontend/src/components/agent/PipelineProgress.tsx` | Step progress indicator |
| `frontend/src/components/agent/AnalystGrid.tsx` | Parallel analyst visualization |
| `frontend/src/components/agent/MetricsPanel.tsx` | Real-time metrics display |
| `frontend/src/components/agent/SharedDocumentViewer.tsx` | Analyst contributions viewer |

### Files to Modify

| File | Changes |
|------|---------|
| `backend/app/main.py` | Register rag_stream router |
| `backend/app/rag_agent/graph.py` | Add streaming support |
| `frontend/src/components/ChatInterface.tsx` | Integrate AgentStateViewer |
| `frontend/src/api/client.ts` | Add streaming types |
| `frontend/nginx.conf` | Configure SSE proxy |
| `docker-compose.yml` | Add streaming environment vars |

---

## Implementation Order

### Week 1: Backend Streaming
1. ✅ Create `callbacks.py` with LangGraph event hooks
2. ✅ Create `rag_stream.py` SSE endpoint
3. ✅ Modify `graph.py` for streaming support
4. ✅ Update `main.py` to register router
5. ✅ Test with curl/httpie

### Week 2: Frontend Foundation
1. ✅ Create `ragStream.ts` SSE client
2. ✅ Create `useAgentStream.ts` hook
3. ✅ Create `AgentStateViewer.tsx` skeleton
4. ✅ Integrate into `ChatInterface.tsx`
5. ✅ Test end-to-end locally

### Week 3: UI Polish
1. ✅ Build `PipelineProgress.tsx` with animations
2. ✅ Build `AnalystGrid.tsx` with status indicators
3. ✅ Build `MetricsPanel.tsx` with live updates
4. ✅ Build `SharedDocumentViewer.tsx`
5. ✅ Add dark theme styling

### Week 4: Docker & Deployment
1. ✅ Update `nginx.conf` for SSE
2. ✅ Update `docker-compose.yml`
3. ✅ Test in Docker environment
4. ✅ Performance optimization
5. ✅ Documentation

---

## Event Flow Example

```
User submits query: "What are the key findings?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: pipeline_started                                     │
│ { type: "pipeline_started", query: "What are the key..." } │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: step_started                                         │
│ { type: "step_started", step: "retrieve" }                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: step_completed                                       │
│ { type: "step_completed", step: "retrieve",                 │
│   data: { retrieval_count: 20, duration_ms: 342 } }         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: step_started                                         │
│ { type: "step_started", step: "analyze",                    │
│   data: { total_analysts: 20 } }                            │
└─────────────────────────────────────────────────────────────┘
    │
    ├──▶ Event: analyst_completed (chunk_0, score: 8, relevant: true)
    ├──▶ Event: analyst_completed (chunk_1, score: 3, relevant: false)
    ├──▶ Event: analyst_completed (chunk_2, score: 9, relevant: true)
    │    ... (20 total, arriving in parallel)
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: step_completed                                       │
│ { type: "step_completed", step: "analyze",                  │
│   data: { completed: 20, relevant: 14, duration_ms: 2103 }} │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: step_started                                         │
│ { type: "step_started", step: "synthesize" }                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Event: final_response                                       │
│ { type: "final_response",                                   │
│   data: {                                                   │
│     final_response: "Based on the analysis...",             │
│     retrieval_count: 20,                                    │
│     relevant_count: 14,                                     │
│     sources_used: ["Report A", "Policy B", ...],            │
│     total_duration_ms: 4521                                 │
│   }                                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│  ● Retrieve    ● Distribute    ◐ Analyze     ○ Synthesize       │
│  ✓ 342ms       ✓ 12ms         ⋯ 1.2s        Waiting...          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Analyzing 20 chunks in parallel                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Analysts: 14/20 complete    Relevant: 11/14 (79%)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ chunk_0 │ │ chunk_1 │ │ chunk_2 │ │ chunk_3 │ │ chunk_4 │   │
│  │ ✓ 8/10  │ │ ✓ 3/10  │ │ ✓ 9/10  │ │ ⋯      │ │ ○       │   │
│  │ ██████░░│ │ ███░░░░░│ │ █████████│ │ ░░░░░░░░│ │ ░░░░░░░░│   │
│  │ high    │ │ low     │ │ high    │ │         │ │         │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ chunk_5 │ │ chunk_6 │ │ chunk_7 │ │ chunk_8 │ │ chunk_9 │   │
│  │ ⋯      │ │ ✓ 7/10  │ │ ✓ 2/10  │ │ ✓ 8/10  │ │ ⋯      │   │
│  │ ░░░░░░░░│ │ ███████░│ │ ██░░░░░░│ │ ████████│ │ ░░░░░░░░│   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ... (10 more)                                                   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  📊 Metrics                                                      │
│  ├── Chunks Retrieved: 20                                        │
│  ├── Relevant Chunks: 11 (55%)                                   │
│  ├── Avg Relevance: 6.2/10                                       │
│  └── Elapsed: 1.5s                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing Commands

```bash
# Test SSE endpoint directly
curl -N -X POST http://localhost:8000/api/rag/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?"}'

# Run full stack in Docker
docker-compose up --build

# Watch logs for streaming events
docker-compose logs -f backend

# Test frontend in development
cd frontend && npm run dev
```

---

## Notes

1. **SSE vs WebSocket**: SSE is simpler for one-way streaming and works better with nginx/proxies
2. **Backpressure**: The backend should buffer events if the client disconnects temporarily
3. **Timeout**: Set appropriate timeouts for long-running queries (5 minutes recommended)
4. **Error Recovery**: Frontend should handle reconnection gracefully
5. **Mobile**: Consider simplified view for mobile with just progress bar
