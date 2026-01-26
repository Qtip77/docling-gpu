# Implementation Plan: Query Understanding & Rewriting with Hybrid Search

## Overview

This plan adds a **Query Understanding/Rewriting** node to preprocess queries before retrieval, improving semantic search results through:

1. **Query Rewriting** - Optimized queries for semantic/vector search
2. **Keyword Extraction** - Exact terms for BM25/keyword search (hybrid search)
3. **Query Decomposition** - Sub-questions for complex multi-part queries
4. **Query Classification** - Type detection for response formatting

### Pipeline Evolution

```
Current:  START → retrieve → distribute (Send API) → analyze_chunk (parallel) → synthesize → END
Proposed: START → understand_query → retrieve → distribute → analyze_chunk → synthesize → END
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Query Understanding Node                      │
├─────────────────────────────────────────────────────────────────┤
│  Input: "What's the max Lambda timeout and cold start fix?"     │
│                                                                  │
│  Outputs:                                                        │
│  ├─ rewritten_query: "AWS Lambda maximum timeout configuration  │
│  │                    and cold start latency optimization"      │
│  ├─ search_keywords: ["Lambda", "timeout", "cold start",        │
│  │                    "15 minutes", "provisioned concurrency"]  │
│  ├─ sub_questions: [                                            │
│  │     "What is the maximum AWS Lambda timeout?",               │
│  │     "How to reduce Lambda cold start latency?"               │
│  │   ]                                                          │
│  └─ is_complex: true                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Node (Hybrid)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  Vector Search   │    │  Keyword Search  │                   │
│  │  (Semantic)      │    │  (BM25)          │                   │
│  ├──────────────────┤    ├──────────────────┤                   │
│  │ rewritten_query  │    │ "Lambda"^2       │                   │
│  │ → embedding      │    │ "timeout"^2      │                   │
│  │ → cosine sim     │    │ "cold start"^2   │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│           ┌───────────────────────┐                              │
│           │  Reciprocal Rank      │                              │
│           │  Fusion (RRF)         │                              │
│           └───────────────────────┘                              │
│                       │                                          │
│                       ▼                                          │
│           ┌───────────────────────┐                              │
│           │  Multi-Query Merge    │  (if is_complex=true)        │
│           │  - Main query results │                              │
│           │  - Sub-question 1     │                              │
│           │  - Sub-question 2     │                              │
│           │  → RRF fusion         │                              │
│           └───────────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repo Alignment Notes (Docling RAG current codebase)

> Markup notes to keep this plan executable against the current repository structure and APIs.

### ✅ Keep
- Add an `understand_query` node before retrieval: fits your existing LangGraph pipeline.
- Use Pydantic structured output + prompts: matches the current analyst/synthesizer implementation style.
- Multi-query + RRF: valuable and can be added without changing analyst/synthesizer logic.

### ⚠️ Must Change (repo mismatches)
- **Graph export name**: this repo’s LangGraph config references `./app/rag_agent/graph.py:graph` (`backend/langgraph.json`). Keep the compiled graph variable exported as `graph`.
- **Env example filename**: this repo uses `env.example` at the repo root (not `.env.example`).
- **Retrieval stack mismatch**: the plan introduces `backend/app/services/search_service.py` + `embeddings_service` + `azure_search_client`, which do not exist here. Extend one of the existing retrieval paths instead (see Phase 5 markup).
- **SSE format mismatch**: frontend expects SSE `data: <json>` where json has `{ type, timestamp, data }` (`frontend/src/api/ragStream.ts`). Avoid switching to `event:`-style SSE unless you update the frontend parser.
- **Azure “semantic” ranking assumptions**: the plan uses `query_type="semantic"` + `semantic_configuration_name="default"`, but your index creation code does not configure semantic search (`backend/app/services/azure_search.py`).

## File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/app/rag_agent/state.py` | MODIFY | Add query understanding fields to `OrchestratorState` |
| `backend/app/rag_agent/schemas/query_analysis.py` | CREATE | Pydantic schema for structured LLM output |
| `backend/app/rag_agent/prompts/query_prompts.py` | CREATE | System and user prompts for query understanding |
| `backend/app/rag_agent/nodes/query_processor.py` | CREATE | Query understanding node implementation |
| `backend/app/rag_agent/nodes/retriever.py` | MODIFY | Use `rewritten_query` when present; optionally add multi-query + RRF (repo-aligned approach in Phase 5) |
| `backend/app/rag_agent/tools/search.py` | MODIFY | (Recommended) implement hybrid/keyword/multi-query retrieval here to keep agent retrieval centralized |
| `backend/app/services/azure_search.py` | MODIFY | (Optional) add an “agent-ready” hybrid retrieval helper using Azure SDK, if you move agent retrieval off LangChain |
| `backend/app/rag_agent/graph.py` | MODIFY | Add node, update edges, update initial state (keep exported compiled graph named `graph`) |
| `backend/app/rag_agent/callbacks.py` | MODIFY | Optional: add a new step/event for query understanding (keep existing SSE JSON format) |
| `backend/app/routers/rag_stream.py` | MODIFY | Detect `understand_query` node start/end via `astream_events` and emit progress events |
| `backend/app/rag_agent/config.py` | MODIFY | Add query rewrite feature flags (or create a separate config module; current file is LLM config) |
| `env.example` | MODIFY | Add new environment variables |

---

## Phase 1: State Schema Updates

### 1.1 Update State: `backend/app/rag_agent/state.py`

Add the following fields to `OrchestratorState`:

> Repo note: your current `backend/app/rag_agent/state.py` already uses `operator.add` as the list reducer and defines `concat_markdown`. Do **not** add a second `add()` reducer function as shown in the snippet below; follow the existing patterns in that file.

> Optional cleanup (plan-doc only): to avoid confusion, delete or ignore the `def add(left, right)` function inside the snippet below when you implement this in the repo.

```python
"""RAG Agent State definitions."""
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, TypedDict
from langchain_core.documents import Document


def add(left: list, right: list) -> list:
    """Reducer that appends lists for parallel node outputs."""
    return left + right


def concat_markdown(left: str, right: str) -> str:
    """Reducer that concatenates markdown sections."""
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n---\n\n{right}"


class OrchestratorState(TypedDict):
    """State schema for the RAG orchestration graph.
    
    This state flows through all nodes in the pipeline:
    understand_query → retrieve → distribute → analyze_chunk → synthesize
    """
    
    # === Input ===
    query: str  # Original user query
    filters: Optional[Dict[str, Any]]  # Metadata filters for retrieval
    
    # === Query Understanding Outputs ===
    # Populated by the understand_query node
    rewritten_query: str  # Optimized query for semantic/vector search
    query_type: Literal["factual", "comparison", "procedural", "exploratory", "definition"]
    key_concepts: List[str]  # Core concepts extracted (3-5)
    search_keywords: List[str]  # Exact terms for keyword/BM25 search (3-7)
    sub_questions: List[str]  # Decomposed questions for complex queries (0-3)
    is_complex_query: bool  # Whether query required decomposition
    
    # === Retrieval Outputs ===
    retrieved_chunks: List[Document]
    retrieval_count: int
    
    # === Analysis Outputs ===
    # Use reducers for keys written by parallel analyst nodes
    analyses: Annotated[List[Any], add]
    shared_document: Annotated[str, concat_markdown]
    
    # === Final Outputs ===
    final_response: str
    relevant_count: int
    sources_used: List[str]
```

---

## Phase 2: Query Analysis Schema

### 2.1 Create Schema: `backend/app/rag_agent/schemas/query_analysis.py`

```python
"""Pydantic schemas for query analysis structured output."""
from typing import List, Literal
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured output schema for query understanding.
    
    Used with LLM's with_structured_output() to ensure
    consistent, validated responses from the query analysis step.
    """
    
    rewritten_query: str = Field(
        description=(
            "Optimized query for semantic/vector search. "
            "Expand acronyms (e.g., 'ML' → 'machine learning (ML)'), "
            "add relevant synonyms in parentheses, "
            "make implicit context explicit. "
            "Keep under 100 words. Preserve original intent."
        )
    )
    
    query_type: Literal["factual", "comparison", "procedural", "exploratory", "definition"] = Field(
        description=(
            "Query classification: "
            "factual=specific facts/numbers/dates/named entities, "
            "comparison=comparing multiple items/pros-cons/differences, "
            "procedural=how-to/step-by-step processes/instructions, "
            "exploratory=open-ended research/overviews/summaries, "
            "definition=what-is/explanations/concept definitions"
        )
    )
    
    key_concepts: List[str] = Field(
        description=(
            "3-5 core concepts essential for retrieval, ordered by importance. "
            "Use canonical forms (e.g., 'machine learning' not 'ML'). "
            "Include both specific terms and broader categories when relevant."
        ),
        min_length=1,
        max_length=5
    )
    
    search_keywords: List[str] = Field(
        description=(
            "3-7 precise keywords for keyword/BM25 search. "
            "Include: exact technical terms, error codes (e.g., 'HTTP 503', 'ECONNREFUSED'), "
            "product/service names (e.g., 'Azure Blob Storage'), "
            "acronyms in ORIGINAL form (e.g., 'ML', 'API', 'REST'), "
            "version numbers/identifiers (e.g., 'v2.1', 'GPT-4'), "
            "domain-specific jargon that embeddings might miss. "
            "These should be exact-match terms, NOT semantic concepts."
        ),
        min_length=1,
        max_length=7
    )
    
    sub_questions: List[str] = Field(
        default_factory=list,
        description=(
            "For complex multi-part queries, decompose into 1-3 simpler, "
            "self-contained questions that can be answered independently. "
            "Leave empty [] for simple, focused queries. "
            "Each sub-question should target a distinct information need."
        ),
        max_length=3
    )
    
    is_complex: bool = Field(
        description=(
            "True if the query has multiple distinct information needs "
            "and was decomposed into sub-questions. False for simple queries."
        )
    )
```

---

## Phase 3: Prompts

### 3.1 Create Prompts: `backend/app/rag_agent/prompts/query_prompts.py`

```python
"""Prompts for query understanding and rewriting."""

QUERY_UNDERSTANDING_SYSTEM = """You are a query analysis expert specializing in document retrieval optimization.

Your task is to transform user queries into forms optimized for HYBRID SEARCH, which combines:
1. Semantic/vector search (using embeddings for meaning similarity)
2. Keyword/BM25 search (using exact term matching)

## Your Responsibilities

### 1. Rewritten Query (for Semantic/Vector Search)
Optimize the query for embedding-based similarity search:
- Expand acronyms: "ML" → "machine learning (ML)"
- Add relevant synonyms: "performance" → "performance (speed, efficiency, throughput)"
- Make implicit context explicit
- Keep under 100 words
- PRESERVE the original intent - do not add unrelated concepts

### 2. Search Keywords (for Keyword/BM25 Search)
Extract 3-7 EXACT terms that should match verbatim in documents:
- Technical terms exactly as written
- Error codes and status codes (e.g., "HTTP 503", "ECONNREFUSED", "errno 111")
- Product/service names (e.g., "Azure Blob Storage", "Amazon S3", "PostgreSQL")
- Acronyms in ORIGINAL form (e.g., "ML", "API", "REST", "gRPC")
- Version numbers and identifiers (e.g., "v2.1", "GPT-4", "Python 3.11")
- Domain-specific jargon that embedding models might not capture well

Do NOT include common words, stopwords, or purely conceptual terms as keywords.

### 3. Sub-Questions (for Complex Queries)
Only decompose if the query has MULTIPLE DISTINCT information needs:
- Each sub-question should be independently answerable
- Maximum 3 sub-questions
- Order by logical dependency (answer Q1 before Q2 if Q2 depends on Q1)
- Leave empty [] for simple, focused queries

### 4. Query Type Classification
- **factual**: Specific facts, numbers, dates, named entities ("What is X?", "When did Y?")
- **comparison**: Comparing items, pros/cons, differences ("X vs Y", "compare", "difference between")
- **procedural**: How-to, step-by-step, instructions ("How do I?", "steps to", "guide for")
- **exploratory**: Open-ended research, overviews ("Tell me about", "overview of", "explain")
- **definition**: Concept definitions, explanations ("What is", "define", "meaning of")

## Examples

### Example 1: Simple Technical Query
**Input:** "What's the max file size for S3 uploads?"

**Output:**
- rewritten_query: "maximum file size limit for Amazon S3 (Simple Storage Service) object uploads and size restrictions"
- query_type: "factual"
- key_concepts: ["Amazon S3", "file size limit", "object upload", "storage restrictions"]
- search_keywords: ["S3", "max file size", "upload limit", "5TB", "5 GB", "multipart upload"]
- sub_questions: []
- is_complex: false

### Example 2: Complex Comparison Query
**Input:** "Compare Lambda cold starts vs ECS startup time and which is better for real-time APIs"

**Output:**
- rewritten_query: "comparison of AWS Lambda function cold start latency versus Amazon ECS (Elastic Container Service) container startup time for real-time low-latency API applications"
- query_type: "comparison"
- key_concepts: ["AWS Lambda cold start", "Amazon ECS startup", "real-time APIs", "latency comparison"]
- search_keywords: ["Lambda", "cold start", "ECS", "startup time", "latency", "provisioned concurrency", "Fargate"]
- sub_questions: [
    "What causes AWS Lambda cold starts and what is the typical latency?",
    "What is the container startup time for Amazon ECS tasks?",
    "Which AWS compute service provides lower latency for real-time API workloads?"
  ]
- is_complex: true

### Example 3: Error Troubleshooting Query
**Input:** "Fix ECONNREFUSED error in Node.js when connecting to Redis"

**Output:**
- rewritten_query: "troubleshoot and resolve ECONNREFUSED connection refused error in Node.js application when connecting to Redis database server"
- query_type: "procedural"
- key_concepts: ["ECONNREFUSED error", "Node.js Redis connection", "connection troubleshooting"]
- search_keywords: ["ECONNREFUSED", "Node.js", "Redis", "connection refused", "errno 111", "ioredis", "node-redis"]
- sub_questions: []
- is_complex: false

## Important Notes
- Always preserve the user's original intent
- Do not hallucinate or add information not implied by the query
- Keywords should be terms likely to appear VERBATIM in technical documents
- Sub-questions should only be used for genuinely complex, multi-part queries"""


QUERY_UNDERSTANDING_USER = """Analyze and optimize this query for hybrid document retrieval:

**Original Query:** {query}

Provide your analysis with:
1. rewritten_query - Optimized for semantic/vector search
2. query_type - One of: factual, comparison, procedural, exploratory, definition
3. key_concepts - 3-5 core concepts ordered by importance
4. search_keywords - 3-7 exact terms for keyword/BM25 search
5. sub_questions - Decomposed questions if complex, otherwise empty list
6. is_complex - Boolean indicating if decomposition was needed"""
```

---

## Phase 4: Query Understanding Node

### 4.1 Create Node: `backend/app/rag_agent/nodes/query_processor.py`

> Repo note: this repo’s streaming UX does **not** pass a `StreamWriter` into node functions. The SSE endpoint streams by observing `graph.astream_events(..., version="v2")` and mapping `on_chain_start/on_chain_end` events by node name (`backend/app/routers/rag_stream.py`). Plan to emit query-understanding progress by detecting `name == "understand_query"` in that stream loop (Phase 7), rather than relying on `writer` being non-null here.

```python
"""Query understanding and rewriting node for the RAG pipeline."""
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import StreamWriter

from ..config import get_analyst_llm, get_query_rewrite_config
from ..schemas.query_analysis import QueryAnalysis
from ..prompts.query_prompts import (
    QUERY_UNDERSTANDING_SYSTEM,
    QUERY_UNDERSTANDING_USER,
)
from ..state import OrchestratorState

logger = logging.getLogger(__name__)

> Repo note: `backend/app/rag_agent/config.py` currently exposes `get_analyst_llm()` / `get_orchestrator_llm()` only. The plan’s `get_query_rewrite_config()` does not exist yet; add it (or create a new `runtime_config` module) before implementing this node.


async def understand_query(
    state: OrchestratorState,
    writer: StreamWriter = None,
) -> Dict[str, Any]:
    """
    Analyze and rewrite the user query for optimal hybrid retrieval.
    
    This node performs:
    1. Query rewriting for semantic search optimization
    2. Keyword extraction for BM25/keyword search
    3. Query decomposition for complex multi-part questions
    4. Query type classification for response formatting
    
    Args:
        state: Current graph state containing the original query
        writer: Optional stream writer for emitting progress events
    
    Returns:
        State updates for query understanding fields
    """
    config = get_query_rewrite_config()
    original_query = state["query"]
    
    # Emit start event if streaming
    if writer:
        writer({
            "event": "query_understanding_started",
            "data": {"original_query": original_query}
        })
    
    # Bypass mode: pass through original query unchanged
    if not config.get("enabled", True):
        logger.info("Query rewriting disabled, using original query")
        result = {
            "rewritten_query": original_query,
            "query_type": "exploratory",
            "key_concepts": [],
            "search_keywords": [],
            "sub_questions": [],
            "is_complex_query": False,
        }
        if writer:
            writer({"event": "query_understanding_skipped", "data": result})
        return result
    
    try:
        # Get LLM with structured output for reliable parsing
        llm = get_analyst_llm()
        structured_llm = llm.with_structured_output(QueryAnalysis)
        
        # Build prompt messages
        messages = [
            SystemMessage(content=QUERY_UNDERSTANDING_SYSTEM),
            HumanMessage(content=QUERY_UNDERSTANDING_USER.format(query=original_query)),
        ]
        
        # Invoke LLM with structured output
        analysis: QueryAnalysis = await structured_llm.ainvoke(messages)
        
        # Log the transformation
        logger.info(
            f"Query analyzed: '{original_query[:50]}...' → "
            f"type={analysis.query_type}, "
            f"keywords={analysis.search_keywords}, "
            f"is_complex={analysis.is_complex}"
        )
        
        if analysis.rewritten_query != original_query:
            logger.debug(f"Query rewritten to: '{analysis.rewritten_query[:100]}...'")
        
        if analysis.sub_questions:
            logger.debug(f"Decomposed into {len(analysis.sub_questions)} sub-questions")
        
        result = {
            "rewritten_query": analysis.rewritten_query,
            "query_type": analysis.query_type,
            "key_concepts": analysis.key_concepts,
            "search_keywords": analysis.search_keywords,
            "sub_questions": analysis.sub_questions,
            "is_complex_query": analysis.is_complex,
        }
        
        # Emit completion event if streaming
        if writer:
            writer({
                "event": "query_understanding_completed",
                "data": {
                    "original_query": original_query,
                    "rewritten_query": analysis.rewritten_query,
                    "query_type": analysis.query_type,
                    "key_concepts": analysis.key_concepts,
                    "search_keywords": analysis.search_keywords,
                    "sub_questions": analysis.sub_questions,
                    "is_complex": analysis.is_complex,
                    "was_rewritten": analysis.rewritten_query != original_query,
                }
            })
        
        return result
        
    except Exception as e:
        # Graceful degradation: use original query on any failure
        logger.warning(
            f"Query understanding failed for '{original_query[:50]}...': {e}",
            exc_info=True
        )
        
        result = {
            "rewritten_query": original_query,
            "query_type": "exploratory",
            "key_concepts": [],
            "search_keywords": [],
            "sub_questions": [],
            "is_complex_query": False,
        }
        
        if writer:
            writer({
                "event": "query_understanding_failed",
                "data": {"error": str(e), "fallback": result}
            })
        
        return result
```

---

## Phase 5: Hybrid Search Implementation

### 5.1 Update Retriever: `backend/app/rag_agent/nodes/retriever.py`

> Repo note (important): this snippet imports `...services.search_service` which does not exist in this repo. Prefer a repo-aligned approach:
>
> - **Minimal (recommended first):** update `backend/app/rag_agent/nodes/retriever.py` to retrieve using `state.get("rewritten_query") or state["query"]` and keep calling the existing `retrieve_chunks()` from `backend/app/rag_agent/tools/search.py`.
> - **Then add multi-query + RRF** inside `backend/app/rag_agent/nodes/retriever.py` by calling `retrieve_chunks()` for the main rewritten query + each `sub_question` and fusing results by `chunk_id`.
> - Treat “keyword boosting” as optional: LangChain’s `AzureAISearchRetriever` does not expose Azure’s full query syntax knobs cleanly; you can approximate by appending quoted keywords to the query string, but validate results.

```python
"""Document retrieval node with hybrid search and multi-query support."""
import asyncio
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langgraph.types import StreamWriter

from ..state import OrchestratorState
from ..config import get_retrieval_config
from ...services.search_service import hybrid_search

logger = logging.getLogger(__name__)


async def retrieve_documents(
    state: OrchestratorState,
    writer: StreamWriter = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant document chunks using hybrid search.
    
    Retrieval Strategy:
    1. Simple queries: Single hybrid search with rewritten query + keywords
    2. Complex queries: Multi-query retrieval for main query + sub-questions,
       then merge results using Reciprocal Rank Fusion (RRF)
    
    Hybrid search combines:
    - Semantic/vector search using the rewritten query
    - Keyword/BM25 search using extracted search_keywords
    
    Args:
        state: Current graph state with query understanding outputs
        writer: Optional stream writer for progress events
    
    Returns:
        State updates for retrieved_chunks and retrieval_count
    """
    config = get_retrieval_config()
    filters = state.get("filters")
    
    # Get query understanding outputs
    rewritten_query = state.get("rewritten_query") or state["query"]
    search_keywords = state.get("search_keywords", [])
    sub_questions = state.get("sub_questions", [])
    is_complex = state.get("is_complex_query", False)
    
    # Log retrieval strategy
    if rewritten_query != state["query"]:
        logger.info(f"Using rewritten query: '{rewritten_query[:80]}...'")
    if search_keywords:
        logger.info(f"Using keywords for BM25: {search_keywords}")
    
    # Emit start event
    if writer:
        writer({
            "event": "retrieval_started",
            "data": {
                "strategy": "multi_query" if is_complex else "single_query",
                "num_sub_questions": len(sub_questions),
            }
        })
    
    try:
        if is_complex and sub_questions and config.get("enable_multi_query", True):
            # Multi-query retrieval for complex queries
            chunks = await _multi_query_retrieve(
                main_query=rewritten_query,
                sub_questions=sub_questions,
                keywords=search_keywords,
                filters=filters,
                top_k_per_query=config.get("top_k_per_query", 10),
                final_top_k=config.get("final_top_k", 20),
                writer=writer,
            )
            logger.info(
                f"Multi-query retrieval: 1 main + {len(sub_questions)} sub-questions → "
                f"{len(chunks)} unique chunks"
            )
        else:
            # Standard single hybrid search
            chunks = await hybrid_search(
                query=rewritten_query,
                keywords=search_keywords,
                filters=filters,
                top_k=config.get("top_k", 20),
            )
            logger.info(f"Single hybrid search → {len(chunks)} chunks")
        
        # Emit completion event
        if writer:
            writer({
                "event": "retrieval_completed",
                "data": {
                    "chunk_count": len(chunks),
                    "chunk_ids": [c.metadata.get("chunk_id") for c in chunks[:5]],  # First 5
                }
            })
        
        return {
            "retrieved_chunks": chunks,
            "retrieval_count": len(chunks),
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        if writer:
            writer({"event": "retrieval_failed", "data": {"error": str(e)}})
        # Return empty results on failure - let synthesizer handle gracefully
        return {
            "retrieved_chunks": [],
            "retrieval_count": 0,
        }


async def _multi_query_retrieve(
    main_query: str,
    sub_questions: List[str],
    keywords: List[str],
    filters: dict,
    top_k_per_query: int,
    final_top_k: int,
    writer: StreamWriter = None,
) -> List[Document]:
    """
    Retrieve documents for multiple queries and merge using RRF.
    
    Executes hybrid search for:
    1. The main rewritten query
    2. Each sub-question
    
    Then combines results using Reciprocal Rank Fusion (RRF) to produce
    a final ranked list that balances relevance across all queries.
    
    Args:
        main_query: Primary rewritten query
        sub_questions: List of decomposed sub-questions
        keywords: Search keywords (shared across all queries)
        filters: Metadata filters
        top_k_per_query: Results to retrieve per query
        final_top_k: Final number of results after fusion
        writer: Optional stream writer
    
    Returns:
        List of documents ranked by RRF score
    """
    all_queries = [main_query] + sub_questions
    
    # Log multi-query execution
    logger.debug(f"Executing {len(all_queries)} queries in parallel")
    
    # Execute all queries in parallel
    tasks = [
        hybrid_search(
            query=q,
            keywords=keywords,  # Same keywords boost for all queries
            filters=filters,
            top_k=top_k_per_query,
        )
        for q in all_queries
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any failed queries gracefully
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Sub-query {i} failed: {result}")
        else:
            valid_results.append(result)
    
    if not valid_results:
        logger.error("All queries failed in multi-query retrieval")
        return []
    
    # Reciprocal Rank Fusion (RRF)
    # RRF score = Σ 1/(k + rank) for each query where doc appears
    # k is a constant (typically 60) that dampens the effect of high ranks
    k = 60
    chunk_scores: Dict[str, float] = {}
    chunk_docs: Dict[str, Document] = {}
    
    for query_idx, query_results in enumerate(valid_results):
        for rank, doc in enumerate(query_results):
            # Use chunk_id as unique identifier, fallback to content hash
            chunk_id = doc.metadata.get("chunk_id") or hash(doc.page_content[:200])
            chunk_id = str(chunk_id)
            
            rrf_score = 1.0 / (k + rank + 1)
            
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += rrf_score
            else:
                chunk_scores[chunk_id] = rrf_score
                chunk_docs[chunk_id] = doc
    
    # Sort by combined RRF score (descending) and take top_k
    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:final_top_k]
    
    # Return documents in RRF-ranked order
    ranked_docs = [chunk_docs[chunk_id] for chunk_id, score in sorted_chunks]
    
    logger.debug(
        f"RRF fusion: {sum(len(r) for r in valid_results)} total → "
        f"{len(chunk_scores)} unique → {len(ranked_docs)} final"
    )
    
    return ranked_docs
```

### 5.2 Add/Update Search Service: `backend/app/services/search_service.py`

Add the `hybrid_search` function to your existing search service:

> Repo note: this repo does not currently have `backend/app/services/search_service.py`, `embeddings_service.py`, or `azure_search_client.py`.
>
> If you decide to implement an Azure SDK-based hybrid search helper, base it on the existing `backend/app/services/azure_search.py`:
> - It already has `get_search_client()` and uses `VectorizableTextQuery(text=...)` (no client-side embedding needed).
> - It currently does *not* configure semantic ranking on the index. Avoid `query_type="semantic"` until you add semantic config to index creation.

```python
"""Azure AI Search service with hybrid search support."""
import logging
from typing import Any, Dict, List, Optional

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from langchain_core.documents import Document

from .embeddings_service import get_embeddings
from .azure_search_client import get_search_client

> Repo note: the Azure SDK usage in this snippet does not match the code you already have.
> In this repo you already use `VectorizableTextQuery(text=..., k_nearest_neighbors=..., fields="content_vector")`
> in `backend/app/services/azure_search.py`. Prefer that pattern if you implement an SDK-based helper.

logger = logging.getLogger(__name__)


async def hybrid_search(
    query: str,
    keywords: List[str] = None,
    filters: Dict[str, Any] = None,
    top_k: int = 20,
) -> List[Document]:
    """
    Perform hybrid search combining semantic vectors and keyword matching.
    
    Azure AI Search hybrid search uses:
    - Vector search: Embedding similarity on content_vector field
    - Full-text search: BM25 on content field with keyword boosting
    - Score fusion: Reciprocal Rank Fusion (RRF) to combine results
    
    Args:
        query: Rewritten/optimized query for semantic search
        keywords: Exact keywords for BM25 boosting (from query understanding)
        filters: Metadata filters (e.g., {"doc_id": "...", "doc_type": "report"})
        top_k: Number of results to return
    
    Returns:
        List of Document objects with content and metadata
    """
    search_client = get_search_client()
    embeddings = get_embeddings()
    
    # Generate embedding for vector search
    try:
        query_embedding = await embeddings.aembed_query(query)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        raise
    
    # Build vector query for semantic search
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    
    # Build search text with keyword boosting
    # Keywords get ^2 boost for exact matches
    if keywords:
        # Quote keywords for exact phrase matching, boost with ^2
        keyword_clauses = [f'"{kw}"^2' for kw in keywords if kw]
        search_text = f"{query} {' '.join(keyword_clauses)}"
    else:
        search_text = query
    
    # Build OData filter expression
    filter_expr = _build_filter_expression(filters) if filters else None
    
    # Execute hybrid search
    try:
        results = search_client.search(
            search_text=search_text,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=top_k,
            query_type="semantic",  # Enable semantic ranking
            semantic_configuration_name="default",  # Must match your index config
            select=[
                "chunk_id",
                "content", 
                "filename",
                "page_numbers",
                "section_title",
                "doc_id",
                "created_date",
            ],
        )
    except Exception as e:
        logger.error(f"Azure Search query failed: {e}")
        raise
    
    # Convert to LangChain Document objects
    documents = []
    for result in results:
        doc = Document(
            page_content=result.get("content", ""),
            metadata={
                "chunk_id": result.get("chunk_id"),
                "filename": result.get("filename"),
                "page_numbers": result.get("page_numbers", []),
                "section_title": result.get("section_title"),
                "doc_id": result.get("doc_id"),
                "created_date": result.get("created_date"),
                # Search scores for debugging/logging
                "search_score": result.get("@search.score"),
                "reranker_score": result.get("@search.reranker_score"),
            }
        )
        documents.append(doc)
    
    logger.debug(
        f"Hybrid search returned {len(documents)} results "
        f"(query: '{query[:50]}...', keywords: {keywords})"
    )
    
    return documents


def _build_filter_expression(filters: Dict[str, Any]) -> Optional[str]:
    """
    Build OData filter expression for Azure AI Search.
    
    Supports:
    - String equality: {"field": "value"} → field eq 'value'
    - List membership: {"field": ["a", "b"]} → search.in(field, 'a,b')
    - Boolean: {"field": True} → field eq true
    - Numeric: {"field": 123} → field eq 123
    - None values are skipped
    
    Args:
        filters: Dictionary of field names to filter values
    
    Returns:
        OData filter string or None if no valid filters
    """
    if not filters:
        return None
    
    expressions = []
    
    for key, value in filters.items():
        if value is None:
            continue
            
        if isinstance(value, list):
            if not value:  # Skip empty lists
                continue
            # Use search.in for list membership
            values_str = ",".join(str(v) for v in value)
            expressions.append(f"search.in({key}, '{values_str}')")
        elif isinstance(value, str):
            # Escape single quotes in string values
            escaped = value.replace("'", "''")
            expressions.append(f"{key} eq '{escaped}'")
        elif isinstance(value, bool):
            expressions.append(f"{key} eq {str(value).lower()}")
        elif isinstance(value, (int, float)):
            expressions.append(f"{key} eq {value}")
        else:
            logger.warning(f"Unsupported filter type for {key}: {type(value)}")
    
    return " and ".join(expressions) if expressions else None


# Keep existing retrieve_chunks function for backward compatibility
async def retrieve_chunks(
    query: str,
    top_k: int = 20,
    filters: Dict[str, Any] = None,
) -> List[Document]:
    """
    Legacy retrieval function - wraps hybrid_search without keywords.
    
    Maintained for backward compatibility with existing code.
    """
    return await hybrid_search(
        query=query,
        keywords=None,
        filters=filters,
        top_k=top_k,
    )
```

---

## Phase 6: Graph Construction

### 6.1 Update Graph: `backend/app/rag_agent/graph.py`

> Repo note: keep the exported compiled graph named `graph` because `backend/langgraph.json` references `./app/rag_agent/graph.py:graph`.
>
> Also update **both** initial-state builders in this repo:
> - `backend/app/rag_agent/graph.py:get_initial_state()`
> - `backend/app/routers/rag_stream.py` (it currently constructs `initial_state` inline)
> so the new query-understanding keys exist during streaming runs.

```python
"""RAG Agent graph construction using LangGraph."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from .state import OrchestratorState
from .nodes.query_processor import understand_query
from .nodes.retriever import retrieve_documents
from .nodes.distributor import distribute_to_analysts
from .nodes.analyst import analyze_chunk
from .nodes.synthesizer import synthesize_response


def get_initial_state(query: str, filters: dict = None) -> OrchestratorState:
    """
    Build initial state for RAG query invocation.
    
    Args:
        query: User's original query string
        filters: Optional metadata filters for retrieval
    
    Returns:
        Initialized state dictionary with all required fields
    """
    return {
        # === Input ===
        "query": query,
        "filters": filters,
        
        # === Query Understanding Outputs ===
        # Populated by understand_query node
        "rewritten_query": "",
        "query_type": "exploratory",
        "key_concepts": [],
        "search_keywords": [],
        "sub_questions": [],
        "is_complex_query": False,
        
        # === Retrieval Outputs ===
        "retrieved_chunks": [],
        "retrieval_count": 0,
        
        # === Analysis Outputs ===
        # These use reducers, so start with empty defaults
        "analyses": [],
        "shared_document": "",
        
        # === Final Outputs ===
        "final_response": "",
        "relevant_count": 0,
        "sources_used": [],
    }


def build_rag_graph():
    """
    Construct the RAG orchestration graph.
    
    Pipeline flow:
    START → understand_query → retrieve → [distribute] → analyze_chunk (parallel) → synthesize → END
    
    Returns:
        Compiled LangGraph graph ready for invocation
    """
    # Initialize graph with state schema
    builder = StateGraph(OrchestratorState)
    
    # Configure retry policy for LLM-dependent nodes
    # Retries help handle transient API failures
    llm_retry = RetryPolicy(
        initial_interval=0.5,  # Start with 500ms delay
        backoff_factor=2.0,    # Double delay each retry
        max_interval=10.0,     # Cap at 10 seconds
        max_attempts=3,        # Maximum 3 attempts
        jitter=True,           # Add randomness to prevent thundering herd
    )
    
    # Add nodes to the graph
    builder.add_node("understand_query", understand_query, retry=llm_retry)
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node("analyze_chunk", analyze_chunk, retry=llm_retry)
    builder.add_node("synthesize", synthesize_response, retry=llm_retry)
    
    # Define edges (control flow)
    
    # Entry point: START → understand_query
    builder.add_edge(START, "understand_query")
    
    # understand_query → retrieve
    builder.add_edge("understand_query", "retrieve")
    
    # retrieve → conditional fan-out to analysts OR direct to synthesize
    # distribute_to_analysts returns either:
    # - List of Send() commands to spawn parallel analysts
    # - "synthesize" string to skip analysis (no relevant chunks)
    builder.add_conditional_edges(
        "retrieve",
        distribute_to_analysts,
        ["analyze_chunk", "synthesize"],
    )
    
    # All analyst nodes fan-in to synthesize
    builder.add_edge("analyze_chunk", "synthesize")
    
    # synthesize → END
    builder.add_edge("synthesize", END)
    
    # Compile and return the graph
    return builder.compile()


# Create singleton graph instance
# This is compiled once at module load time
graph = build_rag_graph()
```

---

## Phase 7: Callback & Streaming Updates

### 7.1 Update Callbacks: `backend/app/rag_agent/callbacks.py`

Add new event types for query understanding:

> Repo note: your current event contract is `data: { type, timestamp, data }` (`backend/app/rag_agent/callbacks.py`) and the frontend parses only `data:` lines (`frontend/src/api/ragStream.ts`). If you add new event types like `query_understood`, keep the same JSON envelope.
>
> Also note: adding `StepName.UNDERSTAND_QUERY` would require updating the frontend `StepName` union and potentially expanding the 4-step UI. A lower-impact approach is to emit a dedicated `type: "query_understood"` event and let the frontend optionally display it without changing the progress bar.

```python
"""Callback system for RAG pipeline events."""
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class EventType(str, Enum):
    """Event types emitted during RAG pipeline execution."""
    
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    
    # Step tracking
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    
    # Query understanding events (NEW)
    QUERY_UNDERSTANDING_STARTED = "query_understanding_started"
    QUERY_UNDERSTANDING_COMPLETED = "query_understanding_completed"
    QUERY_UNDERSTANDING_SKIPPED = "query_understanding_skipped"
    QUERY_UNDERSTANDING_FAILED = "query_understanding_failed"
    
    # Retrieval events
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    RETRIEVAL_FAILED = "retrieval_failed"
    
    # Analysis events
    ANALYST_SPAWNED = "analyst_spawned"
    ANALYST_COMPLETED = "analyst_completed"
    ANALYST_FAILED = "analyst_failed"
    
    # Synthesis events
    SYNTHESIS_STARTED = "synthesis_started"
    SYNTHESIS_STREAMING = "synthesis_streaming"
    SYNTHESIS_COMPLETED = "synthesis_completed"
    
    # Final output
    FINAL_RESPONSE = "final_response"
    
    # Errors
    ERROR = "error"


class StepName(str, Enum):
    """Named steps in the RAG pipeline for progress tracking."""
    UNDERSTAND_QUERY = "understand_query"
    RETRIEVE = "retrieve"
    DISTRIBUTE = "distribute"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"


class AgentEvent(BaseModel):
    """Structured event emitted by the RAG agent."""
    type: EventType
    timestamp: datetime
    step: Optional[StepName] = None
    data: Dict[str, Any]
    
    class Config:
        use_enum_values = True


class QueryUnderstandingData(BaseModel):
    """Data payload for query understanding events."""
    original_query: str
    rewritten_query: Optional[str] = None
    query_type: Optional[str] = None
    key_concepts: List[str] = []
    search_keywords: List[str] = []
    sub_questions: List[str] = []
    is_complex: bool = False
    was_rewritten: bool = False
    error: Optional[str] = None
```

### 7.2 Update SSE Router: `backend/app/routers/rag_stream.py`

Add handler for query understanding events:

> Repo note: the snippet below switches to `event: ...` SSE messages. Do not do that in this repo unless you also update the frontend parser. Prefer continuing to emit `data: <json>` using the existing `AgentEvent.to_sse()` helper.
>
> Repo-aligned implementation approach:
> - In the `graph.astream_events(..., version="v2")` loop, detect:
>   - `event_type == "on_chain_start"` and `event_name == "understand_query"` → emit `step_started` or a dedicated `query_understanding_started` event.
>   - `event_type == "on_chain_end"` and `event_name == "understand_query"` → emit `query_understood` (include rewritten query + keywords + is_complex).
> - Move the existing “pre-emit retrieve step started” logic to occur after query understanding completes, so steps appear in the correct order.

```python
# Add to your existing SSE event handler

async def format_sse_event(event: AgentEvent) -> str:
    """Format an AgentEvent as an SSE message.
    
    NOTE (this repo): prefer emitting `data: <json>` using the existing
    AgentEvent.to_sse() format instead of `event: ...`.
    This function is illustrative only unless you also update the frontend parser.
    """
    
    if event.type == EventType.QUERY_UNDERSTANDING_COMPLETED:
        return f"event: query_understood\ndata: {json.dumps({
            'type': 'query_understood',
            'timestamp': event.timestamp.isoformat(),
            'data': {
                'original_query': event.data.get('original_query'),
                'rewritten_query': event.data.get('rewritten_query'),
                'query_type': event.data.get('query_type'),
                'key_concepts': event.data.get('key_concepts', []),
                'search_keywords': event.data.get('search_keywords', []),
                'sub_questions': event.data.get('sub_questions', []),
                'is_complex': event.data.get('is_complex', False),
                'was_rewritten': event.data.get('was_rewritten', False),
            }
        })}\n\n"
    
    elif event.type == EventType.QUERY_UNDERSTANDING_SKIPPED:
        return f"event: query_understanding_skipped\ndata: {json.dumps({
            'type': 'query_understanding_skipped',
            'timestamp': event.timestamp.isoformat(),
            'data': {'reason': 'Feature disabled'}
        })}\n\n"
    
    elif event.type == EventType.RETRIEVAL_STARTED:
        return f"event: retrieval_started\ndata: {json.dumps({
            'type': 'retrieval_started',
            'timestamp': event.timestamp.isoformat(),
            'data': {
                'strategy': event.data.get('strategy'),
                'num_sub_questions': event.data.get('num_sub_questions', 0),
            }
        })}\n\n"
    
    # ... handle other event types ...
```

---

## Phase 8: Configuration

### 8.1 Update Config: `backend/app/rag_agent/config.py`

> Repo note: `backend/app/rag_agent/config.py` currently contains Azure OpenAI LLM wiring (`get_analyst_llm()`, `get_orchestrator_llm()`), not feature flags. You can add the flag helpers here, but it may be cleaner to create a small separate module (e.g. `backend/app/rag_agent/runtime_config.py`) to avoid mixing concerns.

```python
"""RAG Agent configuration management."""
import os
from functools import lru_cache
from typing import TypedDict


class QueryRewriteConfig(TypedDict):
    """Configuration for query understanding/rewriting feature."""
    enabled: bool


class RetrievalConfig(TypedDict):
    """Configuration for document retrieval."""
    top_k: int
    enable_multi_query: bool
    top_k_per_query: int
    final_top_k: int


@lru_cache()
def get_query_rewrite_config() -> QueryRewriteConfig:
    """
    Get query rewriting configuration from environment.
    
    Environment variables:
        ENABLE_QUERY_REWRITING: Enable/disable query understanding (default: true)
    """
    return {
        "enabled": os.getenv("ENABLE_QUERY_REWRITING", "true").lower() == "true",
    }


@lru_cache()
def get_retrieval_config() -> RetrievalConfig:
    """
    Get retrieval configuration from environment.
    
    Environment variables:
        RETRIEVAL_TOP_K: Number of chunks to retrieve (default: 20)
        ENABLE_MULTI_QUERY_RETRIEVAL: Enable sub-question retrieval (default: true)
        MULTI_QUERY_TOP_K: Chunks per sub-question (default: 10)
        FINAL_TOP_K: Final chunks after RRF fusion (default: 20)
    """
    return {
        "top_k": int(os.getenv("RETRIEVAL_TOP_K", "20")),
        "enable_multi_query": os.getenv("ENABLE_MULTI_QUERY_RETRIEVAL", "true").lower() == "true",
        "top_k_per_query": int(os.getenv("MULTI_QUERY_TOP_K", "10")),
        "final_top_k": int(os.getenv("FINAL_TOP_K", "20")),
    }


def clear_config_cache():
    """Clear cached configuration (useful for testing)."""
    get_query_rewrite_config.cache_clear()
    get_retrieval_config.cache_clear()
```

### 8.2 Update `env.example`

```bash
# ==============================================================================
# Query Understanding Configuration
# ==============================================================================

# Enable/disable query rewriting and analysis (default: true)
# When disabled, original query passes through unchanged
ENABLE_QUERY_REWRITING=true

# ==============================================================================
# Retrieval Configuration  
# ==============================================================================

# Number of chunks to retrieve for simple queries (default: 20)
RETRIEVAL_TOP_K=20

# Enable multi-query retrieval for complex queries with sub-questions (default: true)
# When enabled, retrieves for main query + each sub-question and merges results
ENABLE_MULTI_QUERY_RETRIEVAL=true

# Chunks to retrieve per query in multi-query mode (default: 10)
MULTI_QUERY_TOP_K=10

# Final number of chunks after RRF fusion in multi-query mode (default: 20)
FINAL_TOP_K=20
```

---

## Phase 9: Frontend Updates (Optional)

### 9.1 TypeScript Types: `frontend/src/types/ragEvents.ts`

```typescript
// Add to your existing event types

export interface QueryUnderstoodEvent {
  type: 'query_understood';
  timestamp: string;
  data: {
    original_query: string;
    rewritten_query: string;
    query_type: 'factual' | 'comparison' | 'procedural' | 'exploratory' | 'definition';
    key_concepts: string[];
    search_keywords: string[];
    sub_questions: string[];
    is_complex: boolean;
    was_rewritten: boolean;
  };
}

export interface RetrievalStartedEvent {
  type: 'retrieval_started';
  timestamp: string;
  data: {
    strategy: 'single_query' | 'multi_query';
    num_sub_questions: number;
  };
}

// Update your union type
export type RAGEvent = 
  | PipelineStartedEvent
  | QueryUnderstoodEvent  // NEW
  | RetrievalStartedEvent // NEW
  | RetrievalCompletedEvent
  | AnalystSpawnedEvent
  | AnalystCompletedEvent
  | SynthesisStartedEvent
  | FinalResponseEvent
  | ErrorEvent;
```

### 9.2 State Reducer: `frontend/src/api/ragStream.ts`

```typescript
// Add to your state reducer

case 'query_understood':
  return {
    ...state,
    queryAnalysis: {
      originalQuery: event.data.original_query,
      rewrittenQuery: event.data.rewritten_query,
      queryType: event.data.query_type,
      keyConcepts: event.data.key_concepts,
      searchKeywords: event.data.search_keywords,
      subQuestions: event.data.sub_questions,
      isComplex: event.data.is_complex,
      wasRewritten: event.data.was_rewritten,
    },
    currentStep: 'retrieve',
  };

case 'retrieval_started':
  return {
    ...state,
    retrievalStrategy: event.data.strategy,
    numSubQuestions: event.data.num_sub_questions,
  };
```

---

## Implementation Checklist

### Step 1: State & Schema (20 min)
- [ ] Update `backend/app/rag_agent/state.py` with new fields
- [ ] Create `backend/app/rag_agent/schemas/query_analysis.py`
- [ ] Create `backend/app/rag_agent/prompts/query_prompts.py`

### Step 2: Query Understanding Node (30 min)
- [ ] Create `backend/app/rag_agent/nodes/query_processor.py`
- [ ] Test structured output with your LLM
- [ ] Verify graceful degradation on errors

### Step 3: Hybrid Search (45 min)
- [ ] Implement hybrid/multi-query retrieval using existing codepaths:
      - Preferred: extend `backend/app/rag_agent/tools/search.py` + `backend/app/rag_agent/nodes/retriever.py`
      - Alternative: add an agent-ready Azure SDK helper in `backend/app/services/azure_search.py` and adapt the agent to use it
- [ ] Update `backend/app/rag_agent/nodes/retriever.py` with multi-query support
- [ ] Test Azure AI Search hybrid queries
- [ ] Verify RRF fusion logic

### Step 4: Graph Wiring (15 min)
- [ ] Update `backend/app/rag_agent/graph.py`
- [ ] Update `get_initial_state()` helper
- [ ] Add retry policy to new node

### Step 5: Configuration (10 min)
- [ ] Update `backend/app/rag_agent/config.py`
- [ ] Update `env.example`
- [ ] Test feature flags

### Step 6: Callbacks & Streaming (25 min)
- [ ] Update `backend/app/rag_agent/callbacks.py` with new events
- [ ] Update SSE router handlers
- [ ] Test event streaming

### Step 7: Testing (45 min)
- [ ] Unit test `QueryAnalysis` schema validation
- [ ] Unit test `understand_query` node with mocked LLM
- [ ] Unit test retrieval logic (mock LangChain retriever OR mock Azure SDK helper, depending on chosen Phase 5 approach)
- [ ] Unit test RRF fusion logic
- [ ] Integration test full pipeline
- [ ] Test with `ENABLE_QUERY_REWRITING=false`
- [ ] Test with `ENABLE_MULTI_QUERY_RETRIEVAL=false`
- [ ] Verify SSE events in browser

### Step 8: Frontend (Optional, 30 min)
- [ ] Add TypeScript types
- [ ] Update state reducer
- [ ] Add UI for query analysis display

---

## Rollback Strategy

All features are fully bypassable via environment variables:

```bash
# Disable query rewriting entirely
ENABLE_QUERY_REWRITING=false

# Disable multi-query retrieval (still uses hybrid search)
ENABLE_MULTI_QUERY_RETRIEVAL=false
```

When `ENABLE_QUERY_REWRITING=false`:
- `understand_query` node passes original query through unchanged
- No LLM call is made
- `search_keywords` and `sub_questions` remain empty
- Retrieval uses original query with standard hybrid search

---

## Performance Considerations

| Component | Estimated Latency | Token Usage |
|-----------|-------------------|-------------|
| Query Understanding LLM Call | 300-800ms | ~300-500 tokens |
| Multi-Query Retrieval (3 queries) | +200-400ms | N/A |
| RRF Fusion | <10ms | N/A |

**Total added latency:** ~500-1200ms for complex queries

### Optimization Opportunities (Future)

1. **Query Caching**: Cache query analysis results by query hash (5-minute TTL)
2. **Smaller Model**: Use GPT-4o-mini or Claude Haiku for query understanding
3. **Parallel Execution**: Run query understanding in parallel with initial retrieval (speculative)
4. **Adaptive Complexity**: Only decompose queries when confidence is high

---

## Success Metrics

Track these metrics to validate the feature:

| Metric | How to Measure | Target |
|--------|----------------|--------|
| Query Rewrite Rate | % queries where `rewritten_query != original_query` | 60-80% |
| Keyword Extraction Rate | Avg keywords per query | 4-6 |
| Complex Query Rate | % queries with `is_complex=true` | 10-20% |
| Retrieval Relevance | Manual evaluation of top-5 results | +15-25% improvement |
| Latency Impact | P50/P95 of `understand_query` node | <500ms / <800ms |
| Error Rate | % queries falling back to original | <2% |

---

## Testing Queries

Use these queries to test the implementation:

### Simple Factual
```
"What is the maximum file size for S3 uploads?"
Expected: No sub-questions, keywords include "S3", "max file size"
```

### Complex Comparison
```
"Compare Lambda cold starts vs ECS startup time for real-time APIs"
Expected: 3 sub-questions, keywords include "Lambda", "cold start", "ECS"
```

### Error Troubleshooting
```
"Fix ECONNREFUSED error in Node.js Redis connection"
Expected: No sub-questions, keywords include "ECONNREFUSED", "Node.js", "Redis"
```

### Exploratory
```
"Tell me about best practices for securing AWS infrastructure"
Expected: Possibly 2-3 sub-questions, broad keywords
```

### Definition
```
"What is infrastructure as code and how does Terraform work?"
Expected: 2 sub-questions, keywords include "IaC", "Terraform"
```
