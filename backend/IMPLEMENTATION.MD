# Multi-Agent RAG System with LangGraph: Complete Implementation Plan

A comprehensive blueprint for building a production-ready document analysis system where a central orchestrator dynamically spawns analyst agents to process retrieved chunks in parallel, collecting findings with full source attribution into a collaborative document before final synthesis with citations.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Step 1: Project Setup with uv](#step-1-project-setup-with-uv)
4. [Step 2: Environment Configuration](#step-2-environment-configuration)
5. [Step 3: State Definitions with Reducers](#step-3-state-definitions-with-reducers)
6. [Step 4: Pydantic Schemas with Source Attribution](#step-4-pydantic-schemas-with-source-attribution)
7. [Step 5: Azure AI Search Integration](#step-5-azure-ai-search-integration)
8. [Step 6: Azure OpenAI Configuration](#step-6-azure-openai-configuration)
9. [Step 7: Analyst Prompts](#step-7-analyst-prompts)
10. [Step 8: Node Implementations](#step-8-node-implementations)
11. [Step 9: Main Graph Definition](#step-9-main-graph-definition)
12. [Step 10: LangGraph Server Configuration](#step-10-langgraph-server-configuration)
13. [Step 11: Docker Deployment](#step-11-docker-deployment)
14. [Step 12: Build and Deploy Commands](#step-12-build-and-deploy-commands)
15. [Implementation Checklist](#implementation-checklist)
16. [Example Output](#example-output)
17. [Azure AI Search Index Schema](#azure-ai-search-index-schema)

---

## Architecture Overview

The system follows a **Query → Retrieve → Distribute → Analyze → Synthesize** flow using LangGraph's `Send()` API for dynamic parallelism. An orchestrator node retrieves document chunks from Azure AI Search, then uses conditional edges to spawn variable numbers of lightweight analyst agents based on chunk count. Each analyst evaluates relevance, extracts insights with full source metadata, and writes to a shared state using reducer functions. A larger model synthesizes all findings into a final response with properly formatted citations including page numbers, authors, and dates.

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Azure OpenAI GPT-4o)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  START   │───▶│  RETRIEVE   │───▶│  DISTRIBUTE via  │───▶│ SYNTHESIZE│  │
│  │          │    │  (Azure AI  │    │    Send() API    │    │ (Generate │  │
│  │          │    │   Search)   │    │                  │    │ Citations)│  │
│  └──────────┘    └─────────────┘    └────────┬─────────┘    └───────────┘  │
│                                              │                              │
│                    ┌─────────────────────────┼─────────────────────────┐    │
│                    │                         │                         │    │
│                    ▼                         ▼                         ▼    │
│            ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│            │  ANALYST 1  │           │  ANALYST 2  │    ...    │  ANALYST N  │
│            │ (GPT-4o-mini│           │ (GPT-4o-mini│           │ (GPT-4o-mini│
│            │  Azure)     │           │  Azure)     │           │  Azure)     │
│            └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
│                   │                         │                         │
│                   └─────────────────────────┴─────────────────────────┘
│                                             │
│                                             ▼
│                            ┌─────────────────────────────────┐
│                            │       SHARED STATE              │
│                            │  - analyses: Annotated[list,    │
│                            │              operator.add]      │
│                            │  - shared_document: Annotated   │
│                            │    [str, concat_reducer]        │
│                            │  - Full source metadata with    │
│                            │    page numbers, authors, dates │
│                            └─────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘

Infrastructure:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  LangGraph API  │───▶│     Redis       │    │   PostgreSQL    │
│   (Docker)      │    │  (Task Queue)   │    │ (Checkpointing) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Why Redis and PostgreSQL?

**Redis** handles:
- Task queue and pub/sub for the LangGraph Server
- Background run coordination
- Streaming state updates
- Real-time communication between API server and running graph executions

**PostgreSQL** handles:
- **Checkpoints**: Intermediate graph state snapshots enabling resumability, time-travel debugging, and human-in-the-loop interrupts
- **Thread state**: Conversation history for multi-turn interactions
- **Run metadata**: Logs of completed runs for observability

---

## Project Structure

```
multi-agent-rag/
├── src/
│   └── rag_agent/
│       ├── __init__.py
│       ├── graph.py              # Main graph definition & compilation
│       ├── state.py              # TypedDict state schemas with reducers
│       ├── config.py             # Azure OpenAI model configuration
│       ├── nodes/
│       │   ├── __init__.py
│       │   ├── retriever.py      # Azure AI Search retrieval node
│       │   ├── distributor.py    # Send() routing logic
│       │   ├── analyst.py        # Document analyst agent
│       │   └── synthesizer.py    # Final synthesis with citations
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── outputs.py        # Pydantic schemas with source metadata
│       ├── prompts/
│       │   ├── __init__.py
│       │   └── analyst_prompts.py
│       └── tools/
│           ├── __init__.py
│           └── search.py         # Azure AI Search retriever wrapper
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_analyst.py
│   │   └── test_search.py
│   └── integration/
│       ├── __init__.py
│       └── test_full_flow.py
├── langgraph.json
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Step 1: Project Setup with uv

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "multi-agent-rag"
version = "0.1.0"
description = "Multi-agent RAG system with LangGraph, Azure OpenAI, and Azure AI Search"
requires-python = ">=3.11"

dependencies = [
    "langgraph>=0.6.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-community>=0.3.0",
    "langchain-core>=0.3.0",
    "langgraph-sdk>=0.1.66",
    "langgraph-checkpoint>=2.0.23",
    "langgraph-checkpoint-postgres>=2.0.0",
    "azure-search-documents>=11.4",
    "azure-identity>=1.15.0",
    "python-dotenv>=1.0.0",
    "pydantic>=2.0.0",
    "psycopg[binary]>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "langgraph-cli[inmem]>=0.4.0",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/rag_agent"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Initialize with uv

```bash
# Create project directory
mkdir multi-agent-rag && cd multi-agent-rag

# Initialize with uv
uv init

# Install Python 3.11
uv python install 3.11

# Add core dependencies
uv add langgraph langchain-openai langchain-community langchain-core
uv add langgraph-sdk langgraph-checkpoint langgraph-checkpoint-postgres
uv add azure-search-documents azure-identity python-dotenv pydantic
uv add "psycopg[binary]"

# Add dev dependencies
uv add --group dev "langgraph-cli[inmem]" pytest pytest-asyncio ruff

# Sync all dependencies
uv sync --group dev

# Create directory structure
mkdir -p src/rag_agent/{nodes,schemas,prompts,tools}
mkdir -p tests/{unit,integration}
touch src/rag_agent/__init__.py
touch src/rag_agent/nodes/__init__.py
touch src/rag_agent/schemas/__init__.py
touch src/rag_agent/prompts/__init__.py
touch src/rag_agent/tools/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

---

## Step 2: Environment Configuration

### .env.example

```bash
# ============================================
# Azure OpenAI Configuration
# ============================================
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# Deployment names (must match your Azure OpenAI deployments)
AZURE_OPENAI_ORCHESTRATOR_DEPLOYMENT=gpt-4o
AZURE_OPENAI_ANALYST_DEPLOYMENT=gpt-4o-mini

# ============================================
# Azure AI Search Configuration
# ============================================
AZURE_AI_SEARCH_SERVICE_NAME=your-search-service
AZURE_AI_SEARCH_INDEX_NAME=docling-chunks-index
AZURE_AI_SEARCH_API_KEY=your-query-key

# ============================================
# LangGraph Server (Docker deployment)
# ============================================
REDIS_URI=redis://langgraph-redis:6379
DATABASE_URI=postgres://langgraph:langgraph@langgraph-postgres:5432/langgraph?sslmode=disable

# ============================================
# LangSmith (Optional - for tracing)
# ============================================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=multi-agent-rag

# ============================================
# LangGraph Server Auth
# ============================================
LANGSMITH_API_KEY=lsv2_...
```

### Azure OpenAI Deployment Prerequisites

Before running the system, ensure you have these deployments created in your Azure OpenAI resource:

| Deployment Name | Model | Purpose | Recommended TPM |
|-----------------|-------|---------|-----------------|
| `gpt-4o-mini` | gpt-4o-mini | Analyst agents (parallel) | 150K+ (scales with parallelism) |
| `gpt-4o` | gpt-4o | Orchestrator synthesis | 30K |

**Note on rate limits**: Since analyst agents run in parallel, you may need higher TPM (tokens per minute) limits on the `gpt-4o-mini` deployment. For 20 parallel analysts with ~2K tokens each, plan for at least 40K TPM burst capacity. Consider enabling dynamic quota if available in your region.

---

## Step 3: State Definitions with Reducers

### src/rag_agent/state.py

```python
"""State schemas with reducer functions for parallel agent accumulation."""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from operator import add
from langchain_core.documents import Document

# Import will be available after creating schemas
# from .schemas.outputs import ChunkAnalysis


def concat_markdown(left: str, right: str) -> str:
    """Reducer for accumulating markdown content from multiple agents."""
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n---\n\n{right}"


class OrchestratorState(TypedDict):
    """Main orchestrator state - shared across entire graph."""
    # Input
    query: str
    filters: Optional[Dict[str, Any]]  # Optional metadata filters for retrieval
    
    # Retrieved chunks from Azure AI Search
    retrieved_chunks: List[Document]
    
    # Accumulated from parallel analyst agents via reducer
    # Type hint uses Any to avoid circular import; actual type is List[ChunkAnalysis]
    analyses: Annotated[List[Any], add]
    
    # Shared collaborative document - all agents write here
    shared_document: Annotated[str, concat_markdown]
    
    # Final output
    final_response: str
    
    # Metadata
    retrieval_count: int
    relevant_count: int
    sources_used: List[str]  # Document titles used in response


class AnalystState(TypedDict):
    """State passed to individual analyst agents via Send()."""
    # Input for this analyst
    query: str
    chunk: Document
    chunk_id: str
    chunk_metadata: Dict[str, Any]  # Extracted metadata for this chunk
    
    # Output - written back to orchestrator via reducer
    analyses: Annotated[List[Any], add]
    shared_document: Annotated[str, concat_markdown]
```

---

## Step 4: Pydantic Schemas with Source Attribution

### src/rag_agent/schemas/outputs.py

```python
"""Structured output schemas with comprehensive source attribution."""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class SourceMetadata(BaseModel):
    """Metadata about the source document and chunk location."""
    
    # Document identification
    document_id: str = Field(description="Unique identifier for the source document")
    document_title: str = Field(description="Title or filename of the source document")
    
    # Location within document
    page_numbers: List[int] = Field(
        default_factory=list,
        description="Page number(s) where this chunk appears"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section or chapter title containing this chunk"
    )
    hierarchy_path: Optional[str] = Field(
        default=None,
        description="Full hierarchy path e.g. 'Chapter 1 > Section 1.2 > Subsection A'"
    )
    
    # Authorship
    author: Optional[str] = Field(default=None, description="Document author(s)")
    
    # Timestamps
    date_created: Optional[datetime] = Field(
        default=None,
        description="Document creation date"
    )
    date_modified: Optional[datetime] = Field(
        default=None,
        description="Document last modified date"
    )
    date_indexed: Optional[datetime] = Field(
        default=None,
        description="When the document was indexed"
    )
    
    # Additional metadata
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document: report, policy, manual, etc."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags or categories assigned to the document"
    )


class ChunkAnalysis(BaseModel):
    """Analysis result from a single document analyst agent."""
    
    chunk_id: str = Field(description="Unique identifier for the analyzed chunk")
    
    # Source attribution - REQUIRED
    source: SourceMetadata = Field(description="Full source metadata for citation")
    
    relevance_score: int = Field(
        description="Relevance score from 0-10 where 0=not relevant, 10=perfectly relevant",
        ge=0, le=10
    )
    
    is_relevant: bool = Field(
        description="True if relevance_score >= 5, indicating useful information"
    )
    
    summary: str = Field(
        description="Concise 2-3 sentence summary of relevant information, or 'Not relevant' if score < 5"
    )
    
    key_points: List[str] = Field(
        default_factory=list,
        description="List of 1-5 key facts, insights, or data points extracted"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the analysis accuracy"
    )
    
    source_quotes: List[str] = Field(
        default_factory=list,
        description="Verbatim quotes that directly answer the query (max 3)"
    )


class Citation(BaseModel):
    """Formatted citation for final response."""
    
    citation_id: int = Field(description="Citation number for reference [1], [2], etc.")
    document_title: str
    author: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    section_title: Optional[str] = None
    date_modified: Optional[datetime] = None
    chunk_ids: List[str] = Field(
        description="Chunk IDs that contributed from this source"
    )
    
    def format_citation(self) -> str:
        """Format as readable citation string."""
        parts = [f"[{self.citation_id}]"]
        
        if self.author:
            parts.append(f"{self.author}.")
        
        parts.append(f'"{self.document_title}"')
        
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                parts.append(f"p. {self.page_numbers[0]}")
            else:
                pages = sorted(self.page_numbers)
                parts.append(f"pp. {pages[0]}-{pages[-1]}")
        
        if self.section_title:
            parts.append(f"§ {self.section_title}")
        
        if self.date_modified:
            parts.append(f"(Modified: {self.date_modified.strftime('%Y-%m-%d')})")
        
        return " ".join(parts)


class SynthesizedResponse(BaseModel):
    """Final synthesized response with full source attribution."""
    
    answer: str = Field(
        description="Comprehensive answer with inline citation markers [1], [2], etc."
    )
    
    supporting_evidence: List[str] = Field(
        description="Key evidence points with citation markers"
    )
    
    citations: List[Citation] = Field(
        default_factory=list,
        description="Full citation list for all referenced sources"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence based on relevance and coverage of sources"
    )
    
    coverage_summary: str = Field(
        description="Brief assessment of how well the sources covered the query"
    )
    
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Aspects of the query not fully addressed by available sources"
    )
```

---

## Step 5: Azure AI Search Integration

### src/rag_agent/tools/search.py

```python
"""Azure AI Search retriever with comprehensive metadata extraction."""
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.documents import Document

from ..schemas.outputs import SourceMetadata


# Mapping of Azure AI Search index fields to our metadata schema
# Update these to match your actual index field names from Docling
METADATA_FIELD_MAPPING = {
    # Document identification
    "document_id": "document_id",           # or "id", "doc_id"
    "document_title": "title",              # or "filename", "document_name"
    
    # Location fields (from Docling)
    "page_numbers": "page_numbers",         # Array field
    "page_number": "page_number",           # Single page (alternative)
    "section_title": "section_title",       # or "heading", "section_name"
    "hierarchy_path": "hierarchy_path",     # or "breadcrumb"
    "hierarchy_level": "hierarchy_level",
    
    # Authorship
    "author": "author",                     # or "authors", "created_by"
    
    # Timestamps
    "date_created": "created_date",         # or "creation_date", "date_created"
    "date_modified": "modified_date",       # or "last_modified", "date_modified"
    "date_indexed": "indexed_at",           # or "index_date"
    
    # Additional metadata
    "document_type": "document_type",       # or "doc_type", "category"
    "tags": "tags",                         # or "keywords", "labels"
    
    # Docling-specific fields
    "chunk_type": "chunk_type",             # heading, paragraph, table, etc.
    "parent_id": "parent_id",
}


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse datetime from various formats."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def extract_source_metadata(doc: Document, chunk_id: str) -> SourceMetadata:
    """
    Extract SourceMetadata from a LangChain Document's metadata.
    
    Handles various field naming conventions and missing fields gracefully.
    """
    meta = doc.metadata or {}
    
    # Helper to get field with fallbacks
    def get_field(primary: str, *fallbacks, default=None):
        mapped = METADATA_FIELD_MAPPING.get(primary, primary)
        if mapped in meta and meta[mapped] is not None:
            return meta[mapped]
        for fallback in fallbacks:
            if fallback in meta and meta[fallback] is not None:
                return meta[fallback]
        return default
    
    # Extract page numbers - handle both array and single value
    page_numbers = []
    pages_value = get_field("page_numbers", "page_number", "pages")
    if isinstance(pages_value, list):
        page_numbers = [int(p) for p in pages_value if p is not None]
    elif pages_value is not None:
        page_numbers = [int(pages_value)]
    
    # Extract tags - handle string and array
    tags_value = get_field("tags", "keywords", "labels", default=[])
    if isinstance(tags_value, str):
        tags = [t.strip() for t in tags_value.split(",")]
    elif isinstance(tags_value, list):
        tags = tags_value
    else:
        tags = []
    
    return SourceMetadata(
        document_id=get_field("document_id", "id", "doc_id", default=chunk_id),
        document_title=get_field("document_title", "title", "filename", "source", default="Unknown Document"),
        page_numbers=page_numbers,
        section_title=get_field("section_title", "heading", "section_name"),
        hierarchy_path=get_field("hierarchy_path", "breadcrumb"),
        author=get_field("author", "authors", "created_by"),
        date_created=parse_datetime(get_field("date_created", "created_date", "creation_date")),
        date_modified=parse_datetime(get_field("date_modified", "modified_date", "last_modified")),
        date_indexed=parse_datetime(get_field("date_indexed", "indexed_at")),
        document_type=get_field("document_type", "doc_type", "category"),
        tags=tags
    )


def construct_odata_filter(filters: Optional[Dict]) -> Optional[str]:
    """
    Build OData filter expression for Azure AI Search.
    
    Supports Docling hierarchical metadata fields and date range filtering.
    
    Examples:
        {"document_type": "report"} -> "document_type eq 'report'"
        {"hierarchy_level": {"op": "le", "value": 2}} -> "hierarchy_level le 2"
        {"tags": ["finance", "q3"]} -> "search.in(tags, 'finance,q3', ',')"
    """
    if not filters:
        return None
    
    clauses = []
    
    for field, value in filters.items():
        mapped_field = METADATA_FIELD_MAPPING.get(field, field)
        
        if isinstance(value, list):
            # Multiple values: use search.in()
            values_str = ",".join(f"'{v}'" if isinstance(v, str) else str(v) for v in value)
            clauses.append(f"search.in({mapped_field}, '{values_str}', ',')")
        elif isinstance(value, dict):
            # Range filter: {"op": "ge", "value": "2024-01-01"}
            op = value.get("op", "eq")
            val = value["value"]
            if isinstance(val, str):
                clauses.append(f"{mapped_field} {op} '{val}'")
            else:
                clauses.append(f"{mapped_field} {op} {val}")
        elif isinstance(value, str):
            clauses.append(f"{mapped_field} eq '{value}'")
        elif isinstance(value, bool):
            clauses.append(f"{mapped_field} eq {str(value).lower()}")
        else:
            clauses.append(f"{mapped_field} eq {value}")
    
    return " and ".join(clauses) if clauses else None


def get_retriever(
    top_k: int = 20,
    filters: Optional[Dict] = None
) -> AzureAISearchRetriever:
    """
    Create configured Azure AI Search retriever.
    
    Ensure your index includes these retrievable fields:
    - content (searchable text)
    - All metadata fields listed in METADATA_FIELD_MAPPING
    
    Environment variables required:
    - AZURE_AI_SEARCH_SERVICE_NAME
    - AZURE_AI_SEARCH_INDEX_NAME  
    - AZURE_AI_SEARCH_API_KEY (Query key recommended for reads)
    """
    filter_expr = construct_odata_filter(filters)
    
    return AzureAISearchRetriever(
        service_name=os.getenv("AZURE_AI_SEARCH_SERVICE_NAME"),
        index_name=os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
        api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
        content_key="content",  # Field containing document text
        top_k=top_k,
        filter=filter_expr
    )


async def retrieve_chunks(
    query: str,
    top_k: int = 20,
    filters: Optional[Dict] = None
) -> List[Document]:
    """Async retrieval with metadata extraction."""
    retriever = get_retriever(top_k=top_k, filters=filters)
    
    try:
        docs = await retriever.ainvoke(query)
        
        # Ensure chunk_id in metadata
        for i, doc in enumerate(docs):
            if "chunk_id" not in doc.metadata:
                doc.metadata["chunk_id"] = doc.metadata.get("id", f"chunk_{i}")
        
        return docs
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
```

---

## Step 6: Azure OpenAI Configuration

### src/rag_agent/config.py

```python
"""Configuration for Azure OpenAI models."""
import os
from langchain_openai import AzureChatOpenAI


def get_analyst_llm() -> AzureChatOpenAI:
    """
    Get configured Azure OpenAI LLM for analyst agents (smaller model).
    
    Uses GPT-4o-mini deployment for cost-effective parallel processing.
    Optimized for fast, focused chunk analysis.
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_ANALYST_DEPLOYMENT", "gpt-4o-mini"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0.1,  # Low temperature for consistent analysis
        max_retries=2,
    )


def get_orchestrator_llm() -> AzureChatOpenAI:
    """
    Get configured Azure OpenAI LLM for orchestrator (larger model).
    
    Uses GPT-4o deployment for high-quality synthesis and citation generation.
    """
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_ORCHESTRATOR_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        temperature=0.2,  # Slightly higher for synthesis creativity
        max_retries=2,
    )
```

---

## Step 7: Analyst Prompts

### src/rag_agent/prompts/analyst_prompts.py

```python
"""Prompts optimized for Azure OpenAI with source attribution."""

DOCUMENT_ANALYST_SYSTEM_PROMPT = """You are a document analyst agent. Your job is to evaluate text chunks for relevance and extract key information while preserving source attribution.

RESPONSIBILITIES:
1. Assess relevance to the user's query (score 0-10)
2. Extract key facts, numbers, and insights
3. Summarize relevant information concisely
4. Identify direct quotes that answer the query

GUIDELINES:
- Be precise and concise
- Only use information present in the chunk - never hallucinate
- If not relevant (score < 5), state "Not relevant" clearly
- Extract verbatim quotes when they directly answer the query
- Note specific facts, figures, dates, and named entities

SCORING CRITERIA:
- 0-2: Not relevant - no information related to query
- 3-4: Marginally relevant - tangentially related
- 5-6: Somewhat relevant - contains useful background
- 7-8: Highly relevant - directly addresses query
- 9-10: Perfectly relevant - comprehensively answers query"""


ANALYST_USER_PROMPT_TEMPLATE = """Analyze this document chunk for the given query.

QUERY: {query}

SOURCE INFORMATION:
- Document: {document_title}
- Page(s): {page_numbers}
- Section: {section_title}
- Author: {author}
- Chunk ID: {chunk_id}

CHUNK CONTENT:
{chunk_content}

Provide your analysis:
1. relevance_score (0-10)
2. is_relevant (true if score >= 5)
3. summary (2-3 sentences if relevant, "Not relevant" otherwise)
4. key_points (1-5 key facts, empty list if not relevant)
5. confidence (high/medium/low)
6. source_quotes (max 3 verbatim quotes if highly relevant)"""


SHARED_DOCUMENT_CONTRIBUTION_TEMPLATE = """### Chunk {chunk_id}
**Source:** {document_title} | **Page(s):** {page_numbers} | **Section:** {section_title}
**Author:** {author}
**Relevance:** {relevance_score}/10 | **Confidence:** {confidence}

{summary}

**Key Points:**
{key_points_formatted}
"""


SYNTHESIS_SYSTEM_PROMPT = """You are a research synthesizer. Your job is to combine analyses from multiple document analysts into a comprehensive, well-cited answer.

CITATION REQUIREMENTS:
1. Use inline citation markers [1], [2], etc. when referencing specific sources
2. Every factual claim must have a citation
3. Group related information from the same source
4. Include page numbers in citations when available

GUIDELINES:
1. Prioritize information from high-relevance chunks (score 7+)
2. Identify patterns and connections across sources
3. Note any contradictions between sources
4. Be honest about uncertainty and limitations
5. Structure the answer clearly with the most important information first"""


SYNTHESIS_USER_PROMPT_TEMPLATE = """Synthesize these document analyses into a comprehensive, well-cited answer.

ORIGINAL QUERY: {query}

ANALYST FINDINGS:
{analyses_formatted}

SOURCE LIST FOR CITATIONS:
{source_list}

Requirements:
1. Provide a clear answer with inline citations [1], [2], etc.
2. List supporting evidence with citations
3. Assess overall confidence
4. Note any gaps in the available information

Use the source numbers provided in the SOURCE LIST for your citations."""
```

---

## Step 8: Node Implementations

### src/rag_agent/nodes/retriever.py

```python
"""Retrieval node for Azure AI Search."""
from typing import Dict, Any
from ..state import OrchestratorState
from ..tools.search import retrieve_chunks


async def retrieve_documents(state: OrchestratorState) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from Azure AI Search.
    
    Supports optional metadata filters passed in state.
    """
    query = state["query"]
    filters = state.get("filters")
    
    # Default filter: get top-level chunks (hierarchy_level <= 2)
    default_filters = {"hierarchy_level": {"op": "le", "value": 2}}
    if filters:
        default_filters.update(filters)
    
    chunks = await retrieve_chunks(
        query=query,
        top_k=20,  # Retrieve more, filter after analysis
        filters=default_filters
    )
    
    return {
        "retrieved_chunks": chunks,
        "retrieval_count": len(chunks)
    }
```

### src/rag_agent/nodes/distributor.py

```python
"""Distribution logic using Send() API for dynamic parallelism."""
from typing import List
from langgraph.types import Send
from ..state import OrchestratorState
from ..tools.search import extract_source_metadata


def distribute_to_analysts(state: OrchestratorState) -> List[Send]:
    """
    Distribute retrieved chunks to parallel analyst agents.
    
    Uses LangGraph's Send() API to dynamically spawn one analyst
    per chunk. The number of analysts scales with retrieval results.
    
    Extracts and passes metadata with each chunk for source attribution.
    """
    sends = []
    
    for i, chunk in enumerate(state["retrieved_chunks"]):
        chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")
        
        # Pre-extract metadata to pass to analyst
        source_metadata = extract_source_metadata(chunk, chunk_id)
        
        sends.append(
            Send("analyze_chunk", {
                "query": state["query"],
                "chunk": chunk,
                "chunk_id": chunk_id,
                "chunk_metadata": source_metadata.model_dump(),  # Serializable dict
                "analyses": [],
                "shared_document": ""
            })
        )
    
    return sends
```

### src/rag_agent/nodes/analyst.py

```python
"""Document analyst agent with comprehensive source attribution."""
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from ..state import AnalystState
from ..schemas.outputs import ChunkAnalysis, SourceMetadata
from ..config import get_analyst_llm
from ..prompts.analyst_prompts import (
    DOCUMENT_ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_PROMPT_TEMPLATE,
    SHARED_DOCUMENT_CONTRIBUTION_TEMPLATE
)


class ChunkAnalysisLLM(BaseModel):
    """Simplified schema for LLM - source metadata injected after."""
    relevance_score: int = Field(ge=0, le=10)
    is_relevant: bool
    summary: str
    key_points: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    source_quotes: List[str] = Field(default_factory=list)


async def analyze_chunk(state: AnalystState) -> Dict[str, Any]:
    """
    Individual analyst agent processes a single chunk.
    
    Uses Azure OpenAI GPT-4o-mini with structured output for reliable responses.
    Preserves full source metadata for citation in final response.
    """
    llm = get_analyst_llm()
    
    # Reconstruct SourceMetadata from passed dict
    source_metadata = SourceMetadata(**state["chunk_metadata"])
    
    # Format page numbers for display
    page_display = ", ".join(str(p) for p in source_metadata.page_numbers) if source_metadata.page_numbers else "Unknown"
    
    # Format the user prompt with source context
    user_prompt = ANALYST_USER_PROMPT_TEMPLATE.format(
        query=state["query"],
        chunk_id=state["chunk_id"],
        document_title=source_metadata.document_title,
        page_numbers=page_display,
        section_title=source_metadata.section_title or "N/A",
        author=source_metadata.author or "Unknown",
        chunk_content=state["chunk"].page_content
    )
    
    structured_llm = llm.with_structured_output(ChunkAnalysisLLM)
    
    try:
        llm_analysis = await structured_llm.ainvoke([
            {"role": "system", "content": DOCUMENT_ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        
        # Combine LLM analysis with source metadata
        analysis = ChunkAnalysis(
            chunk_id=state["chunk_id"],
            source=source_metadata,
            relevance_score=llm_analysis.relevance_score,
            is_relevant=llm_analysis.is_relevant,
            summary=llm_analysis.summary,
            key_points=llm_analysis.key_points,
            confidence=llm_analysis.confidence,
            source_quotes=llm_analysis.source_quotes
        )
        
    except Exception as e:
        analysis = ChunkAnalysis(
            chunk_id=state["chunk_id"],
            source=source_metadata,
            relevance_score=0,
            is_relevant=False,
            summary=f"Analysis failed: {str(e)}",
            key_points=[],
            confidence="low",
            source_quotes=[]
        )
    
    # Format contribution to shared document with source info
    key_points_formatted = "\n".join(f"- {kp}" for kp in analysis.key_points) or "- None"
    page_str = ", ".join(str(p) for p in source_metadata.page_numbers) if source_metadata.page_numbers else "N/A"
    
    document_contribution = ""
    if analysis.is_relevant:
        document_contribution = SHARED_DOCUMENT_CONTRIBUTION_TEMPLATE.format(
            chunk_id=analysis.chunk_id,
            document_title=source_metadata.document_title,
            page_numbers=page_str,
            author=source_metadata.author or "Unknown",
            section_title=source_metadata.section_title or "N/A",
            relevance_score=analysis.relevance_score,
            confidence=analysis.confidence,
            summary=analysis.summary,
            key_points_formatted=key_points_formatted
        )
    
    return {
        "analyses": [analysis],
        "shared_document": document_contribution
    }
```

### src/rag_agent/nodes/synthesizer.py

```python
"""Synthesis node with comprehensive citation generation."""
from typing import Dict, Any, List
from collections import defaultdict
from pydantic import BaseModel, Field

from ..state import OrchestratorState
from ..schemas.outputs import SynthesizedResponse, Citation, ChunkAnalysis
from ..config import get_orchestrator_llm
from ..prompts.analyst_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT_TEMPLATE
)


class SynthesisLLMResponse(BaseModel):
    """Schema for LLM synthesis - citations built separately."""
    answer: str = Field(description="Answer with inline citations [1], [2], etc.")
    supporting_evidence: List[str] = Field(description="Evidence points with citations")
    confidence: str = Field(description="high, medium, or low")
    coverage_summary: str
    gaps_identified: List[str] = Field(default_factory=list)


def build_citation_list(analyses: List[ChunkAnalysis]) -> tuple[List[Citation], Dict[str, int]]:
    """
    Build deduplicated citation list grouped by document.
    
    Returns:
        - List of Citation objects
        - Mapping of chunk_id to citation number
    """
    # Group chunks by document
    doc_chunks: Dict[str, List[ChunkAnalysis]] = defaultdict(list)
    for analysis in analyses:
        doc_key = analysis.source.document_id
        doc_chunks[doc_key].append(analysis)
    
    citations = []
    chunk_to_citation: Dict[str, int] = {}
    
    for citation_num, (doc_id, chunks) in enumerate(doc_chunks.items(), start=1):
        # Use first chunk's metadata as representative
        first = chunks[0].source
        
        # Collect all page numbers from this document's chunks
        all_pages = set()
        chunk_ids = []
        for chunk in chunks:
            all_pages.update(chunk.source.page_numbers)
            chunk_ids.append(chunk.chunk_id)
            chunk_to_citation[chunk.chunk_id] = citation_num
        
        citation = Citation(
            citation_id=citation_num,
            document_title=first.document_title,
            author=first.author,
            page_numbers=sorted(all_pages),
            section_title=first.section_title,
            date_modified=first.date_modified,
            chunk_ids=chunk_ids
        )
        citations.append(citation)
    
    return citations, chunk_to_citation


def format_analyses_with_citations(
    analyses: List[ChunkAnalysis],
    chunk_to_citation: Dict[str, int]
) -> str:
    """Format analyses with citation numbers for synthesis prompt."""
    formatted = []
    for analysis in analyses:
        cite_num = chunk_to_citation.get(analysis.chunk_id, "?")
        page_str = ", ".join(str(p) for p in analysis.source.page_numbers) if analysis.source.page_numbers else "N/A"
        
        quotes_str = "; ".join(f'"{q}"' for q in analysis.source_quotes[:2]) if analysis.source_quotes else "None"
        
        formatted.append(
            f"**[{cite_num}] {analysis.source.document_title}** "
            f"(p. {page_str}, Relevance: {analysis.relevance_score}/10)\n"
            f"Summary: {analysis.summary}\n"
            f"Key Points: {'; '.join(analysis.key_points)}\n"
            f"Quotes: {quotes_str}"
        )
    
    return "\n\n".join(formatted)


def format_source_list(citations: List[Citation]) -> str:
    """Format source list for the synthesis prompt."""
    lines = []
    for c in citations:
        line = f"[{c.citation_id}] {c.document_title}"
        if c.author:
            line += f" by {c.author}"
        if len(c.page_numbers) > 1:
            line += f" (pp. {c.page_numbers[0]}-{c.page_numbers[-1]})"
        elif c.page_numbers:
            line += f" (p. {c.page_numbers[0]})"
        lines.append(line)
    return "\n".join(lines)


async def synthesize_response(state: OrchestratorState) -> Dict[str, Any]:
    """
    Synthesize final response with full citations.
    
    Groups sources by document, generates citation list,
    and produces a well-cited comprehensive answer.
    """
    analyses = state["analyses"]
    query = state["query"]
    
    # Filter and sort by relevance
    relevant = [a for a in analyses if a.is_relevant]
    relevant_sorted = sorted(relevant, key=lambda x: x.relevance_score, reverse=True)
    
    # Early return if no relevant chunks
    if not relevant_sorted:
        return {
            "final_response": "I couldn't find relevant information to answer your query in the available documents.",
            "relevant_count": 0,
            "sources_used": []
        }
    
    # Build citation list from relevant analyses
    citations, chunk_to_citation = build_citation_list(relevant_sorted)
    
    # Format for synthesis
    top_analyses = relevant_sorted[:15]  # Top 15 most relevant
    analyses_formatted = format_analyses_with_citations(top_analyses, chunk_to_citation)
    source_list = format_source_list(citations)
    
    llm = get_orchestrator_llm()
    structured_llm = llm.with_structured_output(SynthesisLLMResponse)
    
    user_prompt = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        query=query,
        analyses_formatted=analyses_formatted,
        source_list=source_list
    )
    
    try:
        synthesis = await structured_llm.ainvoke([
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        
        # Format citations
        citations_formatted = "\n".join([c.format_citation() for c in citations])
        
        # Build final response with all sections
        final_response = f"""## Answer

{synthesis.answer}

---

### Supporting Evidence
{chr(10).join(f'• {e}' for e in synthesis.supporting_evidence)}

---

### Sources

{citations_formatted}

---

### Assessment
**Confidence:** {synthesis.confidence}

**Coverage:** {synthesis.coverage_summary}
"""
        if synthesis.gaps_identified:
            final_response += f"\n**Gaps identified:** {', '.join(synthesis.gaps_identified)}"
        
    except Exception as e:
        final_response = f"Synthesis error: {str(e)}\n\nPartial findings available in shared document."
        citations = []
    
    return {
        "final_response": final_response,
        "relevant_count": len(relevant_sorted),
        "sources_used": [c.document_title for c in citations]
    }
```

---

## Step 9: Main Graph Definition

### src/rag_agent/graph.py

```python
"""Main LangGraph definition with dynamic multi-agent orchestration."""
from langgraph.graph import StateGraph, START, END
from langgraph.pregel import RetryPolicy

from .state import OrchestratorState
from .nodes.retriever import retrieve_documents
from .nodes.distributor import distribute_to_analysts
from .nodes.analyst import analyze_chunk
from .nodes.synthesizer import synthesize_response


def build_rag_graph():
    """
    Build the multi-agent RAG graph.
    
    Flow: START → retrieve → [distribute via Send()] → analyze_chunk (parallel) → synthesize → END
    
    The distribute_to_analysts function returns a list of Send() objects,
    one per retrieved chunk. LangGraph executes all these in parallel within
    a single superstep, then collects results via the reducer functions
    defined in OrchestratorState.
    """
    builder = StateGraph(OrchestratorState)
    
    # Configure retry policy for analyst nodes (handles transient Azure OpenAI errors)
    analyst_retry = RetryPolicy(
        initial_interval=0.5,
        backoff_factor=2.0,
        max_interval=10.0,
        max_attempts=3,
        jitter=True
    )
    
    # Add nodes
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node(
        "analyze_chunk", 
        analyze_chunk,
        retry=analyst_retry  # Auto-retry on failures
    )
    builder.add_node("synthesize", synthesize_response)
    
    # Add edges
    builder.add_edge(START, "retrieve")
    
    # Conditional edge with Send() for dynamic parallelism
    # distribute_to_analysts returns List[Send] objects
    builder.add_conditional_edges(
        "retrieve",
        distribute_to_analysts,
        ["analyze_chunk"]  # Possible destinations
    )
    
    # All analysts fan-in to synthesize
    builder.add_edge("analyze_chunk", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()


# Compiled graph - this is what langgraph.json references
graph = build_rag_graph()


# Convenience function for direct invocation
async def run_rag_query(
    query: str,
    filters: dict = None
) -> dict:
    """Run a RAG query through the multi-agent system."""
    initial_state = {
        "query": query,
        "filters": filters,
        "retrieved_chunks": [],
        "analyses": [],
        "shared_document": "",
        "final_response": "",
        "retrieval_count": 0,
        "relevant_count": 0,
        "sources_used": []
    }
    
    result = await graph.ainvoke(initial_state)
    return result
```

### src/rag_agent/__init__.py

```python
"""Multi-agent RAG system with LangGraph and Azure OpenAI."""
from .graph import graph, run_rag_query

__all__ = ["graph", "run_rag_query"]
```

---

## Step 10: LangGraph Server Configuration

### langgraph.json

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "rag_agent": "./src/rag_agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.11",
  "pip_installer": "uv"
}
```

---

## Step 11: Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY langgraph.json .

# Install dependencies
RUN uv pip install --system -e .

# Install LangGraph CLI
RUN uv pip install --system "langgraph-cli[inmem]"

EXPOSE 8000

CMD ["langgraph", "up", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: "3.8"

volumes:
  langgraph-data:
    driver: local

services:
  langgraph-redis:
    image: redis:6
    healthcheck:
      test: redis-cli ping
      interval: 5s
      timeout: 1s
      retries: 5
    restart: unless-stopped

  langgraph-postgres:
    image: postgres:16
    ports:
      - "5433:5432"
    environment:
      POSTGRES_DB: langgraph
      POSTGRES_USER: langgraph
      POSTGRES_PASSWORD: langgraph
    volumes:
      - langgraph-data:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U langgraph
      start_period: 10s
      timeout: 1s
      retries: 5
      interval: 5s
    restart: unless-stopped

  langgraph-api:
    image: ${IMAGE_NAME:-multi-agent-rag:latest}
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8123:8000"
    depends_on:
      langgraph-redis:
        condition: service_healthy
      langgraph-postgres:
        condition: service_healthy
    env_file:
      - .env
    environment:
      REDIS_URI: redis://langgraph-redis:6379
      DATABASE_URI: postgres://langgraph:langgraph@langgraph-postgres:5432/langgraph?sslmode=disable
    restart: unless-stopped
```

---

## Step 12: Build and Deploy Commands

```bash
# ============================================
# Development: Run locally without Docker
# ============================================
cd multi-agent-rag
uv sync --group dev
uv run langgraph dev --port 2024

# ============================================
# Build Docker image
# ============================================
langgraph build -t multi-agent-rag:latest

# ============================================
# Run with Docker Compose (includes Redis + Postgres)
# ============================================
docker compose up -d

# Check health
curl http://localhost:8123/ok

# View logs
docker compose logs -f langgraph-api

# ============================================
# Run single query via API
# ============================================
curl -X POST http://localhost:8123/runs \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "rag_agent",
    "input": {
      "query": "What are the key findings about quarterly revenue?",
      "filters": {"document_type": "report"}
    }
  }'

# ============================================
# Run with streaming
# ============================================
curl -X POST http://localhost:8123/runs/stream \
  -H "Content-Type: application/json" \
  -d '{
    "assistant_id": "rag_agent",
    "input": {
      "query": "Summarize the main compliance requirements"
    },
    "stream_mode": "values"
  }'

# ============================================
# Stop services
# ============================================
docker compose down

# Remove all data (clean restart)
docker compose down -v
```

---

## Implementation Checklist

Execute these steps sequentially for a working implementation:

| Step | Task | File(s) | Validation |
|------|------|---------|------------|
| 1 | Initialize project with uv | `pyproject.toml` | `uv sync` succeeds |
| 2 | Create directory structure | All folders | Directories exist |
| 3 | Create `.env` from example | `.env` | Environment variables load |
| 4 | Implement state definitions | `state.py` | Import without errors |
| 5 | Implement Pydantic schemas | `schemas/outputs.py` | Instantiate models with test data |
| 6 | Implement Azure AI Search tools | `tools/search.py` | Test retrieval with sample query |
| 7 | Implement Azure OpenAI config | `config.py` | LLMs initialize without error |
| 8 | Implement prompts | `prompts/analyst_prompts.py` | Review prompt quality |
| 9 | Implement retriever node | `nodes/retriever.py` | Unit test retrieval node |
| 10 | Implement distributor node | `nodes/distributor.py` | Verify Send objects created |
| 11 | Implement analyst node | `nodes/analyst.py` | Test single chunk analysis |
| 12 | Implement synthesizer node | `nodes/synthesizer.py` | Test synthesis with mock analyses |
| 13 | Implement main graph | `graph.py` | `langgraph dev` starts successfully |
| 14 | Create `langgraph.json` | `langgraph.json` | CLI recognizes graph |
| 15 | Integration test full flow | `tests/integration/` | End-to-end query works |
| 16 | Create Dockerfile | `Dockerfile` | Image builds successfully |
| 17 | Create docker-compose.yml | `docker-compose.yml` | `docker compose up` succeeds |
| 18 | Production deployment | - | Health check passes at `/ok` |

---

## Example Output

When you run a query, the final response will look like this:

```markdown
## Answer

The quarterly revenue showed a 15% increase compared to the previous year [1], 
driven primarily by growth in the enterprise segment [1][2]. The CFO noted that 
"recurring revenue now represents 67% of total revenue" [1], indicating a 
successful transition to subscription-based models. International markets 
contributed 42% of total revenue [2], with EMEA showing the strongest growth 
at 23% year-over-year [3].

---

### Supporting Evidence
• Revenue increased 15% YoY to $4.2B [1]
• Enterprise segment grew 28% [1]
• Recurring revenue reached 67% of total [1]
• International revenue at 42% of total [2]
• EMEA growth rate of 23% [3]

---

### Sources

[1] Smith, J. "Q3 2024 Earnings Report" pp. 12-15 § Financial Highlights (Modified: 2024-10-15)
[2] "Annual Revenue Analysis 2024" pp. 45-48 § Geographic Breakdown (Modified: 2024-09-30)
[3] "EMEA Market Report" p. 8 § Growth Metrics (Modified: 2024-10-01)

---

### Assessment
**Confidence:** high

**Coverage:** Strong coverage of revenue metrics and geographic breakdown. Multiple corroborating sources.

**Gaps identified:** No information found on revenue by product line
```

---

## Azure AI Search Index Schema

For full source attribution, your Azure AI Search index needs these fields. Adjust field names in `METADATA_FIELD_MAPPING` in `tools/search.py` if yours differ:

```json
{
  "name": "docling-chunks-index",
  "fields": [
    {"name": "id", "type": "Edm.String", "key": true, "filterable": true},
    {"name": "chunk_id", "type": "Edm.String", "filterable": true},
    {"name": "content", "type": "Edm.String", "searchable": true, "analyzer": "standard.lucene"},
    {"name": "content_vector", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 1536, "vectorSearchProfile": "default"},
    
    {"name": "document_id", "type": "Edm.String", "filterable": true, "facetable": true},
    {"name": "title", "type": "Edm.String", "searchable": true, "filterable": true, "retrievable": true},
    
    {"name": "page_numbers", "type": "Collection(Edm.Int32)", "filterable": true, "retrievable": true},
    {"name": "section_title", "type": "Edm.String", "searchable": true, "filterable": true, "retrievable": true},
    {"name": "hierarchy_path", "type": "Edm.String", "filterable": true, "retrievable": true},
    {"name": "hierarchy_level", "type": "Edm.Int32", "filterable": true, "retrievable": true},
    
    {"name": "author", "type": "Edm.String", "filterable": true, "facetable": true, "retrievable": true},
    {"name": "created_date", "type": "Edm.DateTimeOffset", "filterable": true, "sortable": true, "retrievable": true},
    {"name": "modified_date", "type": "Edm.DateTimeOffset", "filterable": true, "sortable": true, "retrievable": true},
    {"name": "indexed_at", "type": "Edm.DateTimeOffset", "filterable": true, "retrievable": true},
    
    {"name": "document_type", "type": "Edm.String", "filterable": true, "facetable": true, "retrievable": true},
    {"name": "tags", "type": "Collection(Edm.String)", "filterable": true, "facetable": true, "retrievable": true},
    
    {"name": "chunk_type", "type": "Edm.String", "filterable": true, "retrievable": true},
    {"name": "parent_id", "type": "Edm.String", "filterable": true, "retrievable": true}
  ],
  "vectorSearch": {
    "profiles": [
      {"name": "default", "algorithm": "hnsw-default", "vectorizer": "openai-vectorizer"}
    ],
    "algorithms": [
      {"name": "hnsw-default", "kind": "hnsw"}
    ]
  }
}
```

**Important**: Ensure all metadata fields have `"retrievable": true` so they are returned with search results.

---

## Key Architecture Patterns

### Dynamic Parallelism with Send()

The `distribute_to_analysts` function returns a list of `Send("analyze_chunk", state)` objects. LangGraph automatically executes all sends in parallel within a single superstep, collecting results via the reducer on `analyses`. The number of analysts scales automatically with chunk count.

### Reducers for Accumulation

The `Annotated[List[ChunkAnalysis], operator.add]` pattern automatically concatenates results from all parallel analyst agents. The custom `concat_markdown` reducer joins document contributions with separators.

### Structured Output

Using `llm.with_structured_output(ChunkAnalysis)` ensures consistent, validated responses from smaller models, preventing parsing errors and enabling reliable downstream processing.

### Error Isolation

Each analyst operates independently with retry policies. If one fails after retries, others continue, and the synthesizer handles partial results gracefully.

### Source Attribution Flow

```
Azure AI Search → Document metadata extracted → Passed to analysts via Send() →
Analysts preserve metadata in ChunkAnalysis → Synthesizer groups by document →
Citations generated with page numbers, authors, dates → Final response formatted
```

---

*End of Implementation Plan*