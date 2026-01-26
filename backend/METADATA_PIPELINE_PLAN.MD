# Metadata Pipeline Upgrade Plan

This plan details the necessary refactoring to enable full source attribution (page numbers, section titles, authors) in the RAG system. This bridges the gap between the rich data extracted by Docling and the requirements of the multi-agent system.

---

## 1. Strategy Overview

We will upgrade the pipeline to pass a **Metadata Dictionary** alongside the text content from ingestion through to indexing.

**Current Flow (Data Loss):**
`Docling` -> `(chunk_id, text)` -> `Router` -> `Azure Search`

**Target Flow (Rich Metadata):**
`Docling` -> `(chunk_id, text, metadata)` -> `Router` -> `Azure Search` -> `RAG Agent`

---

## 2. Implementation Steps

### Step 1: Define Metadata Structure
We need a standard structure to pass between services. Field names must align with the existing RAG agent's `METADATA_FIELD_MAPPING` in `rag_agent/tools/search.py`.

**File:** `backend/app/models/schemas.py`

```python
from typing import TypedDict, List, Optional

class ChunkMetadata(TypedDict):
    """Metadata extracted from Docling chunks for Azure AI Search indexing."""
    page_numbers: List[int]
    section_title: Optional[str]
    hierarchy_path: Optional[str]
    source: str                    # filename - stored at chunk level for consistency
    chunk_type: str                # "paragraph", "heading", "table", etc.
    created_date: Optional[str]    # ISO format datetime
    indexed_at: str                # ISO format datetime - when chunk was indexed
```

### Step 2: Refactor Docling Parser
Update extraction logic to pull metadata from `docling` objects.

**File:** `backend/app/services/docling_parser.py`

**Changes:**
1.  Update `parse_document` return type hint to `list[tuple[str, str, dict]]`.
2.  Update `_extract_chunks` to iterate `DocChunk` objects and extract:
    *   **Page Numbers**: From `chunk.prov` (provenance items) with graceful fallback.
    *   **Section Title**: From `chunk.meta.headings`.
    *   **Hierarchy**: Construct from headings list.
    *   **Chunk Type**: From `chunk.meta.doc_item_type`.

**Code Preview:**
```python
from datetime import datetime

def _extract_chunks(result, file_path: str) -> list[tuple[str, str, dict]]:
    chunker = HierarchicalChunker()
    doc_chunks = list(chunker.chunk(result.document))
    
    filename_hash = hashlib.md5(Path(file_path).name.encode("utf-8")).hexdigest()
    
    # Extract document-level metadata (if available)
    doc_created = None
    if hasattr(result.document, 'origin') and result.document.origin:
        if hasattr(result.document.origin, 'created_at'):
            doc_created = result.document.origin.created_at
    
    chunks = []
    for idx, chunk in enumerate(doc_chunks):
        chunk_id = f"{filename_hash}_chunk_{idx}"
        
        # Extract page numbers from provenance (with graceful fallback)
        pages = []
        if hasattr(chunk, 'prov') and chunk.prov:
            pages = sorted(list(set(
                prov.page_no for prov in chunk.prov 
                if hasattr(prov, 'page_no') and prov.page_no is not None
            )))
        
        # Extract section context
        headings = []
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings'):
            headings = chunk.meta.headings or []
        section_title = headings[-1] if headings else None
        hierarchy_path = " > ".join(headings) if headings else None
        
        # Extract chunk type
        chunk_type = "text"
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_item_type'):
            chunk_type = str(chunk.meta.doc_item_type) if chunk.meta.doc_item_type else "text"
        
        meta = {
            "page_numbers": pages,
            "section_title": section_title,
            "hierarchy_path": hierarchy_path,
            "source": Path(file_path).name,
            "chunk_type": chunk_type,
            "created_date": doc_created.isoformat() if doc_created else None,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }
        
        chunks.append((chunk_id, chunk.text, meta))
    return chunks
```

### Step 3: Upgrade Azure Search Schema
Update the index definition to store the new fields.

**File:** `backend/app/services/azure_search.py`

**Changes:**

#### 3.1 Add new fields to `ensure_index_exists`:

```python
fields = [
    # Existing fields
    SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=settings.vector_dim,
        vector_search_profile_name="default",
    ),
    
    # NEW: Metadata fields for source attribution
    SearchField(
        name="page_numbers",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Int32),
        filterable=True,
    ),
    SearchableField(
        name="section_title",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SearchableField(
        name="hierarchy_path",
        type=SearchFieldDataType.String,
    ),
    SimpleField(
        name="chunk_type",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SimpleField(
        name="created_date",
        type=SearchFieldDataType.DateTimeOffset,
        filterable=True,
        sortable=True,
    ),
    SimpleField(
        name="indexed_at",
        type=SearchFieldDataType.DateTimeOffset,
        filterable=True,
        sortable=True,
    ),
]
```

#### 3.2 Update `upload_chunks` signature:

```python
def upload_chunks(
    doc_id: str,
    filename: str,
    chunks: list[tuple[str, str, list[float], dict]]  # Added metadata dict
):
    """
    Upload document chunks with embeddings and metadata to Azure AI Search.
    chunks: list of (chunk_id, content, embedding_vector, metadata)
    """
    client = get_search_client()
    
    documents = []
    for chunk_id, content, embedding, meta in chunks:
        doc = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "content": content,
            "content_vector": embedding,
            # Map metadata fields
            "page_numbers": meta.get("page_numbers", []),
            "section_title": meta.get("section_title"),
            "hierarchy_path": meta.get("hierarchy_path"),
            "chunk_type": meta.get("chunk_type", "text"),
            "created_date": meta.get("created_date"),
            "indexed_at": meta.get("indexed_at"),
        }
        documents.append(doc)
    
    # Upload in batches of 50
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        client.upload_documents(documents=batch)
```

#### 3.3 Update `search_chunks` to return metadata:

```python
def search_chunks(query: str, embedding: list[float], top_k: int = 5) -> list[dict]:
    """Hybrid search returning chunks with full metadata for citations."""
    client = get_search_client()
    
    vector_query = VectorizableTextQuery(
        text=query,
        k_nearest_neighbors=top_k,
        fields="content_vector",
    )
    
    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=[
            "chunk_id", "content", "filename", "doc_id",
            "page_numbers", "section_title", "hierarchy_path", "chunk_type"
        ],
        top=top_k
    )
    
    return [
        {
            "chunk_id": r["chunk_id"],
            "content": r["content"],
            "score": r["@search.score"],
            "filename": r.get("filename"),
            "doc_id": r.get("doc_id"),
            "page_numbers": r.get("page_numbers", []),
            "section_title": r.get("section_title"),
            "hierarchy_path": r.get("hierarchy_path"),
            "chunk_type": r.get("chunk_type"),
        }
        for r in results
    ]
```

**Critical Note:** Since we are adding fields, we must **delete and recreate** the index for the changes to take effect cleanly during development.

### Step 4: Update Router Integration
Connect the parser output to the search uploader.

**File:** `backend/app/routers/documents.py`

**Changes:**
```python
async def process_document(job_id: str, file_path: str, filename: str, use_vlm: bool = False):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Starting processing job {job_id} for {filename} (VLM: {use_vlm})")
        jobs[job_id].status = "processing"
        
        # Parse and chunk document - now returns (chunk_id, content, metadata)
        logger.info(f"Parsing document: {file_path}")
        chunks = docling_parser.parse_document(file_path, use_vlm=use_vlm)
        logger.info(f"Parsed {len(chunks)} chunks from {filename}")
        
        # Generate embeddings and prepare for upload with metadata
        chunks_with_embeddings = []
        for i, (chunk_id, content, meta) in enumerate(chunks):
            logger.info(f"Generating embedding {i+1}/{len(chunks)}")
            embedding = azure_openai.embed_text(content)
            chunks_with_embeddings.append((chunk_id, content, embedding, meta))
        
        # Upload to Azure AI Search (with metadata)
        logger.info(f"Uploading {len(chunks_with_embeddings)} chunks to Azure Search")
        azure_search.upload_chunks(job_id, filename, chunks_with_embeddings)
        
        jobs[job_id].status = "completed"
        jobs[job_id].chunks_count = len(chunks)
        logger.info(f"Job {job_id} completed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)
```

### Step 5: Verify RAG Agent Compatibility
The RAG agent already expects these fields via `METADATA_FIELD_MAPPING` in `rag_agent/tools/search.py`.

**Verification Checklist:**
- [ ] `extract_source_metadata()` in `search.py` receives populated metadata from search results
- [ ] `SourceMetadata` in `schemas/outputs.py` fields match index field names
- [ ] Analyst nodes pass metadata through to synthesizer for citation generation

**No code changes required** if field names match the existing mapping. The agent's `retrieve_chunks()` function uses `AzureAISearchRetriever` which automatically includes all indexed fields in document metadata.

---

## 3. Migration / Execution Strategy

Since this changes the index schema, existing data will be incompatible or missing fields.

1.  **Stop Backend**: Ensure no uploads are in progress.
2.  **Apply Code Changes**: Implement the refactors in order (Steps 1-4).
3.  **Reset Index**:
    *   Option A: Delete the index in Azure Portal
    *   Option B: Call `ensure_index_exists(force_recreate=True)` on first run
    *   Option C: Add temporary startup script: `azure_search.delete_index()`
4.  **Re-upload Documents**: Run `upload_docs.py` to re-process all PDFs with full metadata.

---

## 4. Verification

After re-uploading, verify the full pipeline:

### 4.1 Index Verification
1.  Upload a document (e.g., "Annual Report.pdf").
2.  Wait for processing to complete.
3.  Check Azure Portal > Search Service > Index > Documents.
    *   Confirm `page_numbers` field is populated (e.g., `[1, 2]`).
    *   Confirm `section_title` is populated (e.g., "Financial Highlights").
    *   Confirm `chunk_type` shows values like "paragraph", "heading", "table".
    *   Confirm `indexed_at` has a valid timestamp.

### 4.2 RAG Agent Verification
Run a RAG query and verify citations include metadata:
```
Query: "What was the revenue growth?"

Expected Response:
"Revenue grew by 5% year-over-year [1]"

[1] Annual Report.pdf, p. 5, § Financial Highlights
```

### 4.3 API Response Verification
Check that `/api/search` endpoint returns metadata in results:
```json
{
  "results": [
    {
      "chunk_id": "abc123_chunk_5",
      "content": "Revenue increased by 5%...",
      "score": 0.92,
      "page_numbers": [5],
      "section_title": "Financial Highlights",
      "hierarchy_path": "Annual Report > Financial Overview > Financial Highlights"
    }
  ]
}
```

---

## 5. Summary of Files Changed

| Order | File | Changes |
|-------|------|---------|
| 1 | `app/models/schemas.py` | Add `ChunkMetadata` TypedDict |
| 2 | `app/services/docling_parser.py` | Update `_extract_chunks()` to return metadata |
| 3 | `app/services/azure_search.py` | Add index fields, update `upload_chunks()` and `search_chunks()` |
| 4 | `app/routers/documents.py` | Update tuple unpacking to include metadata |
| 5 | *(verify only)* `app/rag_agent/tools/search.py` | Confirm field mapping compatibility |
