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
    "document_id": "doc_id",           # matches existing index
    "document_title": "filename",       # matches existing index
    
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
    - AZURE_SEARCH_ENDPOINT (from existing config)
    - AZURE_SEARCH_INDEX_NAME (from existing config)
    - AZURE_SEARCH_KEY (from existing config)
    """
    filter_expr = construct_odata_filter(filters)
    
    # Extract service name from endpoint
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    service_name = endpoint.replace("https://", "").replace(".search.windows.net", "")
    
    return AzureAISearchRetriever(
        service_name=service_name,
        index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
        api_key=os.getenv("AZURE_SEARCH_KEY"),
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
