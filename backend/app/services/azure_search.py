from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizableTextQuery

from app.config import settings


_index_client: SearchIndexClient | None = None
_search_client: SearchClient | None = None


def get_index_client() -> SearchIndexClient:
    global _index_client
    if _index_client is None:
        _index_client = SearchIndexClient(
            settings.azure_search_endpoint,
            AzureKeyCredential(settings.azure_search_key)
        )
    return _index_client


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            settings.azure_search_endpoint,
            settings.azure_search_index_name,
            AzureKeyCredential(settings.azure_search_key)
        )
    return _search_client


def delete_index():
    """Delete the search index if it exists."""
    client = get_index_client()
    index_name = settings.azure_search_index_name
    try:
        client.delete_index(index_name)
    except Exception:
        pass


def ensure_index_exists(force_recreate: bool = False):
    """Create the search index if it doesn't exist."""
    client = get_index_client()
    index_name = settings.azure_search_index_name
    
    if force_recreate:
        delete_index()
    
    # Check if index exists
    try:
        client.get_index(index_name)
        return  # Index exists
    except Exception:
        pass  # Index doesn't exist, create it
    
    fields = [
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
    
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default")],
        profiles=[
            VectorSearchProfile(
                name="default",
                algorithm_configuration_name="default",
                vectorizer_name="default",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="default",
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=settings.azure_openai_endpoint,
                    deployment_name=settings.azure_openai_embeddings,
                    model_name=settings.azure_openai_embeddings_model,
                    api_key=settings.azure_openai_api_key,
                ),
            )
        ],
    )
    
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    client.create_or_update_index(index)


def upload_chunks(doc_id: str, filename: str, chunks: list[tuple[str, str, list[float], dict]]):
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


def search_chunks(query: str, embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Hybrid search returning chunks with full metadata for citations.
    Returns list of {chunk_id, content, score, filename, doc_id, page_numbers, section_title, hierarchy_path, chunk_type}.
    """
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


def delete_document_chunks(doc_id: str):
    """Delete all chunks belonging to a document."""
    client = get_search_client()
    
    # Search for all chunks with this doc_id
    results = client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["chunk_id"],
        top=1000
    )
    
    chunk_ids = [r["chunk_id"] for r in results]
    if chunk_ids:
        documents = [{"chunk_id": cid} for cid in chunk_ids]
        client.delete_documents(documents=documents)
    
    return len(chunk_ids)


def get_documents() -> list[dict]:
    """Get unique documents from the index."""
    client = get_search_client()
    
    # Get all unique doc_ids with their filenames and chunk counts
    results = client.search(
        search_text="*",
        select=["doc_id", "filename"],
        top=1000
    )
    
    # Group by doc_id
    docs = {}
    for r in results:
        doc_id = r["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {"doc_id": doc_id, "filename": r["filename"], "chunks_count": 0}
        docs[doc_id]["chunks_count"] += 1
    
    return list(docs.values())
