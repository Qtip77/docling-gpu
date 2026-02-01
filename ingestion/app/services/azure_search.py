"""Bulk Azure AI Search upload with parallel batches."""
import asyncio
import logging
from typing import Optional

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

from app.config import settings
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

_index_client: Optional[SearchIndexClient] = None
_search_client: Optional[SearchClient] = None


def get_index_client() -> SearchIndexClient:
    global _index_client
    if _index_client is None:
        _index_client = SearchIndexClient(
            settings.azure_search_endpoint,
            AzureKeyCredential(settings.azure_search_key),
        )
    return _index_client


def get_search_client() -> SearchClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchClient(
            settings.azure_search_endpoint,
            settings.azure_search_index_name,
            AzureKeyCredential(settings.azure_search_key),
        )
    return _search_client


def ensure_index_exists():
    """Create search index if it doesn't exist."""
    client = get_index_client()
    index_name = settings.azure_search_index_name

    try:
        client.get_index(index_name)
        return
    except Exception:
        pass

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
        SearchField(
            name="page_numbers",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Int32),
            filterable=True,
        ),
        SearchableField(name="section_title", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="hierarchy_path", type=SearchFieldDataType.String),
        SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="indexed_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
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
    logger.info(f"Created index: {index_name}")


async def upload_chunks_bulk(
    doc_id: str,
    filename: str,
    chunks: list[DocumentChunk],
):
    """
    Upload chunks to Azure Search with parallel batch uploads.
    """
    if not chunks:
        return

    client = get_search_client()
    batch_size = settings.search_upload_batch_size
    max_concurrent = settings.search_max_concurrent

    # Prepare documents
    documents = []
    for chunk in chunks:
        doc = {
            "chunk_id": chunk.chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "content": chunk.content,
            "content_vector": chunk.embedding,
            "page_numbers": chunk.metadata.page_numbers,
            "section_title": chunk.metadata.section_title,
            "hierarchy_path": chunk.metadata.hierarchy_path,
            "chunk_type": chunk.metadata.chunk_type,
            "indexed_at": chunk.metadata.indexed_at,
        }
        documents.append(doc)

    # Split into batches
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

    if len(batches) == 1:
        client.upload_documents(documents=batches[0])
        logger.info(f"Uploaded {len(batches[0])} chunks for {filename}")
        return

    # Parallel batch upload
    semaphore = asyncio.Semaphore(max_concurrent)

    async def upload_batch(batch: list[dict]):
        async with semaphore:
            await asyncio.to_thread(client.upload_documents, documents=batch)

    logger.info(f"Uploading {len(documents)} chunks in {len(batches)} batches")
    tasks = [upload_batch(batch) for batch in batches]
    await asyncio.gather(*tasks)
    logger.info(f"Upload complete for {filename}")


def delete_document_chunks(doc_id: str) -> int:
    """Delete all chunks for a document."""
    client = get_search_client()

    results = client.search(
        search_text="*",
        filter=f"doc_id eq '{doc_id}'",
        select=["chunk_id"],
        top=1000,
    )

    chunk_ids = [r["chunk_id"] for r in results]
    if chunk_ids:
        documents = [{"chunk_id": cid} for cid in chunk_ids]
        client.delete_documents(documents=documents)

    return len(chunk_ids)


def get_documents() -> list[dict]:
    """Get unique documents from index."""
    client = get_search_client()

    results = client.search(
        search_text="*",
        select=["doc_id", "filename"],
        top=1000,
    )

    docs = {}
    for r in results:
        doc_id = r["doc_id"]
        if doc_id not in docs:
            docs[doc_id] = {"doc_id": doc_id, "filename": r["filename"], "chunks_count": 0}
        docs[doc_id]["chunks_count"] += 1

    return list(docs.values())
