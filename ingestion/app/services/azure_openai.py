"""Batch embedding service with concurrent requests."""
import asyncio
import logging
from typing import Optional

import httpx
from openai import AsyncAzureOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncAzureOpenAI] = None


def get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts using batch processing.
    Handles batching and concurrent requests automatically.
    """
    if not texts:
        return []

    client = get_client()
    batch_size = settings.embedding_batch_size
    max_concurrent = settings.embedding_max_concurrent

    # Split into batches
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    if len(batches) == 1:
        # Single batch, no concurrency needed
        return await _embed_batch(client, batches[0])

    # Multiple batches - process concurrently with limit
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await _embed_batch(client, batch)

    logger.info(f"Embedding {len(texts)} texts in {len(batches)} batches")
    tasks = [process_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)

    # Flatten results
    embeddings = []
    for batch_embeddings in results:
        embeddings.extend(batch_embeddings)

    return embeddings


async def _embed_batch(client: AsyncAzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a single batch of texts."""
    response = await client.embeddings.create(
        input=texts,
        model=settings.azure_openai_embeddings,
    )
    # Sort by index to maintain order
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [d.embedding for d in sorted_data]


async def embed_text(text: str) -> list[float]:
    """Embed single text (convenience wrapper)."""
    results = await embed_texts([text])
    return results[0] if results else []
