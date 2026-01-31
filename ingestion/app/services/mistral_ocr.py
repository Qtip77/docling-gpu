"""Mistral OCR service with parallel page batch processing."""
import asyncio
import base64
import hashlib
import logging
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def _ocr_batch(
    client: httpx.AsyncClient,
    doc_base64: str,
    pages: list[int],
    doc_type: str = "document_url",
) -> dict:
    """Process a batch of pages with Mistral OCR."""
    # Mistral Azure MaaS OCR endpoint
    url = f"{settings.mistral_azure_endpoint.rstrip('/')}/v1/ocr"

    payload = {
        "model": "mistral-ocr-latest",
        "document": {
            "type": "base64",
            "data": doc_base64,
        },
        "include_image_base64": False,
    }

    # Add page range if specified
    if pages:
        payload["pages"] = pages

    headers = {
        "Authorization": f"Bearer {settings.mistral_azure_api_key}",
        "Content-Type": "application/json",
    }

    response = await client.post(
        url,
        json=payload,
        headers=headers,
        timeout=settings.ocr_timeout,
    )
    response.raise_for_status()
    return response.json()


async def process_document(file_path: str) -> tuple[str, int]:
    """
    Process document with Mistral OCR using parallel page batches.
    Returns (markdown_content, page_count).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read and encode file
    with open(file_path, "rb") as f:
        doc_bytes = f.read()
    doc_base64 = base64.standard_b64encode(doc_bytes).decode("utf-8")

    async with httpx.AsyncClient() as client:
        # First, get page count with a minimal request
        initial = await _ocr_batch(client, doc_base64, [1])
        
        # Extract page count from response
        page_count = initial.get("pages_processed", 1)
        if "document" in initial and "page_count" in initial["document"]:
            page_count = initial["document"]["page_count"]

        # For single-page or small docs, return immediately
        if page_count <= settings.ocr_page_batch_size:
            result = await _ocr_batch(client, doc_base64, [])
            markdown = _extract_markdown(result)
            return markdown, page_count

        # Create page batches for parallel processing
        batches = []
        batch_size = settings.ocr_page_batch_size
        for i in range(1, page_count + 1, batch_size):
            batch_pages = list(range(i, min(i + batch_size, page_count + 1)))
            batches.append(batch_pages)

        logger.info(f"Processing {page_count} pages in {len(batches)} batches")

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(settings.ocr_max_concurrent)

        async def process_batch(pages: list[int]) -> dict:
            async with semaphore:
                return await _ocr_batch(client, doc_base64, pages)

        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results in order
        all_markdown = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i} failed: {result}")
                continue
            all_markdown.append(_extract_markdown(result))

        combined = "\n\n".join(all_markdown)
        return combined, page_count


def _extract_markdown(result: dict) -> str:
    """Extract markdown from OCR response."""
    # Mistral OCR returns pages with markdown content
    if "pages" in result:
        parts = []
        for page in result["pages"]:
            if "markdown" in page:
                parts.append(page["markdown"])
        return "\n\n".join(parts)

    # Fallback: check for direct markdown field
    if "markdown" in result:
        return result["markdown"]

    # Last resort: check text field
    if "text" in result:
        return result["text"]

    return ""


def generate_chunk_id(filename: str, chunk_idx: int) -> str:
    """Generate unique chunk ID from filename hash."""
    filename_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{filename_hash}_chunk_{chunk_idx}"
