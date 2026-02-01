"""Mistral OCR service using Azure AI Services REST API."""
import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


@dataclass
class PageImage:
    """Image extracted from a page."""
    id: str
    base64: Optional[str] = None
    top_left: tuple[int, int] = (0, 0)
    bottom_right: tuple[int, int] = (0, 0)


@dataclass
class PageTable:
    """Table extracted from a page."""
    id: str
    content: str  # HTML or markdown
    top_left: tuple[int, int] = (0, 0)
    bottom_right: tuple[int, int] = (0, 0)


@dataclass
class PageResult:
    """OCR result for a single page."""
    index: int
    markdown: str
    images: list[PageImage] = field(default_factory=list)
    tables: list[PageTable] = field(default_factory=list)
    hyperlinks: list[dict] = field(default_factory=list)
    header: Optional[str] = None
    footer: Optional[str] = None
    dimensions: dict = field(default_factory=dict)


@dataclass
class OCRResult:
    """Full OCR result with all pages and metadata."""
    pages: list[PageResult]
    model: str
    document_annotation: Optional[dict] = None
    usage_info: dict = field(default_factory=dict)


def get_client() -> httpx.AsyncClient:
    """Get or create async HTTP client."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(settings.ocr_timeout))
    return _client


async def process_document(file_path: str) -> OCRResult:
    """
    Process document with Mistral Document AI on Azure.
    Returns full OCRResult with page-level metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        doc_bytes = f.read()
    doc_base64 = base64.standard_b64encode(doc_bytes).decode("utf-8")

    suffix = path.suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    mime_type = mime_map.get(suffix, "application/pdf")

    client = get_client()

    payload = {
        "model": "mistral-document-ai-2505",
        "document": {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{doc_base64}",
        },
        "include_image_base64": False,
    }

    response = await client.post(
        settings.mistral_azure_endpoint,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.mistral_azure_api_key}",
        },
    )
    response.raise_for_status()
    data = response.json()

    # Parse into structured result
    pages = []
    for p in data.get("pages", []):
        images = [
            PageImage(
                id=img.get("id", ""),
                base64=img.get("image_base64"),
                top_left=tuple(img.get("top_left_x", 0), img.get("top_left_y", 0)) if "top_left_x" in img else (0, 0),
                bottom_right=tuple(img.get("bottom_right_x", 0), img.get("bottom_right_y", 0)) if "bottom_right_x" in img else (0, 0),
            )
            for img in p.get("images", [])
        ]
        tables = [
            PageTable(
                id=tbl.get("id", ""),
                content=tbl.get("content", ""),
            )
            for tbl in p.get("tables", [])
        ]
        pages.append(PageResult(
            index=p.get("index", 0),
            markdown=p.get("markdown", ""),
            images=images,
            tables=tables,
            hyperlinks=p.get("hyperlinks", []),
            header=p.get("header"),
            footer=p.get("footer"),
            dimensions=p.get("dimensions", {}),
        ))

    result = OCRResult(
        pages=pages,
        model=data.get("model", ""),
        document_annotation=data.get("document_annotation"),
        usage_info=data.get("usage_info", {}),
    )

    logger.info(f"OCR complete: {len(pages)} pages")
    return result


def generate_chunk_id(filename: str, chunk_idx: int) -> str:
    """Generate unique chunk ID from filename hash."""
    filename_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{filename_hash}_chunk_{chunk_idx}"
