"""Mistral OCR service using Azure AI Services REST API."""
import asyncio
import base64
import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from pypdf import PdfReader, PdfWriter

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


def _split_pdf(pdf_bytes: bytes, chunk_size: int | None = None) -> list[bytes]:
    """Split PDF into chunks of chunk_size pages, return list of PDF bytes."""
    if chunk_size is None:
        chunk_size = settings.ocr_max_pages_per_request
    
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    
    if total_pages <= chunk_size:
        return [pdf_bytes]
    
    chunks = []
    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        end = min(start + chunk_size, total_pages)
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())
    
    logger.info(f"Split {total_pages}-page PDF into {len(chunks)} chunks")
    return chunks


def _parse_ocr_response(data: dict, page_offset: int = 0) -> list[PageResult]:
    """Parse OCR API response into PageResult list."""
    pages = []
    for p in data.get("pages", []):
        images = [
            PageImage(
                id=img.get("id", ""),
                base64=img.get("image_base64"),
                top_left=(img.get("top_left_x", 0), img.get("top_left_y", 0)) if "top_left_x" in img else (0, 0),
                bottom_right=(img.get("bottom_right_x", 0), img.get("bottom_right_y", 0)) if "bottom_right_x" in img else (0, 0),
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
            index=p.get("index", 0) + page_offset,  # Adjust for chunk offset
            markdown=p.get("markdown", ""),
            images=images,
            tables=tables,
            hyperlinks=p.get("hyperlinks", []),
            header=p.get("header"),
            footer=p.get("footer"),
            dimensions=p.get("dimensions", {}),
        ))
    return pages


async def _ocr_single_chunk(
    client: httpx.AsyncClient, 
    doc_base64: str, 
    mime_type: str,
    include_images: bool = False
) -> dict:
    """Call OCR API for a single document chunk with retries."""
    payload = {
        "model": settings.mistral_azure_model,
        "document": {
            "type": "document_url",
            "document_url": f"data:{mime_type};base64,{doc_base64}",
        },
        "include_image_base64": include_images,
    }

    max_retries = 3
    base_delay = 2.0
    
    for attempt in range(max_retries):
        response = await client.post(
            settings.mistral_azure_endpoint,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.mistral_azure_api_key}",
            },
        )
        
        if response.status_code < 500:
            break
            
        logger.warning(f"OCR API error {response.status_code} (attempt {attempt + 1}/{max_retries}): {response.text}")
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"Retrying in {delay}s...")
            await asyncio.sleep(delay)
    
    if response.status_code >= 400:
        logger.error(f"OCR API error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()


async def process_document(file_path: str, include_images: bool = False) -> OCRResult:
    """
    Process document with Mistral Document AI on Azure.
    Automatically splits large PDFs (>30 pages) into chunks.
    
    Args:
        file_path: Path to document file
        include_images: Include base64 image data for vision processing
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        doc_bytes = f.read()

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

    # Split large PDFs into chunks
    if suffix == ".pdf":
        chunks = _split_pdf(doc_bytes)
    else:
        chunks = [doc_bytes]

    all_pages = []
    model_name = ""
    total_usage = {}
    
    max_pages = settings.ocr_max_pages_per_request
    for chunk_idx, chunk_bytes in enumerate(chunks):
        chunk_b64 = base64.standard_b64encode(chunk_bytes).decode("utf-8")
        page_offset = chunk_idx * max_pages
        
        logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (pages {page_offset + 1}-{page_offset + max_pages})")
        
        data = await _ocr_single_chunk(client, chunk_b64, mime_type, include_images)
        
        pages = _parse_ocr_response(data, page_offset)
        all_pages.extend(pages)
        
        if not model_name:
            model_name = data.get("model", "")
        
        # Accumulate usage
        for k, v in data.get("usage_info", {}).items():
            total_usage[k] = total_usage.get(k, 0) + v

    result = OCRResult(
        pages=all_pages,
        model=model_name,
        document_annotation=None,
        usage_info=total_usage,
    )

    logger.info(f"OCR complete: {len(all_pages)} pages")
    return result


def generate_chunk_id(filename: str, chunk_idx: int) -> str:
    """Generate unique chunk ID from filename hash."""
    filename_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()
    return f"{filename_hash}_chunk_{chunk_idx}"
