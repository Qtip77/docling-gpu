"""Hierarchical markdown chunker for OCR output."""
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.models.schemas import ChunkMetadata, DocumentChunk
from app.services.mistral_ocr import generate_chunk_id


@dataclass
class ChunkConfig:
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    overlap: int = 100


def chunk_markdown(
    markdown: str,
    filename: str,
    config: Optional[ChunkConfig] = None,
) -> list[DocumentChunk]:
    """
    Chunk markdown content hierarchically by headers.
    Preserves document structure and hierarchy context.
    """
    if config is None:
        config = ChunkConfig()

    chunks: list[DocumentChunk] = []
    now = datetime.utcnow().isoformat() + "Z"

    # Split by headers while preserving hierarchy
    sections = _split_by_headers(markdown)

    chunk_idx = 0
    for section in sections:
        header_stack = section["headers"]
        content = section["content"].strip()

        if not content or len(content) < config.min_chunk_size:
            continue

        # Split large sections into smaller chunks
        sub_chunks = _split_content(content, config)

        for sub_content in sub_chunks:
            if len(sub_content.strip()) < config.min_chunk_size:
                continue

            chunk_id = generate_chunk_id(filename, chunk_idx)

            # Build hierarchy path from header stack
            hierarchy = " > ".join(header_stack) if header_stack else None
            section_title = header_stack[-1] if header_stack else None

            # Detect chunk type
            chunk_type = _detect_chunk_type(sub_content)

            # Extract page numbers if present in content (e.g., <!-- page: 5 -->)
            page_numbers = _extract_page_numbers(sub_content)

            metadata = ChunkMetadata(
                page_numbers=page_numbers,
                section_title=section_title,
                hierarchy_path=hierarchy,
                source=filename,
                chunk_type=chunk_type,
                indexed_at=now,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=sub_content.strip(),
                    metadata=metadata,
                )
            )
            chunk_idx += 1

    return chunks


def _split_by_headers(markdown: str) -> list[dict]:
    """Split markdown by headers, tracking hierarchy."""
    # Match headers: # H1, ## H2, etc.
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections = []
    current_headers: list[str] = []
    current_content: list[str] = []
    last_level = 0

    lines = markdown.split("\n")
    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save previous section
            if current_content:
                sections.append({
                    "headers": current_headers.copy(),
                    "content": "\n".join(current_content),
                })
                current_content = []

            # Update header stack
            level = len(match.group(1))
            header_text = match.group(2).strip()

            # Adjust stack based on header level
            if level <= last_level:
                current_headers = current_headers[: level - 1]
            current_headers.append(header_text)
            last_level = level
        else:
            current_content.append(line)

    # Don't forget last section
    if current_content:
        sections.append({
            "headers": current_headers.copy(),
            "content": "\n".join(current_content),
        })

    return sections


def _split_content(content: str, config: ChunkConfig) -> list[str]:
    """Split content into chunks respecting size limits."""
    if len(content) <= config.max_chunk_size:
        return [content]

    chunks = []
    # Try to split on paragraph boundaries
    paragraphs = re.split(r"\n\n+", content)

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= config.max_chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Handle paragraphs larger than max size
            if len(para) > config.max_chunk_size:
                # Split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) + 1 <= config.max_chunk_size:
                        current_chunk = f"{current_chunk} {sent}" if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _detect_chunk_type(content: str) -> str:
    """Detect content type from markdown patterns."""
    if re.search(r"^\|.*\|$", content, re.MULTILINE):
        return "table"
    if re.search(r"^```", content, re.MULTILINE):
        return "code"
    if re.search(r"^[-*]\s", content, re.MULTILINE):
        return "list"
    if re.search(r"^\d+\.\s", content, re.MULTILINE):
        return "list"
    return "text"


def _extract_page_numbers(content: str) -> list[int]:
    """Extract page numbers from content annotations."""
    # Look for page markers like <!-- page: 5 --> or [page 5]
    pages = set()
    for match in re.finditer(r"(?:<!--\s*page:\s*(\d+)\s*-->|\[page\s+(\d+)\])", content):
        page = match.group(1) or match.group(2)
        if page:
            pages.add(int(page))
    return sorted(pages)
