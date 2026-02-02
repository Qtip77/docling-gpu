"""Hierarchical markdown chunker for OCR output with page-aware chunking."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.models.schemas import ChunkMetadata, DocumentChunk
from app.services.mistral_ocr import OCRResult, PageResult, generate_chunk_id


@dataclass
class ChunkConfig:
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    overlap: int = 100
    preserve_tables: bool = True  # Keep tables as separate chunks


def chunk_ocr_result(
    ocr_result: OCRResult,
    filename: str,
    config: Optional[ChunkConfig] = None,
) -> list[DocumentChunk]:
    """
    Chunk OCR result with page-level metadata preservation.
    Handles images, tables, and hyperlinks from Mistral output.
    """
    if config is None:
        config = ChunkConfig()

    chunks: list[DocumentChunk] = []
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    chunk_idx = 0

    for page in ocr_result.pages:
        page_num = page.index + 1  # 1-indexed for users
        markdown = page.markdown

        # Replace image/table placeholders with inline content markers
        markdown = _expand_placeholders(markdown, page)

        # Split by headers while preserving hierarchy
        sections = _split_by_headers(markdown)

        for section in sections:
            header_stack = section["headers"]
            content = section["content"].strip()

            if not content or len(content) < config.min_chunk_size:
                continue

            # Handle tables separately if configured
            if config.preserve_tables:
                table_chunks, remaining = _extract_table_chunks(
                    content, filename, page_num, header_stack, chunk_idx, now
                )
                chunks.extend(table_chunks)
                chunk_idx += len(table_chunks)
                content = remaining

            if not content or len(content) < config.min_chunk_size:
                continue

            # Split large sections into smaller chunks
            sub_chunks = _split_content(content, config)

            for sub_content in sub_chunks:
                if len(sub_content.strip()) < config.min_chunk_size:
                    continue

                chunk_id = generate_chunk_id(filename, chunk_idx)
                hierarchy = " > ".join(header_stack) if header_stack else None
                section_title = header_stack[-1] if header_stack else None
                chunk_type = _detect_chunk_type(sub_content)

                # Add hyperlinks context if present on this page
                hyperlink_context = _format_hyperlinks(page.hyperlinks, sub_content)
                if hyperlink_context:
                    sub_content = f"{sub_content}\n\n{hyperlink_context}"

                metadata = ChunkMetadata(
                    page_numbers=[page_num],
                    section_title=section_title,
                    hierarchy_path=hierarchy,
                    source=filename,
                    chunk_type=chunk_type,
                    indexed_at=now,
                )

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=sub_content.strip(),
                    metadata=metadata,
                ))
                chunk_idx += 1

    return chunks


def _expand_placeholders(markdown: str, page: PageResult) -> str:
    """Replace image/table placeholders with content markers for better retrieval."""
    # Replace image placeholders like ![img-0.jpeg](img-0.jpeg)
    for img in page.images:
        placeholder = f"![{img.id}]({img.id})"
        marker = f"[IMAGE: {img.id}]"
        markdown = markdown.replace(placeholder, marker)

    # Replace table placeholders like [tbl-3.html](tbl-3.html)
    for tbl in page.tables:
        # Handle both link and raw placeholder formats
        placeholder_link = f"[{tbl.id}]({tbl.id})"
        placeholder_raw = tbl.id
        if tbl.content:
            # Inline the table content
            markdown = markdown.replace(placeholder_link, f"\n{tbl.content}\n")
            markdown = markdown.replace(placeholder_raw, f"\n{tbl.content}\n")
        else:
            marker = f"[TABLE: {tbl.id}]"
            markdown = markdown.replace(placeholder_link, marker)

    return markdown


def _extract_table_chunks(
    content: str,
    filename: str,
    page_num: int,
    headers: list[str],
    start_idx: int,
    timestamp: str,
) -> tuple[list[DocumentChunk], str]:
    """Extract tables as separate chunks for better structured retrieval."""
    chunks = []
    chunk_idx = start_idx
    remaining = content

    # Match markdown pipe tables
    md_table_pattern = re.compile(
        r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n?)+)",
        re.MULTILINE
    )
    # Match HTML tables (Mistral often returns these)
    html_table_pattern = re.compile(
        r"(<table[^>]*>[\s\S]*?</table>)",
        re.IGNORECASE
    )

    def create_table_chunk(table_content: str) -> DocumentChunk:
        nonlocal chunk_idx
        chunk_id = generate_chunk_id(filename, chunk_idx)
        hierarchy = " > ".join(headers) if headers else None
        metadata = ChunkMetadata(
            page_numbers=[page_num],
            section_title=headers[-1] if headers else None,
            hierarchy_path=hierarchy,
            source=filename,
            chunk_type="table",
            indexed_at=timestamp,
        )
        chunk_idx += 1
        return DocumentChunk(chunk_id=chunk_id, content=table_content, metadata=metadata)

    # Extract markdown tables
    for match in md_table_pattern.finditer(content):
        table_content = match.group(1).strip()
        if len(table_content) < 20:
            continue
        chunks.append(create_table_chunk(table_content))
        remaining = remaining.replace(match.group(0), "\n[TABLE]\n", 1)

    # Extract HTML tables
    for match in html_table_pattern.finditer(remaining):
        table_content = match.group(1).strip()
        if len(table_content) < 20:
            continue
        chunks.append(create_table_chunk(table_content))
        remaining = remaining.replace(match.group(0), "\n[TABLE]\n", 1)

    return chunks, remaining


def _format_hyperlinks(hyperlinks: list[dict], content: str) -> str:
    """Format relevant hyperlinks as context."""
    if not hyperlinks:
        return ""

    # Only include hyperlinks whose text appears in the content
    relevant = []
    for link in hyperlinks:
        text = link.get("text", "")
        url = link.get("url", "")
        if text and url and text.lower() in content.lower():
            relevant.append(f"- [{text}]({url})")

    if relevant:
        return "**Links:**\n" + "\n".join(relevant[:5])  # Limit to 5
    return ""


def _split_by_headers(markdown: str) -> list[dict]:
    """Split markdown by headers, tracking hierarchy."""
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections = []
    current_headers: list[str] = []
    current_content: list[str] = []
    last_level = 0

    for line in markdown.split("\n"):
        match = header_pattern.match(line)
        if match:
            if current_content:
                sections.append({
                    "headers": current_headers.copy(),
                    "content": "\n".join(current_content),
                })
                current_content = []

            level = len(match.group(1))
            header_text = match.group(2).strip()

            if level <= last_level:
                current_headers = current_headers[: level - 1]
            current_headers.append(header_text)
            last_level = level
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "headers": current_headers.copy(),
            "content": "\n".join(current_content),
        })

    return sections


def _split_content(content: str, config: ChunkConfig) -> list[str]:
    """Split content into chunks respecting size limits with overlap."""
    if len(content) <= config.max_chunk_size:
        return [content]

    chunks = []
    paragraphs = re.split(r"\n\n+", content)
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= config.max_chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Add overlap from end of previous chunk
                if config.overlap > 0:
                    overlap_text = current_chunk[-config.overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = para

            # Handle paragraphs larger than max size
            if len(current_chunk) > config.max_chunk_size:
                sub_chunks = _split_by_sentences(current_chunk, config)
                chunks.extend(sub_chunks[:-1])
                current_chunk = sub_chunks[-1] if sub_chunks else ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _split_by_sentences(text: str, config: ChunkConfig) -> list[str]:
    """Split text by sentences when paragraphs are too large."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= config.max_chunk_size:
            current = f"{current} {sent}" if current else sent
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


def _detect_chunk_type(content: str) -> str:
    """Detect content type from markdown patterns."""
    if re.search(r"^\|.*\|$", content, re.MULTILINE):
        return "table"
    if re.search(r"^```", content, re.MULTILINE):
        return "code"
    if re.search(r"^\s*[-*]\s", content, re.MULTILINE):
        return "list"
    if re.search(r"^\s*\d+\.\s", content, re.MULTILINE):
        return "list"
    if re.search(r"\[IMAGE:", content):
        return "image_context"
    return "text"


# Legacy function for backward compatibility
def chunk_markdown(
    markdown: str,
    filename: str,
    config: Optional[ChunkConfig] = None,
) -> list[DocumentChunk]:
    """Legacy: Chunk plain markdown (no page metadata)."""
    if config is None:
        config = ChunkConfig()

    from app.services.mistral_ocr import PageResult, OCRResult

    # Wrap in single-page OCRResult
    fake_result = OCRResult(
        pages=[PageResult(index=0, markdown=markdown)],
        model="legacy",
    )
    return chunk_ocr_result(fake_result, filename, config)
