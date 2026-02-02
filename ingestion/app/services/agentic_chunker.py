"""Agentic chunker using Chonkie's SlumberChunker for intelligent document splitting.

Best for: contracts, legal agreements, vote ballots, cost estimates
Trade-off: ~2-5s per page (LLM calls) vs ~10ms for rule-based chunking
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Literal

from chonkie import SlumberChunker
from chonkie.genie import AzureOpenAIGenie
from openai import AzureOpenAI

from app.config import settings
from app.models.schemas import ChunkMetadata, DocumentChunk
from app.services.mistral_ocr import OCRResult, PageResult, PageImage, generate_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class AgenticChunkConfig:
    """Config for agentic chunking with SlumberChunker."""
    chunk_size: int = 1024  # Target tokens per chunk
    candidate_size: int = 256  # Tokens around split points for LLM to evaluate
    min_characters_per_chunk: int = 50
    model: str = "gpt-4o-mini"  # Cost-effective for chunking decisions
    preserve_tables: bool = True
    verbose: bool = False
    describe_images: bool = True  # Use vision model to describe images


_genie: Optional[AzureOpenAIGenie] = None
_chunker: Optional[SlumberChunker] = None
_vision_client: Optional[AzureOpenAI] = None


def get_vision_client() -> AzureOpenAI:
    """Get Azure OpenAI client for vision tasks."""
    global _vision_client
    if _vision_client is None:
        _vision_client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    return _vision_client


def _clean_base64(b64: str) -> str:
    """Clean base64 string for API consumption."""
    # Remove data URI prefix if present
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    # Remove whitespace/newlines
    b64 = b64.replace("\n", "").replace("\r", "").replace(" ", "")
    # Ensure proper padding
    padding = 4 - (len(b64) % 4)
    if padding != 4:
        b64 += "=" * padding
    return b64


def _validate_base64(b64: str) -> bool:
    """Check if base64 string is valid."""
    import base64
    try:
        decoded = base64.b64decode(b64)
        # Check for common image magic bytes
        if decoded[:2] == b'\xff\xd8':  # JPEG
            return True
        if decoded[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
            return True
        if decoded[:6] in (b'GIF87a', b'GIF89a'):  # GIF
            return True
        if decoded[:4] == b'RIFF' and decoded[8:12] == b'WEBP':  # WebP
            return True
        # Allow if it decodes successfully even without magic bytes
        return len(decoded) > 100
    except Exception:
        return False


def describe_image(image: PageImage, context: str = "") -> str:
    """Generate description for an image using vision model."""
    if not image.base64:
        return f"[IMAGE: {image.id}]"
    
    # Clean and validate base64
    clean_b64 = _clean_base64(image.base64)
    if not _validate_base64(clean_b64):
        logger.warning(f"Invalid base64 data for {image.id} (len={len(image.base64)}, prefix={image.base64[:50]}...)")
        return f"[IMAGE: {image.id}]"
    
    try:
        client = get_vision_client()
        
        # Determine image mime type from ID
        mime_type = "image/jpeg"
        if image.id.endswith(".png"):
            mime_type = "image/png"
        elif image.id.endswith(".gif"):
            mime_type = "image/gif"
        elif image.id.endswith(".webp"):
            mime_type = "image/webp"
        
        response = client.chat.completions.create(
            model=settings.agentic_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": "Describe this image concisely for document search indexing. Focus on: diagrams, charts, text content, key visual elements. Keep under 100 words."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{clean_b64}",
                                "detail": "low"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Document context: {context[:200]}" if context else "Describe this image."
                        }
                    ]
                }
            ],
            max_tokens=150,
        )
        description = response.choices[0].message.content.strip()
        return f"[IMAGE: {description}]"
    except Exception as e:
        logger.warning(f"Failed to describe image {image.id}: {e}")
        return f"[IMAGE: {image.id}]"


def _expand_image_placeholders(
    markdown: str, 
    page: PageResult, 
    describe: bool = True
) -> str:
    """Replace image placeholders with descriptions or markers."""
    # Build lookup of images by ID
    image_map = {img.id: img for img in page.images}
    
    # Pattern matches ![img-0.jpeg](img-0.jpeg) or ![alt](filename.ext)
    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    
    def replace_image(match):
        alt_text = match.group(1)
        img_id = match.group(2)
        
        if describe and img_id in image_map:
            img = image_map[img_id]
            if img.base64:
                # Get surrounding context for better description
                start = max(0, match.start() - 200)
                end = min(len(markdown), match.end() + 200)
                context = markdown[start:end]
                return describe_image(img, context)
        
        # Fallback to simple marker
        return f"[IMAGE: {alt_text or img_id}]"
    
    return img_pattern.sub(replace_image, markdown)


def get_genie(model: str = "gpt-4o-mini") -> AzureOpenAIGenie:
    """Get or create Azure OpenAI Genie instance."""
    global _genie
    if _genie is None:
        _genie = AzureOpenAIGenie(
            model=model,
            azure_api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
    return _genie


def get_chunker(config: AgenticChunkConfig) -> SlumberChunker:
    """Get or create SlumberChunker instance."""
    global _chunker
    if _chunker is None:
        genie = get_genie(config.model)
        _chunker = SlumberChunker(
            genie=genie,
            tokenizer="character",  # Character-based for legal precision
            chunk_size=config.chunk_size,
            candidate_size=config.candidate_size,
            min_characters_per_chunk=config.min_characters_per_chunk,
            verbose=config.verbose,
        )
    return _chunker


def chunk_ocr_result_agentic(
    ocr_result: OCRResult,
    filename: str,
    config: Optional[AgenticChunkConfig] = None,
) -> list[DocumentChunk]:
    """
    Chunk OCR result using agentic SlumberChunker.
    
    The LLM understands document structure and makes intelligent split decisions:
    - Respects clause/section boundaries in contracts
    - Keeps numbered items together (ballots, cost line items)
    - Preserves key-value pairs in forms
    - Maintains legal continuity across related paragraphs
    """
    if config is None:
        config = AgenticChunkConfig()

    chunker = get_chunker(config)
    chunks: list[DocumentChunk] = []
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    chunk_idx = 0

    for page in ocr_result.pages:
        page_num = page.index + 1
        markdown = page.markdown

        if not markdown or len(markdown.strip()) < config.min_characters_per_chunk:
            continue

        # Expand image placeholders with descriptions
        markdown = _expand_image_placeholders(markdown, page, describe=config.describe_images)

        # Extract tables first if configured (same as hierarchical chunker)
        table_chunks = []
        if config.preserve_tables:
            table_chunks, markdown = _extract_tables_as_chunks(
                markdown, filename, page_num, chunk_idx, now
            )
            chunks.extend(table_chunks)
            chunk_idx += len(table_chunks)

        if not markdown or len(markdown.strip()) < config.min_characters_per_chunk:
            continue

        # Use SlumberChunker for intelligent splitting
        try:
            slumber_chunks = chunker.chunk(markdown)
        except Exception as e:
            logger.warning(f"SlumberChunker failed on page {page_num}: {e}, falling back")
            # Fallback to simple splitting
            slumber_chunks = [type('Chunk', (), {'text': markdown, 'start_index': 0, 'end_index': len(markdown)})]

        for sc in slumber_chunks:
            content = sc.text.strip()
            if len(content) < config.min_characters_per_chunk:
                continue

            chunk_id = generate_chunk_id(filename, chunk_idx)
            
            # Detect hierarchy from content
            hierarchy, section_title = _extract_hierarchy_from_content(content)
            chunk_type = _detect_chunk_type(content)

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
                content=content,
                metadata=metadata,
            ))
            chunk_idx += 1

    logger.info(f"Agentic chunking: {len(chunks)} chunks from {len(ocr_result.pages)} pages")
    return chunks


def _extract_tables_as_chunks(
    markdown: str,
    filename: str,
    page_num: int,
    start_idx: int,
    timestamp: str,
) -> tuple[list[DocumentChunk], str]:
    """Extract markdown tables as separate chunks."""
    import re
    
    chunks = []
    chunk_idx = start_idx
    
    table_pattern = re.compile(
        r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n?)+)",
        re.MULTILINE
    )

    remaining = markdown
    for match in table_pattern.finditer(markdown):
        table_content = match.group(1).strip()
        if len(table_content) < 30:
            continue

        chunk_id = generate_chunk_id(filename, chunk_idx)
        
        metadata = ChunkMetadata(
            page_numbers=[page_num],
            section_title=None,
            hierarchy_path=None,
            source=filename,
            chunk_type="table",
            indexed_at=timestamp,
        )

        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            content=table_content,
            metadata=metadata,
        ))
        chunk_idx += 1
        remaining = remaining.replace(match.group(0), "\n[TABLE EXTRACTED]\n", 1)

    return chunks, remaining


def _extract_hierarchy_from_content(content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract hierarchy path from markdown headers in content."""
    import re
    
    headers = []
    for line in content.split('\n')[:10]:  # Check first 10 lines
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            headers.append(match.group(2).strip())
    
    if headers:
        # Clean HTML artifacts
        headers = [h.replace('<br>', ' ').replace('<br/>', ' ') for h in headers]
        return " > ".join(headers), headers[-1]
    return None, None


def _detect_chunk_type(content: str) -> str:
    """Detect content type from patterns."""
    import re
    
    if re.search(r"^\|.*\|$", content, re.MULTILINE):
        return "table"
    if re.search(r"^```", content, re.MULTILINE):
        return "code"
    if re.search(r"\[IMAGE:", content):
        return "image_context"
    if re.search(r"^\s*[-*]\s", content, re.MULTILINE):
        return "list"
    if re.search(r"^\s*\d+\.\s", content, re.MULTILINE):
        return "numbered_list"
    # Legal document patterns
    if re.search(r"(?i)(article|section|clause)\s+\d+", content):
        return "legal_clause"
    if re.search(r"(?i)(whereas|hereby|hereinafter)", content):
        return "legal_preamble"
    if re.search(r"\$[\d,]+\.?\d*", content):
        return "financial"
    return "text"


# Comparison function for evaluation
async def compare_chunking_strategies(
    ocr_result: OCRResult,
    filename: str,
) -> dict:
    """Compare hierarchical vs agentic chunking for analysis."""
    from app.services.hierarchical_chunker import chunk_ocr_result, ChunkConfig
    
    # Rule-based chunking
    import time
    start = time.time()
    rule_chunks = chunk_ocr_result(ocr_result, filename, ChunkConfig())
    rule_time = time.time() - start
    
    # Agentic chunking
    start = time.time()
    agentic_chunks = chunk_ocr_result_agentic(ocr_result, filename, AgenticChunkConfig())
    agentic_time = time.time() - start
    
    return {
        "rule_based": {
            "count": len(rule_chunks),
            "time_seconds": round(rule_time, 3),
            "avg_size": sum(len(c.content) for c in rule_chunks) // max(len(rule_chunks), 1),
            "types": _count_types(rule_chunks),
        },
        "agentic": {
            "count": len(agentic_chunks),
            "time_seconds": round(agentic_time, 3),
            "avg_size": sum(len(c.content) for c in agentic_chunks) // max(len(agentic_chunks), 1),
            "types": _count_types(agentic_chunks),
        },
        "samples": {
            "rule_based_first": rule_chunks[0].content[:500] if rule_chunks else None,
            "agentic_first": agentic_chunks[0].content[:500] if agentic_chunks else None,
        }
    }


def _count_types(chunks: list[DocumentChunk]) -> dict:
    types = {}
    for c in chunks:
        t = c.metadata.chunk_type or "unknown"
        types[t] = types.get(t, 0) + 1
    return types
