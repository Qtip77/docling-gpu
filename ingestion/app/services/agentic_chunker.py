"""Agentic chunker using Chonkie's SlumberChunker for intelligent document splitting.

Best for: contracts, legal agreements, vote ballots, cost estimates
Trade-off: ~2-5s per page (LLM calls) vs ~10ms for rule-based chunking
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Literal

from chonkie import SlumberChunker
from chonkie.genie import AzureOpenAIGenie

from app.config import settings
from app.models.schemas import ChunkMetadata, DocumentChunk
from app.services.mistral_ocr import OCRResult, generate_chunk_id

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


_genie: Optional[AzureOpenAIGenie] = None
_chunker: Optional[SlumberChunker] = None


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
