import logging
import re
import hashlib
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.chunking import HierarchicalChunker

from app.config import settings

logger = logging.getLogger(__name__)


# Quality check thresholds
@dataclass
class QualityThresholds:
    """Configurable thresholds for OCR quality detection."""
    min_confidence: float = 0.5  # Minimum average confidence score
    min_chars_per_page: int = 100  # Minimum characters per page
    garbage_char_ratio: float = 0.15  # Maximum ratio of garbage characters
    min_chunks_per_page: float = 0.5  # Minimum chunks per page
    min_total_chars: int = 50  # Minimum total characters


# Common OCR garbage/artifact characters
# Removed common bullets and list markers to prevent false positives
GARBAGE_CHARS = set('□■◊◦▯▮▭▬▩▨▧▦▥▤▣▢¤¥¦§¨©ª«¬®¯°±²³´µ¶¸¹º»¼½¾¿')


def check_confidence_scores(result, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Check average confidence scores from Docling result.
    Returns (is_acceptable, average_confidence).
    """
    confidences = []
    
    # Check text elements for confidence scores
    if hasattr(result.document, 'texts'):
        for item in result.document.texts:
            if hasattr(item, 'confidence') and item.confidence is not None:
                confidences.append(item.confidence)
    
    # Check table elements
    if hasattr(result.document, 'tables'):
        for table in result.document.tables:
            if hasattr(table, 'confidence') and table.confidence is not None:
                confidences.append(table.confidence)
    
    if not confidences:
        # No confidence data available, assume acceptable
        return True, 1.0
    
    avg_confidence = sum(confidences) / len(confidences)
    return avg_confidence >= threshold, avg_confidence


def check_text_density(text: str, page_count: int, min_chars_per_page: int = 100) -> tuple[bool, float]:
    """
    Check if text density is acceptable (enough characters per page).
    Returns (is_acceptable, chars_per_page).
    """
    if page_count <= 0:
        page_count = 1
    
    chars_per_page = len(text) / page_count
    return chars_per_page >= min_chars_per_page, chars_per_page


def check_garbage_characters(text: str, threshold: float = 0.15) -> tuple[bool, float]:
    """
    Check for high ratio of garbage/artifact characters indicating OCR failure.
    Returns (is_acceptable, garbage_ratio).
    """
    if not text:
        return True, 0.0
    
    garbage_count = sum(1 for c in text if c in GARBAGE_CHARS)
    garbage_ratio = garbage_count / len(text)
    return garbage_ratio <= threshold, garbage_ratio


def check_chunk_count(chunks: list, page_count: int, min_chunks_per_page: float = 0.5) -> tuple[bool, float]:
    """
    Check if enough chunks were generated relative to page count.
    Returns (is_acceptable, chunks_per_page).
    """
    if page_count <= 0:
        page_count = 1
    
    chunks_per_page = len(chunks) / page_count
    return chunks_per_page >= min_chunks_per_page, chunks_per_page


@dataclass
class QualityReport:
    """Report of quality check results."""
    is_acceptable: bool
    confidence_ok: bool
    avg_confidence: float
    density_ok: bool
    chars_per_page: float
    garbage_ok: bool
    garbage_ratio: float
    chunks_ok: bool
    chunks_per_page: float
    reasons: list[str]
    
    def __str__(self) -> str:
        status = "PASS" if self.is_acceptable else "FAIL"
        return (
            f"Quality Check [{status}]: "
            f"confidence={self.avg_confidence:.2f}, "
            f"chars/page={self.chars_per_page:.1f}, "
            f"garbage={self.garbage_ratio:.2%}, "
            f"chunks/page={self.chunks_per_page:.2f}"
            + (f" | Issues: {', '.join(self.reasons)}" if self.reasons else "")
        )


def assess_parse_quality(
    result,
    chunks: list[tuple[str, str, dict]],
    thresholds: Optional[QualityThresholds] = None
) -> QualityReport:
    """
    Assess the quality of parsed document using multiple heuristics.
    Returns a QualityReport with detailed results.
    """
    if thresholds is None:
        thresholds = QualityThresholds()
    
    # Get page count from document
    page_count = 1
    if hasattr(result.document, 'num_pages'):
        num_pages = result.document.num_pages
        # num_pages might be a method or a property, handle both cases
        if callable(num_pages):
            page_count = num_pages() or 1
        else:
            page_count = num_pages or 1
    elif hasattr(result.document, 'pages'):
        page_count = len(result.document.pages) or 1
    
    # Combine all chunk text for analysis
    full_text = " ".join(chunk[1] for chunk in chunks)
    
    # Run all quality checks
    confidence_ok, avg_confidence = check_confidence_scores(result, thresholds.min_confidence)
    density_ok, chars_per_page = check_text_density(full_text, page_count, thresholds.min_chars_per_page)
    garbage_ok, garbage_ratio = check_garbage_characters(full_text, thresholds.garbage_char_ratio)
    chunks_ok, chunks_per_page = check_chunk_count(chunks, page_count, thresholds.min_chunks_per_page)
    
    # Collect failure reasons
    reasons = []
    if not confidence_ok:
        reasons.append(f"low confidence ({avg_confidence:.2f})")
    if not density_ok:
        reasons.append(f"low text density ({chars_per_page:.1f} chars/page)")
    if not garbage_ok:
        reasons.append(f"high garbage ratio ({garbage_ratio:.2%})")
    if not chunks_ok:
        reasons.append(f"too few chunks ({chunks_per_page:.2f}/page)")
    
    # Also check for minimum total content
    if len(full_text.strip()) < thresholds.min_total_chars:
        reasons.append(f"insufficient content ({len(full_text)} chars)")
    
    # Overall assessment - fail if any check fails
    is_acceptable = confidence_ok and density_ok and garbage_ok and chunks_ok and len(full_text.strip()) >= thresholds.min_total_chars
    
    return QualityReport(
        is_acceptable=is_acceptable,
        confidence_ok=confidence_ok,
        avg_confidence=avg_confidence,
        density_ok=density_ok,
        chars_per_page=chars_per_page,
        garbage_ok=garbage_ok,
        garbage_ratio=garbage_ratio,
        chunks_ok=chunks_ok,
        chunks_per_page=chunks_per_page,
        reasons=reasons,
    )


def azure_openai_vlm_options(
    model: str,
    endpoint: str,
    api_key: str,
    api_version: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    prompt: str = "OCR this page and convert it to markdown. Preserve the document structure including headers, lists, tables, and formatting.",
) -> ApiVlmOptions:
    """
    Configure ApiVlmOptions for Azure OpenAI vision-capable models.
    
    Azure OpenAI uses a different endpoint format than standard OpenAI.
    The chat completions endpoint is: {endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}
    """
    # Azure OpenAI endpoint format
    url = f"{endpoint.rstrip('/')}/openai/deployments/{model}/chat/completions?api-version={api_version}"
    
    options = ApiVlmOptions(
        url=url,
        params=dict(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
        },
        prompt=prompt,
        timeout=120,  # Longer timeout for Azure OpenAI
        scale=2.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options


def get_vlm_converter() -> DocumentConverter:
    """
    Create a DocumentConverter using the VLM pipeline with Azure OpenAI.
    
    This uses vision-capable models (like GPT-4o) to process PDF pages
    as images, which can be more accurate for complex layouts, handwritten
    content, and documents with mixed content types.
    """
    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # Required for calling Azure OpenAI
    )
    
    # Configure Azure OpenAI VLM options
    pipeline_options.vlm_options = azure_openai_vlm_options(
        model=settings.azure_openai_vlm_model,
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        max_tokens=settings.azure_openai_vlm_max_tokens,
        temperature=settings.azure_openai_vlm_temperature,
    )
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )


def get_converter() -> DocumentConverter:
    """Create a GPU-accelerated DocumentConverter with OCR and table extraction."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # For handwritten notes
    pipeline_options.do_table_structure = True  # For tables
    pipeline_options.generate_picture_images = True  # For images

    # Check for GPU availability
    use_gpu = torch.cuda.is_available()
    device = AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
    if not use_gpu:
        logger.info("CUDA not available, falling back to CPU for OCR and processing")
    
    # OCR config for handwritten content with GPU acceleration
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["en"],
        confidence_threshold=0.5,
        use_gpu=use_gpu  # Enable GPU for EasyOCR if available
    )
    
    # Accurate table extraction
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # GPU acceleration with CUDA
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=12,
        device=device  # Use detected device
    )
    

    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def parse_document(
    file_path: str,
    use_vlm: Optional[bool] = None,
    auto_fallback: bool = True,
    quality_thresholds: Optional[QualityThresholds] = None,
) -> list[tuple[str, str, dict]]:
    """
    Parse a document and return chunked text with metadata.
    Returns list of (chunk_id, chunk_text, metadata) tuples.
    
    Args:
        file_path: Path to the document to parse.
        use_vlm: If True, use VLM pipeline with Azure OpenAI for PDF processing.
                 If False, use standard OCR pipeline.
                 If None, uses the USE_VLM_PIPELINE setting from config.
        auto_fallback: If True and standard pipeline produces poor quality results,
                       automatically retry with VLM pipeline. Only applies to PDFs.
        quality_thresholds: Custom thresholds for quality assessment. Uses defaults if None.
    """
    # Determine whether to use VLM pipeline
    should_use_vlm = use_vlm if use_vlm is not None else settings.use_vlm_pipeline
    
    # Only use VLM pipeline for PDFs
    is_pdf = Path(file_path).suffix.lower() == ".pdf"
    
    if should_use_vlm and is_pdf:
        converter = get_vlm_converter()
        result = converter.convert(file_path)
        chunks = _extract_chunks(result, file_path)
        logger.info(f"Parsed '{file_path}' using VLM pipeline: {len(chunks)} chunks")
        return chunks
    
    # Use standard OCR pipeline
    converter = get_converter()
    result = converter.convert(file_path)
    chunks = _extract_chunks(result, file_path)
    
    # Quality assessment and auto-fallback for PDFs
    if is_pdf and auto_fallback:
        quality_report = assess_parse_quality(result, chunks, quality_thresholds)
        logger.info(f"Standard pipeline for '{file_path}': {quality_report}")
        
        if not quality_report.is_acceptable:
            logger.warning(
                f"Poor quality detected for '{file_path}', falling back to VLM pipeline. "
                f"Reasons: {', '.join(quality_report.reasons)}"
            )
            try:
                vlm_converter = get_vlm_converter()
                vlm_result = vlm_converter.convert(file_path)
                vlm_chunks = _extract_chunks(vlm_result, file_path)
                
                # Check if VLM produced better results
                vlm_quality = assess_parse_quality(vlm_result, vlm_chunks, quality_thresholds)
                logger.info(f"VLM pipeline for '{file_path}': {vlm_quality}")
                
                # Use VLM results if they're better or at least acceptable
                if vlm_quality.is_acceptable or len(vlm_chunks) > len(chunks):
                    logger.info(f"Using VLM results for '{file_path}': {len(vlm_chunks)} chunks")
                    return vlm_chunks
                else:
                    logger.warning(
                        f"VLM pipeline also produced poor quality for '{file_path}'. "
                        f"Using original results."
                    )
            except Exception as e:
                logger.error(f"VLM fallback failed for '{file_path}': {e}. Using original results.")
    
    logger.info(f"Parsed '{file_path}' using standard pipeline: {len(chunks)} chunks")
    return chunks


def _extract_chunks(result, file_path: str) -> list[tuple[str, str, dict]]:
    """Extract and format chunks from a conversion result with metadata."""
    chunker = HierarchicalChunker()
    doc_chunks = list(chunker.chunk(result.document))
    
    # Generate unique chunk IDs based on filename
    # Azure Search keys only allow: letters, digits, underscore, dash, equal sign
    # Use MD5 hash to ensure uniqueness and valid characters
    filename_hash = hashlib.md5(Path(file_path).name.encode("utf-8")).hexdigest()
    
    # Extract document-level metadata (if available)
    doc_created = None
    if hasattr(result.document, 'origin') and result.document.origin:
        if hasattr(result.document.origin, 'created_at'):
            doc_created = result.document.origin.created_at
    
    chunks = []
    for idx, chunk in enumerate(doc_chunks):
        chunk_id = f"{filename_hash}_chunk_{idx}"
        
        # Extract page numbers from provenance in meta.doc_items (Docling v2 structure)
        # Page numbers are nested: chunk.meta.doc_items[].prov[].page_no
        pages = set()
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'page_no') and prov.page_no is not None:
                            pages.add(prov.page_no)
        pages = sorted(list(pages))
        
        # Extract section context
        headings = []
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings'):
            headings = chunk.meta.headings or []
        section_title = headings[-1] if headings else None
        hierarchy_path = " > ".join(headings) if headings else None
        
        # Extract chunk type from first doc_item if available
        chunk_type = "text"
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
            first_item = chunk.meta.doc_items[0]
            if hasattr(first_item, 'label') and first_item.label:
                chunk_type = str(first_item.label)
        
        meta = {
            "page_numbers": pages,
            "section_title": section_title,
            "hierarchy_path": hierarchy_path,
            "source": Path(file_path).name,
            "chunk_type": chunk_type,
            "created_date": doc_created.isoformat() if doc_created else None,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        }
        
        chunks.append((chunk_id, chunk.text, meta))
    
    return chunks
