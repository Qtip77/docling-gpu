#!/usr/bin/env python3
"""Analyze Mistral OCR output vs chunked result for RAG quality assessment."""
import asyncio
import json
import sys
from pathlib import Path

# Add ingestion to path
sys.path.insert(0, str(Path(__file__).parent / "ingestion"))

from ingestion.app.services.mistral_ocr import process_document, OCRResult, PageResult
from ingestion.app.services.hierarchical_chunker import chunk_ocr_result, ChunkConfig


def print_separator(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def analyze_ocr_result(ocr_result: OCRResult):
    """Analyze and display raw OCR output structure."""
    print_separator("RAW OCR ANALYSIS")
    
    print(f"\nModel: {ocr_result.model}")
    print(f"Total Pages: {len(ocr_result.pages)}")
    print(f"Usage Info: {ocr_result.usage_info}")
    
    if ocr_result.document_annotation:
        print(f"\nDocument Annotation:")
        print(json.dumps(ocr_result.document_annotation, indent=2))
    
    total_images = sum(len(p.images) for p in ocr_result.pages)
    total_tables = sum(len(p.tables) for p in ocr_result.pages)
    total_hyperlinks = sum(len(p.hyperlinks) for p in ocr_result.pages)
    total_chars = sum(len(p.markdown) for p in ocr_result.pages)
    
    print(f"\n--- DOCUMENT STATISTICS ---")
    print(f"Total Characters: {total_chars:,}")
    print(f"Total Images: {total_images}")
    print(f"Total Tables: {total_tables}")
    print(f"Total Hyperlinks: {total_hyperlinks}")
    
    for i, page in enumerate(ocr_result.pages):
        print_separator(f"PAGE {i+1}")
        print(f"Markdown Length: {len(page.markdown):,} chars")
        print(f"Images: {len(page.images)}")
        print(f"Tables: {len(page.tables)}")
        print(f"Hyperlinks: {len(page.hyperlinks)}")
        if page.header:
            print(f"Header: {page.header[:100]}...")
        if page.footer:
            print(f"Footer: {page.footer[:100]}...")
        
        print(f"\n--- RAW MARKDOWN (first 2000 chars) ---")
        print(page.markdown[:2000])
        if len(page.markdown) > 2000:
            print(f"\n... [{len(page.markdown) - 2000} more chars] ...")
        
        if page.tables:
            print(f"\n--- EXTRACTED TABLES ---")
            for tbl in page.tables:
                print(f"\nTable ID: {tbl.id}")
                print(f"Content Preview: {tbl.content[:500] if tbl.content else 'N/A'}...")
        
        if page.hyperlinks:
            print(f"\n--- HYPERLINKS ---")
            for link in page.hyperlinks[:10]:
                print(f"  - {link}")


def analyze_chunks(chunks: list, config: ChunkConfig):
    """Analyze chunked output."""
    print_separator("CHUNKING ANALYSIS")
    
    print(f"\nChunk Config:")
    print(f"  max_chunk_size: {config.max_chunk_size}")
    print(f"  min_chunk_size: {config.min_chunk_size}")
    print(f"  overlap: {config.overlap}")
    print(f"  preserve_tables: {config.preserve_tables}")
    
    print(f"\nTotal Chunks: {len(chunks)}")
    
    # Statistics
    sizes = [len(c.content) for c in chunks]
    types = {}
    pages_covered = set()
    
    for c in chunks:
        ct = c.metadata.chunk_type or "unknown"
        types[ct] = types.get(ct, 0) + 1
        pages_covered.update(c.metadata.page_numbers or [])
    
    print(f"\n--- CHUNK STATISTICS ---")
    print(f"Min Size: {min(sizes):,} chars")
    print(f"Max Size: {max(sizes):,} chars")
    print(f"Avg Size: {sum(sizes)//len(sizes):,} chars")
    print(f"Total Content: {sum(sizes):,} chars")
    print(f"Pages Covered: {sorted(pages_covered)}")
    print(f"\nChunk Types:")
    for ct, count in sorted(types.items()):
        print(f"  {ct}: {count}")
    
    # Show each chunk
    for i, chunk in enumerate(chunks):
        print_separator(f"CHUNK {i+1}/{len(chunks)}")
        print(f"ID: {chunk.chunk_id}")
        print(f"Type: {chunk.metadata.chunk_type}")
        print(f"Pages: {chunk.metadata.page_numbers}")
        print(f"Section: {chunk.metadata.section_title}")
        print(f"Hierarchy: {chunk.metadata.hierarchy_path}")
        print(f"Size: {len(chunk.content):,} chars")
        print(f"\n--- CONTENT ---")
        print(chunk.content[:1500])
        if len(chunk.content) > 1500:
            print(f"\n... [{len(chunk.content) - 1500} more chars] ...")


def identify_issues_for_complex_docs(ocr_result: OCRResult, chunks: list):
    """Identify potential issues for contracts/agreements/ballots/cost estimates."""
    print_separator("RAG QUALITY ASSESSMENT FOR COMPLEX DOCUMENTS")
    
    issues = []
    recommendations = []
    
    # Check for clause/section handling
    clause_patterns = ["article", "section", "clause", "paragraph", "§", "whereas", "hereby"]
    has_legal_structure = any(
        any(p in page.markdown.lower() for p in clause_patterns)
        for page in ocr_result.pages
    )
    
    if has_legal_structure:
        print("\n✓ Legal/Contract structure detected")
        
        # Check if clauses might be split across chunks
        for chunk in chunks:
            content_lower = chunk.content.lower()
            # Check for incomplete clauses
            if any(content_lower.startswith(p) for p in ["and ", "or ", "provided that", "subject to"]):
                issues.append(f"Chunk {chunk.chunk_id} starts with continuation word - might be mid-clause split")
    
    # Check for numbered lists (important for ballots, cost estimates)
    numbered_items = sum(
        len([line for line in page.markdown.split('\n') if line.strip() and line.strip()[0].isdigit()])
        for page in ocr_result.pages
    )
    if numbered_items > 10:
        print(f"✓ Numbered items detected: {numbered_items}")
    
    # Check table preservation
    table_chunks = [c for c in chunks if c.metadata.chunk_type == "table"]
    total_tables = sum(len(p.tables) for p in ocr_result.pages)
    if total_tables > 0:
        if len(table_chunks) >= total_tables:
            print(f"✓ Tables preserved as separate chunks: {len(table_chunks)}/{total_tables}")
        else:
            issues.append(f"Table chunks ({len(table_chunks)}) < detected tables ({total_tables})")
    
    # Check for signature blocks / dates
    signature_indicators = ["signature", "signed", "date:", "witness", "notary"]
    has_signatures = any(
        any(s in page.markdown.lower() for s in signature_indicators)
        for page in ocr_result.pages
    )
    if has_signatures:
        print("✓ Signature/date blocks detected")
    
    # Check for monetary values (cost estimates)
    money_patterns = ["$", "€", "£", "total:", "amount:", "cost:", "price:"]
    has_financials = any(
        any(p in page.markdown.lower() for p in money_patterns)
        for page in ocr_result.pages
    )
    if has_financials:
        print("✓ Financial/cost data detected")
    
    # Chunk size analysis for context
    large_chunks = [c for c in chunks if len(c.content) > 1200]
    small_chunks = [c for c in chunks if len(c.content) < 200]
    
    if small_chunks:
        issues.append(f"{len(small_chunks)} chunks under 200 chars - might lack context")
    
    # Check hierarchy preservation
    chunks_with_hierarchy = [c for c in chunks if c.metadata.hierarchy_path]
    if chunks_with_hierarchy:
        print(f"✓ Hierarchy preserved: {len(chunks_with_hierarchy)}/{len(chunks)} chunks have context path")
    else:
        recommendations.append("Document lacks header structure - consider adding section markers")
    
    # Output issues
    if issues:
        print("\n⚠️  POTENTIAL ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Output recommendations
    print("\n📋 RECOMMENDATIONS FOR CONTRACT/LEGAL RAG:")
    recommendations.extend([
        "Consider semantic chunking by clause boundaries (Article/Section/Clause)",
        "Add cross-reference linking for 'as defined in Section X' patterns",
        "Extract defined terms as separate index for entity recognition",
        "Preserve exhibit/schedule references with parent document context",
        "Consider larger overlap (200-300 chars) for legal continuity",
    ])
    for rec in recommendations:
        print(f"  → {rec}")


async def main():
    # Select a document to analyze
    uploads_dir = Path("/root/docling-gpu/uploads")
    files = list(uploads_dir.glob("*.pdf"))
    
    if not files:
        print("No PDF files found in uploads/")
        return
    
    print("Available documents:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f.name}")
    
    # Use the first document or specify via command line
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    file_path = files[idx]
    
    print(f"\n🔍 Analyzing: {file_path.name}")
    
    # Process with Mistral OCR
    print("\n⏳ Running Mistral Document AI OCR...")
    ocr_result = await process_document(str(file_path))
    
    # Analyze raw OCR
    analyze_ocr_result(ocr_result)
    
    # Chunk the result
    config = ChunkConfig()
    chunks = chunk_ocr_result(ocr_result, file_path.name, config)
    
    # Analyze chunks
    analyze_chunks(chunks, config)
    
    # RAG quality assessment
    identify_issues_for_complex_docs(ocr_result, chunks)


if __name__ == "__main__":
    asyncio.run(main())
