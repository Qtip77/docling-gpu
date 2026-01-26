"""Synthesis node with comprehensive citation generation."""
import logging
from typing import Dict, Any, List
from collections import defaultdict
from pydantic import BaseModel, Field

from ..state import OrchestratorState

logger = logging.getLogger(__name__)
from ..schemas.outputs import SynthesizedResponse, Citation, ChunkAnalysis
from ..config import get_orchestrator_llm
from ..prompts.analyst_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT_TEMPLATE
)


class SynthesisLLMResponse(BaseModel):
    """Schema for LLM synthesis - citations built separately."""
    answer: str = Field(description="Answer with inline citations [1], [2], etc.")
    supporting_evidence: List[str] = Field(description="Evidence points with citations")
    confidence: str = Field(description="high, medium, or low")
    coverage_summary: str
    gaps_identified: List[str] = Field(default_factory=list)


def build_citation_list(analyses: List[ChunkAnalysis]) -> tuple[List[Citation], Dict[str, int]]:
    """
    Build deduplicated citation list grouped by document.
    
    Returns:
        - List of Citation objects
        - Mapping of chunk_id to citation number
    """
    # Group chunks by document
    doc_chunks: Dict[str, List[ChunkAnalysis]] = defaultdict(list)
    for analysis in analyses:
        doc_key = analysis.source.document_id
        doc_chunks[doc_key].append(analysis)
    
    citations = []
    chunk_to_citation: Dict[str, int] = {}
    
    for citation_num, (doc_id, chunks) in enumerate(doc_chunks.items(), start=1):
        # Use first chunk's metadata as representative
        first = chunks[0].source
        
        # Collect all page numbers from this document's chunks
        all_pages = set()
        chunk_ids = []
        for chunk in chunks:
            all_pages.update(chunk.source.page_numbers)
            chunk_ids.append(chunk.chunk_id)
            chunk_to_citation[chunk.chunk_id] = citation_num
        
        citation = Citation(
            citation_id=citation_num,
            document_title=first.document_title,
            author=first.author,
            page_numbers=sorted(all_pages),
            section_title=first.section_title,
            date_modified=first.date_modified,
            chunk_ids=chunk_ids
        )
        citations.append(citation)
    
    return citations, chunk_to_citation


def format_analyses_with_citations(
    analyses: List[ChunkAnalysis],
    chunk_to_citation: Dict[str, int]
) -> str:
    """Format analyses with citation numbers for synthesis prompt."""
    formatted = []
    for analysis in analyses:
        cite_num = chunk_to_citation.get(analysis.chunk_id, "?")
        page_str = ", ".join(str(p) for p in analysis.source.page_numbers) if analysis.source.page_numbers else "N/A"
        
        quotes_str = "; ".join(f'"{q}"' for q in analysis.source_quotes[:2]) if analysis.source_quotes else "None"
        
        formatted.append(
            f"**[{cite_num}] {analysis.source.document_title}** "
            f"(p. {page_str}, Relevance: {analysis.relevance_score}/10)\n"
            f"Summary: {analysis.summary}\n"
            f"Key Points: {'; '.join(analysis.key_points)}\n"
            f"Quotes: {quotes_str}"
        )
    
    return "\n\n".join(formatted)


def format_source_list(citations: List[Citation]) -> str:
    """Format source list for the synthesis prompt."""
    lines = []
    for c in citations:
        line = f"[{c.citation_id}] {c.document_title}"
        if c.author:
            line += f" by {c.author}"
        if len(c.page_numbers) > 1:
            line += f" (pp. {c.page_numbers[0]}-{c.page_numbers[-1]})"
        elif c.page_numbers:
            line += f" (p. {c.page_numbers[0]})"
        lines.append(line)
    return "\n".join(lines)


async def synthesize_response(state: OrchestratorState) -> Dict[str, Any]:
    """
    Synthesize final response with full citations.
    
    Groups sources by document, generates citation list,
    and produces a well-cited comprehensive answer.
    """
    analyses = state["analyses"]
    query = state["query"]
    
    # Filter and sort by relevance
    relevant = [a for a in analyses if a.is_relevant]
    relevant_sorted = sorted(relevant, key=lambda x: x.relevance_score, reverse=True)
    
    # Early return if no relevant chunks
    if not relevant_sorted:
        return {
            "final_response": "I couldn't find relevant information to answer your query in the available documents.",
            "relevant_count": 0,
            "sources_used": []
        }
    
    # Build citation list from relevant analyses
    citations, chunk_to_citation = build_citation_list(relevant_sorted)
    
    # Format for synthesis
    top_analyses = relevant_sorted[:15]  # Top 15 most relevant
    analyses_formatted = format_analyses_with_citations(top_analyses, chunk_to_citation)
    source_list = format_source_list(citations)
    
    llm = get_orchestrator_llm()
    structured_llm = llm.with_structured_output(SynthesisLLMResponse)
    
    user_prompt = SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        query=query,
        analyses_formatted=analyses_formatted,
        source_list=source_list
    )
    
    try:
        synthesis = await structured_llm.ainvoke([
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        
        # Format citations
        citations_formatted = "\n".join([c.format_citation() for c in citations])
        
        # Build final response with all sections
        final_response = f"""## Answer

{synthesis.answer}

---

### Supporting Evidence
{chr(10).join(f'• {e}' for e in synthesis.supporting_evidence)}

---

### Sources

{citations_formatted}

---

### Assessment
**Confidence:** {synthesis.confidence}

**Coverage:** {synthesis.coverage_summary}
"""
        if synthesis.gaps_identified:
            final_response += f"\n**Gaps identified:** {', '.join(synthesis.gaps_identified)}"
        
    except Exception as e:
        # Log full error details for debugging Azure OpenAI issues
        logger.error(
            f"Synthesizer LLM call failed: {type(e).__name__}: {e}",
            exc_info=True
        )
        # Check for Azure OpenAI specific error details
        if hasattr(e, 'response'):
            try:
                error_body = e.response.json() if hasattr(e.response, 'json') else str(e.response.text)
                logger.error(f"Azure OpenAI error response: {error_body}")
            except Exception:
                logger.error(f"Azure OpenAI response status: {getattr(e.response, 'status_code', 'unknown')}")
        
        final_response = f"Synthesis error: {str(e)}\n\nPartial findings available in shared document."
        citations = []
    
    return {
        "final_response": final_response,
        "relevant_count": len(relevant_sorted),
        "sources_used": [c.document_title for c in citations]
    }
