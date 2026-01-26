"""Document analyst agent with comprehensive source attribution."""
import logging
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from ..state import AnalystState
from ..schemas.outputs import ChunkAnalysis, SourceMetadata
from ..config import get_analyst_llm
from ..prompts.analyst_prompts import (
    DOCUMENT_ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_PROMPT_TEMPLATE,
    SHARED_DOCUMENT_CONTRIBUTION_TEMPLATE
)

logger = logging.getLogger(__name__)


class ChunkAnalysisLLM(BaseModel):
    """Simplified schema for LLM - source metadata injected after."""
    relevance_score: int = Field(ge=0, le=10)
    is_relevant: bool
    summary: str
    key_points: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    source_quotes: List[str] = Field(default_factory=list)


async def analyze_chunk(state: AnalystState) -> Dict[str, Any]:
    """
    Individual analyst agent processes a single chunk.
    
    Uses Azure OpenAI GPT-4o-mini with structured output for reliable responses.
    Preserves full source metadata for citation in final response.
    """
    llm = get_analyst_llm()
    
    # Reconstruct SourceMetadata from passed dict
    source_metadata = SourceMetadata(**state["chunk_metadata"])
    
    # Format page numbers for display
    page_display = ", ".join(str(p) for p in source_metadata.page_numbers) if source_metadata.page_numbers else "Unknown"
    
    # Format the user prompt with source context
    user_prompt = ANALYST_USER_PROMPT_TEMPLATE.format(
        query=state["query"],
        chunk_id=state["chunk_id"],
        document_title=source_metadata.document_title,
        page_numbers=page_display,
        section_title=source_metadata.section_title or "N/A",
        author=source_metadata.author or "Unknown",
        chunk_content=state["chunk"].page_content
    )
    
    structured_llm = llm.with_structured_output(ChunkAnalysisLLM)
    
    try:
        llm_analysis = await structured_llm.ainvoke([
            {"role": "system", "content": DOCUMENT_ANALYST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        
        # Combine LLM analysis with source metadata
        analysis = ChunkAnalysis(
            chunk_id=state["chunk_id"],
            source=source_metadata,
            relevance_score=llm_analysis.relevance_score,
            is_relevant=llm_analysis.is_relevant,
            summary=llm_analysis.summary,
            key_points=llm_analysis.key_points,
            confidence=llm_analysis.confidence,
            source_quotes=llm_analysis.source_quotes
        )
        
    except Exception as e:
        # Log full error details for debugging Azure OpenAI issues
        logger.error(
            f"Analyst LLM call failed for chunk {state['chunk_id']}: {type(e).__name__}: {e}",
            exc_info=True
        )
        # Check for Azure OpenAI specific error details
        if hasattr(e, 'response'):
            try:
                error_body = e.response.json() if hasattr(e.response, 'json') else str(e.response.text)
                logger.error(f"Azure OpenAI error response: {error_body}")
            except Exception:
                logger.error(f"Azure OpenAI response status: {getattr(e.response, 'status_code', 'unknown')}")
        
        analysis = ChunkAnalysis(
            chunk_id=state["chunk_id"],
            source=source_metadata,
            relevance_score=0,
            is_relevant=False,
            summary=f"Analysis failed: {str(e)}",
            key_points=[],
            confidence="low",
            source_quotes=[]
        )
    
    # Format contribution to shared document with source info
    key_points_formatted = "\n".join(f"- {kp}" for kp in analysis.key_points) or "- None"
    page_str = ", ".join(str(p) for p in source_metadata.page_numbers) if source_metadata.page_numbers else "N/A"
    
    document_contribution = ""
    if analysis.is_relevant:
        document_contribution = SHARED_DOCUMENT_CONTRIBUTION_TEMPLATE.format(
            chunk_id=analysis.chunk_id,
            document_title=source_metadata.document_title,
            page_numbers=page_str,
            author=source_metadata.author or "Unknown",
            section_title=source_metadata.section_title or "N/A",
            relevance_score=analysis.relevance_score,
            confidence=analysis.confidence,
            summary=analysis.summary,
            key_points_formatted=key_points_formatted
        )
    
    return {
        "analyses": [analysis],
        "shared_document": document_contribution
    }
