"""Structured output schemas with comprehensive source attribution."""
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class SourceMetadata(BaseModel):
    """Metadata about the source document and chunk location."""
    
    # Document identification
    document_id: str = Field(description="Unique identifier for the source document")
    document_title: str = Field(description="Title or filename of the source document")
    
    # Location within document
    page_numbers: List[int] = Field(
        default_factory=list,
        description="Page number(s) where this chunk appears"
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section or chapter title containing this chunk"
    )
    hierarchy_path: Optional[str] = Field(
        default=None,
        description="Full hierarchy path e.g. 'Chapter 1 > Section 1.2 > Subsection A'"
    )
    
    # Authorship
    author: Optional[str] = Field(default=None, description="Document author(s)")
    
    # Timestamps
    date_created: Optional[datetime] = Field(
        default=None,
        description="Document creation date"
    )
    date_modified: Optional[datetime] = Field(
        default=None,
        description="Document last modified date"
    )
    date_indexed: Optional[datetime] = Field(
        default=None,
        description="When the document was indexed"
    )
    
    # Additional metadata
    document_type: Optional[str] = Field(
        default=None,
        description="Type of document: report, policy, manual, etc."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags or categories assigned to the document"
    )


class ChunkAnalysis(BaseModel):
    """Analysis result from a single document analyst agent."""
    
    chunk_id: str = Field(description="Unique identifier for the analyzed chunk")
    
    # Source attribution - REQUIRED
    source: SourceMetadata = Field(description="Full source metadata for citation")
    
    relevance_score: int = Field(
        description="Relevance score from 0-10 where 0=not relevant, 10=perfectly relevant",
        ge=0, le=10
    )
    
    is_relevant: bool = Field(
        description="True if relevance_score >= 5, indicating useful information"
    )
    
    summary: str = Field(
        description="Concise 2-3 sentence summary of relevant information, or 'Not relevant' if score < 5"
    )
    
    key_points: List[str] = Field(
        default_factory=list,
        description="List of 1-5 key facts, insights, or data points extracted"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the analysis accuracy"
    )
    
    source_quotes: List[str] = Field(
        default_factory=list,
        description="Verbatim quotes that directly answer the query (max 3)"
    )


class Citation(BaseModel):
    """Formatted citation for final response."""
    
    citation_id: int = Field(description="Citation number for reference [1], [2], etc.")
    document_title: str
    author: Optional[str] = None
    page_numbers: List[int] = Field(default_factory=list)
    section_title: Optional[str] = None
    date_modified: Optional[datetime] = None
    chunk_ids: List[str] = Field(
        description="Chunk IDs that contributed from this source"
    )
    
    def format_citation(self) -> str:
        """Format as readable citation string."""
        parts = [f"[{self.citation_id}]"]
        
        if self.author:
            parts.append(f"{self.author}.")
        
        parts.append(f'"{self.document_title}"')
        
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                parts.append(f"p. {self.page_numbers[0]}")
            else:
                pages = sorted(self.page_numbers)
                parts.append(f"pp. {pages[0]}-{pages[-1]}")
        
        if self.section_title:
            parts.append(f"§ {self.section_title}")
        
        if self.date_modified:
            parts.append(f"(Modified: {self.date_modified.strftime('%Y-%m-%d')})")
        
        return " ".join(parts)


class SynthesizedResponse(BaseModel):
    """Final synthesized response with full source attribution."""
    
    answer: str = Field(
        description="Comprehensive answer with inline citation markers [1], [2], etc."
    )
    
    supporting_evidence: List[str] = Field(
        description="Key evidence points with citation markers"
    )
    
    citations: List[Citation] = Field(
        default_factory=list,
        description="Full citation list for all referenced sources"
    )
    
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence based on relevance and coverage of sources"
    )
    
    coverage_summary: str = Field(
        description="Brief assessment of how well the sources covered the query"
    )
    
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Aspects of the query not fully addressed by available sources"
    )
