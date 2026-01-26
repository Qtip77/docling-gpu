"""State schemas with reducer functions for parallel agent accumulation."""
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from operator import add
from langchain_core.documents import Document


def concat_markdown(left: str, right: str) -> str:
    """Reducer for accumulating markdown content from multiple agents."""
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n\n---\n\n{right}"


class OrchestratorState(TypedDict):
    """Main orchestrator state - shared across entire graph."""
    # Input
    query: str
    filters: Optional[Dict[str, Any]]  # Optional metadata filters for retrieval
    
    # Retrieved chunks from Azure AI Search
    retrieved_chunks: List[Document]
    
    # Accumulated from parallel analyst agents via reducer
    # Type hint uses Any to avoid circular import; actual type is List[ChunkAnalysis]
    analyses: Annotated[List[Any], add]
    
    # Shared collaborative document - all agents write here
    shared_document: Annotated[str, concat_markdown]
    
    # Final output
    final_response: str
    
    # Metadata
    retrieval_count: int
    relevant_count: int
    sources_used: List[str]  # Document titles used in response


class AnalystState(TypedDict):
    """State passed to individual analyst agents via Send()."""
    # Input for this analyst
    query: str
    chunk: Document
    chunk_id: str
    chunk_metadata: Dict[str, Any]  # Extracted metadata for this chunk
    
    # Output - written back to orchestrator via reducer
    analyses: Annotated[List[Any], add]
    shared_document: Annotated[str, concat_markdown]
