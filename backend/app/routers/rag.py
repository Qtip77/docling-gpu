"""Router for multi-agent RAG queries using LangGraph."""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag_agent import run_rag_query

router = APIRouter(prefix="/rag", tags=["Multi-Agent RAG"])


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The question to answer using document retrieval")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for document retrieval"
    )


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    final_response: str
    retrieval_count: int
    relevant_count: int
    sources_used: list[str]
    shared_document: str = Field(
        default="",
        description="Collaborative document with analyst contributions"
    )


@router.post("/query", response_model=RAGQueryResponse)
async def query_documents(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Execute a multi-agent RAG query.
    
    This endpoint:
    1. Retrieves relevant document chunks from Azure AI Search
    2. Distributes chunks to parallel analyst agents for evaluation
    3. Synthesizes findings into a comprehensive, cited response
    
    Args:
        request: The query request with optional filters
        
    Returns:
        RAGQueryResponse with the synthesized answer and source citations
    """
    try:
        result = await run_rag_query(
            query=request.query,
            filters=request.filters
        )
        
        return RAGQueryResponse(
            query=request.query,
            final_response=result.get("final_response", ""),
            retrieval_count=result.get("retrieval_count", 0),
            relevant_count=result.get("relevant_count", 0),
            sources_used=result.get("sources_used", []),
            shared_document=result.get("shared_document", "")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@router.get("/health")
async def rag_health():
    """Health check for the multi-agent RAG system."""
    return {
        "status": "healthy",
        "service": "multi-agent-rag",
        "description": "LangGraph-based document analysis system"
    }
