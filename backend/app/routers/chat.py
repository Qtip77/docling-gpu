from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse, SearchResult
from app.services import azure_search, azure_openai

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """RAG chat - retrieves relevant chunks and generates response."""
    # Generate embedding for query
    query_embedding = azure_openai.embed_text(request.message)
    
    # Retrieve relevant chunks
    results = azure_search.search_chunks(
        query=request.message,
        embedding=query_embedding,
        top_k=request.top_k
    )
    
    # Extract content for context
    context_chunks = [r["content"] for r in results]
    
    # Generate RAG response
    answer = azure_openai.generate_rag_response(request.message, context_chunks)
    
    return ChatResponse(
        answer=answer,
        sources=[SearchResult(**r) for r in results]
    )
