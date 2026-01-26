from fastapi import APIRouter

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services import azure_search, azure_openai

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Vector search over indexed documents."""
    # Generate embedding for query
    query_embedding = azure_openai.embed_text(request.query)
    
    # Search
    results = azure_search.search_chunks(
        query=request.query,
        embedding=query_embedding,
        top_k=request.top_k
    )
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results]
    )
