"""Retrieval node for Azure AI Search."""
from typing import Dict, Any
from ..state import OrchestratorState
from ..tools.search import retrieve_chunks


async def retrieve_documents(state: OrchestratorState) -> Dict[str, Any]:
    """
    Retrieve relevant chunks from Azure AI Search.
    
    Supports optional metadata filters passed in state.
    """
    query = state["query"]
    filters = state.get("filters")
    
    chunks = await retrieve_chunks(
        query=query,
        top_k=20,  # Retrieve more, filter after analysis
        filters=filters
    )
    
    return {
        "retrieved_chunks": chunks,
        "retrieval_count": len(chunks)
    }
