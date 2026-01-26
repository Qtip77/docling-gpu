"""Distribution logic using Send() API for dynamic parallelism."""
from typing import List, Union
from langgraph.types import Send
from ..state import OrchestratorState
from ..tools.search import extract_source_metadata


def distribute_to_analysts(state: OrchestratorState) -> Union[List[Send], str]:
    """
    Distribute retrieved chunks to parallel analyst agents.
    
    Uses LangGraph's Send() API to dynamically spawn one analyst
    per chunk. The number of analysts scales with retrieval results.
    
    Extracts and passes metadata with each chunk for source attribution.
    
    IMPORTANT: If no chunks are retrieved, returns "synthesize" directly
    to avoid hanging the graph (empty Send() list causes LangGraph to wait
    indefinitely for nodes that will never execute).
    """
    # Handle empty retrieval - skip analysis and go directly to synthesis
    if not state["retrieved_chunks"]:
        return "synthesize"
    
    sends = []
    
    for i, chunk in enumerate(state["retrieved_chunks"]):
        chunk_id = chunk.metadata.get("chunk_id", f"chunk_{i}")
        
        # Pre-extract metadata to pass to analyst
        source_metadata = extract_source_metadata(chunk, chunk_id)
        
        sends.append(
            Send("analyze_chunk", {
                "query": state["query"],
                "chunk": chunk,
                "chunk_id": chunk_id,
                "chunk_metadata": source_metadata.model_dump(),  # Serializable dict
                "analyses": [],
                "shared_document": ""
            })
        )
    
    return sends
