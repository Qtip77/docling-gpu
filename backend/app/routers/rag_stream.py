"""SSE streaming endpoint for real-time agent state updates."""
import asyncio
import uuid
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag_agent.callbacks import (
    StreamingCallbackHandler,
    get_callback_handler,
    remove_callback_handler,
    StepName
)
from app.rag_agent.graph import graph
from app.rag_agent.schemas.outputs import ChunkAnalysis


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag-stream"])


class StreamQueryRequest(BaseModel):
    """Request body for streaming RAG query."""
    query: str = Field(..., description="The query to process through the RAG pipeline")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for retrieval"
    )


async def event_generator(
    session_id: str,
    query: str,
    filters: Optional[Dict[str, Any]],
    request: Request
):
    """
    Generate SSE events for the RAG pipeline execution.
    
    This function runs the RAG graph and emits progress events through
    the callback handler's event queue.
    """
    handler = get_callback_handler(session_id)
    
    try:
        # Emit pipeline started
        await handler.on_pipeline_start(query)
        yield handler.event_queue.get_nowait().to_sse() if not handler.event_queue.empty() else ""
        
        # Emit retrieve step started
        await handler.on_step_start(StepName.RETRIEVE)
        
        # Drain the queue after each emit
        while not handler.event_queue.empty():
            event = handler.event_queue.get_nowait()
            if event:
                yield event.to_sse()
        
        # Build initial state
        initial_state = {
            "query": query,
            "filters": filters,
            "retrieved_chunks": [],
            "analyses": [],
            "shared_document": "",
            "final_response": "",
            "retrieval_count": 0,
            "relevant_count": 0,
            "sources_used": []
        }
        
        # Stream through the graph execution
        # We use astream_events to get granular progress
        current_step = StepName.RETRIEVE
        analysts_tracked: set = set()
        
        async for event in graph.astream_events(initial_state, version="v2"):
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"Client disconnected for session {session_id}")
                break
            
            event_type = event.get("event")
            event_name = event.get("name", "")
            
            # Track node starts
            if event_type == "on_chain_start":
                if event_name == "retrieve":
                    # Already emitted above
                    pass
                elif event_name == "analyze_chunk":
                    # Track analyst start - happens in parallel
                    run_id = event.get("run_id", "")
                    if run_id and run_id not in analysts_tracked:
                        analysts_tracked.add(run_id)
                elif event_name == "synthesize":
                    await handler.on_step_start(StepName.SYNTHESIZE)
            
            # Track node completions
            elif event_type == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                
                if event_name == "retrieve":
                    # Retrieve completed
                    retrieved_chunks = output.get("retrieved_chunks", [])
                    retrieval_count = len(retrieved_chunks)
                    
                    await handler.on_step_complete(
                        StepName.RETRIEVE,
                        {"retrieval_count": retrieval_count}
                    )
                    
                    # Emit distribute step
                    await handler.on_step_start(StepName.DISTRIBUTE)
                    
                    # Get chunk IDs for spawning
                    chunk_ids = [f"chunk_{i}" for i in range(retrieval_count)]
                    await handler.on_analysts_spawned(retrieval_count, chunk_ids)
                    
                    await handler.on_step_complete(StepName.DISTRIBUTE)
                    
                    # Start analyze phase
                    if retrieval_count > 0:
                        await handler.on_step_start(
                            StepName.ANALYZE,
                            {"total_analysts": retrieval_count}
                        )
                    current_step = StepName.ANALYZE
                
                elif event_name == "analyze_chunk":
                    # Individual analyst completed
                    analyses = output.get("analyses", [])
                    if analyses:
                        for analysis in analyses:
                            # Handle both dict and ChunkAnalysis objects
                            if isinstance(analysis, dict):
                                await handler.on_analyst_complete(
                                    analyst_id=analysis.get("chunk_id", "unknown"),
                                    chunk_id=analysis.get("chunk_id", "unknown"),
                                    relevance_score=analysis.get("relevance_score", 0),
                                    is_relevant=analysis.get("is_relevant", False),
                                    summary=analysis.get("summary", ""),
                                    confidence=analysis.get("confidence", "low")
                                )
                            else:
                                await handler.on_analyst_complete(
                                    analyst_id=analysis.chunk_id,
                                    chunk_id=analysis.chunk_id,
                                    relevance_score=analysis.relevance_score,
                                    is_relevant=analysis.is_relevant,
                                    summary=analysis.summary,
                                    confidence=analysis.confidence
                                )
                
                elif event_name == "synthesize":
                    # Synthesis completed - emit final response
                    await handler.on_step_complete(StepName.SYNTHESIZE)
                    
                    await handler.on_final_response(
                        final_response=output.get("final_response", ""),
                        retrieval_count=output.get("retrieval_count", 0),
                        relevant_count=output.get("relevant_count", 0),
                        sources_used=output.get("sources_used", []),
                        shared_document=output.get("shared_document", "")
                    )
            
            # Drain event queue after each graph event
            while not handler.event_queue.empty():
                sse_event = handler.event_queue.get_nowait()
                if sse_event:
                    yield sse_event.to_sse()
        
        # Close handler when done
        await handler.close()
        
    except Exception as e:
        logger.error(f"Error in streaming RAG query: {e}")
        await handler.on_error(str(e), current_step.value if 'current_step' in dir() else None)
        
        # Emit error event
        while not handler.event_queue.empty():
            event = handler.event_queue.get_nowait()
            if event:
                yield event.to_sse()
        
        await handler.close()
    
    finally:
        # Cleanup
        remove_callback_handler(session_id)


@router.post("/stream")
async def stream_rag_query(request: Request, body: StreamQueryRequest):
    """
    Stream RAG pipeline execution with real-time progress events.
    
    This endpoint uses Server-Sent Events (SSE) to stream progress updates
    as the multi-agent RAG pipeline processes the query.
    
    Event Types:
    - pipeline_started: Query processing has begun
    - step_started: A pipeline step (retrieve/distribute/analyze/synthesize) has started
    - step_completed: A pipeline step has finished
    - analyst_spawned: Analyst agents have been created for parallel chunk analysis
    - analyst_completed: An individual analyst has finished analyzing a chunk
    - synthesis_started: Final synthesis step has begun
    - final_response: The final response is ready
    - error: An error occurred during processing
    
    Returns:
        StreamingResponse: SSE stream of AgentEvent objects
    """
    session_id = str(uuid.uuid4())
    
    return StreamingResponse(
        event_generator(session_id, body.query, body.filters, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/stream/health")
async def stream_health():
    """Health check for the streaming endpoint."""
    return {"status": "healthy", "streaming": True}
