"""Main LangGraph definition with dynamic multi-agent orchestration."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from .state import OrchestratorState
from .nodes.retriever import retrieve_documents
from .nodes.distributor import distribute_to_analysts
from .nodes.analyst import analyze_chunk
from .nodes.synthesizer import synthesize_response


def build_rag_graph():
    """
    Build the multi-agent RAG graph.
    
    Flow: START → retrieve → [distribute via Send()] → analyze_chunk (parallel) → synthesize → END
    
    The distribute_to_analysts function returns a list of Send() objects,
    one per retrieved chunk. LangGraph executes all these in parallel within
    a single superstep, then collects results via the reducer functions
    defined in OrchestratorState.
    """
    builder = StateGraph(OrchestratorState)
    
    # Configure retry policy for analyst nodes (handles transient Azure OpenAI errors)
    analyst_retry = RetryPolicy(
        initial_interval=0.5,
        backoff_factor=2.0,
        max_interval=10.0,
        max_attempts=3,
        jitter=True
    )
    
    # Add nodes
    builder.add_node("retrieve", retrieve_documents)
    builder.add_node(
        "analyze_chunk", 
        analyze_chunk,
        retry=analyst_retry  # Auto-retry on failures
    )
    builder.add_node("synthesize", synthesize_response)
    
    # Add edges
    builder.add_edge(START, "retrieve")
    
    # Conditional edge with Send() for dynamic parallelism
    # distribute_to_analysts returns List[Send] objects OR "synthesize" string
    # when no chunks are retrieved (to avoid hanging on empty Send list)
    builder.add_conditional_edges(
        "retrieve",
        distribute_to_analysts,
        ["analyze_chunk", "synthesize"]  # Possible destinations
    )
    
    # All analysts fan-in to synthesize
    builder.add_edge("analyze_chunk", "synthesize")
    builder.add_edge("synthesize", END)
    
    return builder.compile()


# Compiled graph - this is what langgraph.json references
graph = build_rag_graph()


def get_initial_state(query: str, filters: dict = None) -> dict:
    """Build initial state for RAG query."""
    return {
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


# Convenience function for direct invocation
async def run_rag_query(
    query: str,
    filters: dict = None
) -> dict:
    """Run a RAG query through the multi-agent system."""
    initial_state = get_initial_state(query, filters)
    result = await graph.ainvoke(initial_state)
    return result


async def stream_rag_query(
    query: str,
    filters: dict = None
):
    """
    Stream a RAG query through the multi-agent system.
    
    Yields events from the graph execution for real-time progress tracking.
    Use this with astream_events for SSE streaming.
    """
    initial_state = get_initial_state(query, filters)
    
    async for event in graph.astream_events(initial_state, version="v2"):
        yield event
