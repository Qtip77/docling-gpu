"""LangGraph callback handler for streaming agent state updates via SSE."""
import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(str, Enum):
    """Types of events emitted during pipeline execution."""
    PIPELINE_STARTED = "pipeline_started"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    ANALYST_SPAWNED = "analyst_spawned"
    ANALYST_COMPLETED = "analyst_completed"
    SYNTHESIS_STARTED = "synthesis_started"
    FINAL_RESPONSE = "final_response"
    ERROR = "error"


class StepName(str, Enum):
    """Names of steps in the pipeline."""
    RETRIEVE = "retrieve"
    DISTRIBUTE = "distribute"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"


@dataclass
class AgentEvent:
    """Event data structure for SSE streaming."""
    type: EventType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_sse(self) -> str:
        """Convert to SSE format string."""
        event_dict = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data
        }
        return f"data: {json.dumps(event_dict)}\n\n"


class StreamingCallbackHandler:
    """
    Callback handler that emits SSE events during LangGraph execution.
    
    This handler tracks the progress through the RAG pipeline and emits
    events for each step, allowing the frontend to display real-time
    progress updates.
    """
    
    def __init__(self):
        self._event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._start_time: float = 0
        self._step_start_times: Dict[str, float] = {}
        self._total_analysts: int = 0
        self._completed_analysts: int = 0
        self._relevant_count: int = 0
        self._is_closed: bool = False
    
    @property
    def event_queue(self) -> asyncio.Queue[AgentEvent]:
        """Get the event queue for consuming events."""
        return self._event_queue
    
    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to the queue."""
        if not self._is_closed:
            await self._event_queue.put(event)
    
    async def close(self) -> None:
        """Close the handler and signal end of stream."""
        self._is_closed = True
        # Put a sentinel value to signal stream end
        await self._event_queue.put(None)
    
    async def on_pipeline_start(self, query: str) -> None:
        """Called when the pipeline starts executing."""
        self._start_time = time.time()
        await self.emit(AgentEvent(
            type=EventType.PIPELINE_STARTED,
            data={"query": query}
        ))
    
    async def on_step_start(self, step: StepName, extra_data: Dict[str, Any] = None) -> None:
        """Called when a step starts executing."""
        self._step_start_times[step.value] = time.time()
        data = {"step": step.value}
        if extra_data:
            data.update(extra_data)
        await self.emit(AgentEvent(
            type=EventType.STEP_STARTED,
            data=data
        ))
    
    async def on_step_complete(self, step: StepName, extra_data: Dict[str, Any] = None) -> None:
        """Called when a step completes."""
        duration_ms = 0
        if step.value in self._step_start_times:
            duration_ms = int((time.time() - self._step_start_times[step.value]) * 1000)
        
        data = {"step": step.value, "duration_ms": duration_ms}
        if extra_data:
            data.update(extra_data)
        await self.emit(AgentEvent(
            type=EventType.STEP_COMPLETED,
            data=data
        ))
    
    async def on_analysts_spawned(self, total_analysts: int, chunk_ids: List[str]) -> None:
        """Called when analyst agents are spawned for parallel execution."""
        self._total_analysts = total_analysts
        self._completed_analysts = 0
        self._relevant_count = 0
        await self.emit(AgentEvent(
            type=EventType.ANALYST_SPAWNED,
            data={
                "total_analysts": total_analysts,
                "chunk_ids": chunk_ids
            }
        ))
    
    async def on_analyst_complete(
        self,
        analyst_id: str,
        chunk_id: str,
        relevance_score: int,
        is_relevant: bool,
        summary: str,
        confidence: str
    ) -> None:
        """Called when an individual analyst completes."""
        self._completed_analysts += 1
        if is_relevant:
            self._relevant_count += 1
        
        await self.emit(AgentEvent(
            type=EventType.ANALYST_COMPLETED,
            data={
                "analyst_id": analyst_id,
                "chunk_id": chunk_id,
                "relevance_score": relevance_score,
                "is_relevant": is_relevant,
                "summary": summary,
                "confidence": confidence,
                "completed_analysts": self._completed_analysts,
                "total_analysts": self._total_analysts,
                "relevant_count": self._relevant_count
            }
        ))
    
    async def on_synthesis_start(self) -> None:
        """Called when synthesis begins."""
        await self.emit(AgentEvent(
            type=EventType.SYNTHESIS_STARTED,
            data={
                "total_analysts": self._total_analysts,
                "relevant_count": self._relevant_count
            }
        ))
    
    async def on_final_response(
        self,
        final_response: str,
        retrieval_count: int,
        relevant_count: int,
        sources_used: List[str],
        shared_document: str
    ) -> None:
        """Called when the final response is ready."""
        total_duration_ms = int((time.time() - self._start_time) * 1000)
        await self.emit(AgentEvent(
            type=EventType.FINAL_RESPONSE,
            data={
                "final_response": final_response,
                "retrieval_count": retrieval_count,
                "relevant_count": relevant_count,
                "sources_used": sources_used,
                "shared_document": shared_document,
                "total_duration_ms": total_duration_ms
            }
        ))
    
    async def on_error(self, error: str, step: Optional[str] = None) -> None:
        """Called when an error occurs."""
        await self.emit(AgentEvent(
            type=EventType.ERROR,
            data={
                "error": error,
                "step": step
            }
        ))


# Global registry for callback handlers by session
_callback_handlers: Dict[str, StreamingCallbackHandler] = {}


def get_callback_handler(session_id: str) -> StreamingCallbackHandler:
    """Get or create a callback handler for a session."""
    if session_id not in _callback_handlers:
        _callback_handlers[session_id] = StreamingCallbackHandler()
    return _callback_handlers[session_id]


def remove_callback_handler(session_id: str) -> None:
    """Remove a callback handler for a session."""
    if session_id in _callback_handlers:
        del _callback_handlers[session_id]
