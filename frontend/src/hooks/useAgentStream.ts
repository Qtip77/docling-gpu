/**
 * React hook for managing agent state stream.
 */
import { useState, useCallback, useRef } from 'react';
import {
  PipelineState,
  AgentEvent,
  createInitialPipelineState,
  startStreamingQuery,
} from '../api/ragStream';

export interface UseAgentStreamResult {
  /** Current pipeline state */
  state: PipelineState;
  /** Whether streaming is active */
  isStreaming: boolean;
  /** Start a new query */
  startQuery: (query: string, filters?: Record<string, unknown>) => Promise<void>;
  /** Abort the current query */
  abort: () => void;
  /** Reset state to initial */
  reset: () => void;
  /** All events received (for debugging) */
  events: AgentEvent[];
}

export function useAgentStream(): UseAgentStreamResult {
  const [state, setState] = useState<PipelineState>(createInitialPipelineState());
  const [isStreaming, setIsStreaming] = useState(false);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    setState(createInitialPipelineState());
    setEvents([]);
    setIsStreaming(false);
  }, []);

  const abort = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const startQuery = useCallback(
    async (query: string, filters?: Record<string, unknown>) => {
      // Abort any existing stream
      abort();
      
      // Reset state for new query
      setState(createInitialPipelineState());
      setEvents([]);
      setIsStreaming(true);

      try {
        abortControllerRef.current = await startStreamingQuery(query, filters, {
          onEvent: (event) => {
            setEvents((prev) => [...prev, event]);
          },
          onStateChange: (newState) => {
            setState(newState);
          },
          onComplete: () => {
            setIsStreaming(false);
            abortControllerRef.current = null;
          },
          onError: (error) => {
            console.error('Stream error:', error);
            setState((prev) => ({
              ...prev,
              status: 'error',
              error: error.message,
            }));
            setIsStreaming(false);
            abortControllerRef.current = null;
          },
        });
      } catch (error) {
        console.error('Failed to start stream:', error);
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: (error as Error).message,
        }));
        setIsStreaming(false);
      }
    },
    [abort]
  );

  return {
    state,
    isStreaming,
    startQuery,
    abort,
    reset,
    events,
  };
}
