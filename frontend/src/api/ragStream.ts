/**
 * SSE client utilities for consuming agent state stream.
 */

const API_BASE = '/api';

// Event types from the backend
export type EventType =
  | 'pipeline_started'
  | 'step_started'
  | 'step_completed'
  | 'analyst_spawned'
  | 'analyst_completed'
  | 'synthesis_started'
  | 'final_response'
  | 'error';

export type StepName = 'retrieve' | 'distribute' | 'analyze' | 'synthesize';
export type ConfidenceLevel = 'high' | 'medium' | 'low';

// Agent event from SSE stream
export interface AgentEvent {
  type: EventType;
  timestamp: string;
  data: {
    query?: string;
    step?: StepName;
    duration_ms?: number;
    retrieval_count?: number;
    total_analysts?: number;
    chunk_ids?: string[];
    analyst_id?: string;
    chunk_id?: string;
    relevance_score?: number;
    is_relevant?: boolean;
    summary?: string;
    confidence?: ConfidenceLevel;
    completed_analysts?: number;
    relevant_count?: number;
    final_response?: string;
    sources_used?: string[];
    shared_document?: string;
    total_duration_ms?: number;
    error?: string;
  };
}

// Pipeline state for UI
export type PipelineStatus =
  | 'idle'
  | 'retrieving'
  | 'distributing'
  | 'analyzing'
  | 'synthesizing'
  | 'complete'
  | 'error';

export type StepStatus = 'pending' | 'active' | 'complete';

export interface AnalystState {
  id: string;
  chunkId: string;
  status: 'pending' | 'analyzing' | 'complete';
  relevanceScore?: number;
  isRelevant?: boolean;
  summary?: string;
  confidence?: ConfidenceLevel;
}

export interface RetrievalState {
  status: StepStatus;
  chunksRetrieved: number;
  durationMs?: number;
}

export interface AnalysisState {
  status: StepStatus;
  totalAnalysts: number;
  completedAnalysts: number;
  analysts: AnalystState[];
  durationMs?: number;
}

export interface SynthesisState {
  status: StepStatus;
  durationMs?: number;
}

export interface PipelineResult {
  finalResponse: string;
  retrievalCount: number;
  relevantCount: number;
  sourcesUsed: string[];
  sharedDocument: string;
}

export interface PipelineState {
  status: PipelineStatus;
  currentStep: number; // 1-4
  totalSteps: 4;

  // Retrieval phase
  retrieval: RetrievalState;

  // Analysis phase
  analysis: AnalysisState;

  // Synthesis phase
  synthesis: SynthesisState;

  // Results
  result?: PipelineResult;

  // Error info
  error?: string;
  totalDurationMs?: number;
}

// Initial state factory
export function createInitialPipelineState(): PipelineState {
  return {
    status: 'idle',
    currentStep: 0,
    totalSteps: 4,
    retrieval: {
      status: 'pending',
      chunksRetrieved: 0,
    },
    analysis: {
      status: 'pending',
      totalAnalysts: 0,
      completedAnalysts: 0,
      analysts: [],
    },
    synthesis: {
      status: 'pending',
    },
  };
}

// State reducer for processing events
export function processAgentEvent(
  state: PipelineState,
  event: AgentEvent
): PipelineState {
  const newState = { ...state };

  switch (event.type) {
    case 'pipeline_started':
      return {
        ...createInitialPipelineState(),
        status: 'retrieving',
        currentStep: 1,
        retrieval: {
          status: 'active',
          chunksRetrieved: 0,
        },
      };

    case 'step_started':
      switch (event.data.step) {
        case 'retrieve':
          newState.status = 'retrieving';
          newState.currentStep = 1;
          newState.retrieval.status = 'active';
          break;
        case 'distribute':
          newState.status = 'distributing';
          newState.currentStep = 2;
          break;
        case 'analyze':
          newState.status = 'analyzing';
          newState.currentStep = 3;
          newState.analysis.status = 'active';
          if (event.data.total_analysts !== undefined) {
            newState.analysis.totalAnalysts = event.data.total_analysts;
          }
          break;
        case 'synthesize':
          newState.status = 'synthesizing';
          newState.currentStep = 4;
          newState.synthesis.status = 'active';
          break;
      }
      break;

    case 'step_completed':
      switch (event.data.step) {
        case 'retrieve':
          newState.retrieval.status = 'complete';
          newState.retrieval.durationMs = event.data.duration_ms;
          if (event.data.retrieval_count !== undefined) {
            newState.retrieval.chunksRetrieved = event.data.retrieval_count;
          }
          break;
        case 'distribute':
          // Distribution complete, moving to analyze
          break;
        case 'analyze':
          newState.analysis.status = 'complete';
          newState.analysis.durationMs = event.data.duration_ms;
          break;
        case 'synthesize':
          newState.synthesis.status = 'complete';
          newState.synthesis.durationMs = event.data.duration_ms;
          break;
      }
      break;

    case 'analyst_spawned':
      newState.analysis.totalAnalysts = event.data.total_analysts || 0;
      newState.analysis.analysts = (event.data.chunk_ids || []).map(
        (chunkId: string) => ({
          id: chunkId,
          chunkId,
          status: 'pending' as const,
        })
      );
      break;

    case 'analyst_completed':
      // Update the specific analyst
      const analystIndex = newState.analysis.analysts.findIndex(
        (a) => a.chunkId === event.data.chunk_id
      );
      
      if (analystIndex !== -1) {
        newState.analysis.analysts = [...newState.analysis.analysts];
        newState.analysis.analysts[analystIndex] = {
          ...newState.analysis.analysts[analystIndex],
          status: 'complete',
          relevanceScore: event.data.relevance_score,
          isRelevant: event.data.is_relevant,
          summary: event.data.summary,
          confidence: event.data.confidence,
        };
      }
      
      newState.analysis.completedAnalysts = event.data.completed_analysts || 0;
      break;

    case 'synthesis_started':
      newState.status = 'synthesizing';
      newState.currentStep = 4;
      newState.synthesis.status = 'active';
      break;

    case 'final_response':
      newState.status = 'complete';
      newState.synthesis.status = 'complete';
      newState.totalDurationMs = event.data.total_duration_ms;
      newState.result = {
        finalResponse: event.data.final_response || '',
        retrievalCount: event.data.retrieval_count || 0,
        relevantCount: event.data.relevant_count || 0,
        sourcesUsed: event.data.sources_used || [],
        sharedDocument: event.data.shared_document || '',
      };
      break;

    case 'error':
      newState.status = 'error';
      newState.error = event.data.error;
      break;
  }

  return newState;
}

// Parse SSE data line
function parseSSEData(line: string): AgentEvent | null {
  if (!line.startsWith('data: ')) {
    return null;
  }
  
  try {
    const jsonStr = line.slice(6); // Remove 'data: ' prefix
    return JSON.parse(jsonStr) as AgentEvent;
  } catch (e) {
    console.error('Failed to parse SSE event:', e);
    return null;
  }
}

export interface StreamOptions {
  onEvent?: (event: AgentEvent) => void;
  onStateChange?: (state: PipelineState) => void;
  onComplete?: (state: PipelineState) => void;
  onError?: (error: Error) => void;
}

/**
 * Start a streaming RAG query with real-time progress updates.
 * 
 * @param query - The query to process
 * @param filters - Optional metadata filters
 * @param options - Callbacks for handling events
 * @returns AbortController to cancel the stream
 */
export async function startStreamingQuery(
  query: string,
  filters?: Record<string, unknown>,
  options?: StreamOptions
): Promise<AbortController> {
  const abortController = new AbortController();
  let state = createInitialPipelineState();

  try {
    const response = await fetch(`${API_BASE}/rag/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({ query, filters }),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Stream request failed: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    // Process the stream
    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            options?.onComplete?.(state);
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine) continue;

            const event = parseSSEData(trimmedLine);
            if (event) {
              options?.onEvent?.(event);
              state = processAgentEvent(state, event);
              options?.onStateChange?.(state);

              if (state.status === 'complete' || state.status === 'error') {
                options?.onComplete?.(state);
              }
            }
          }
        }
      } catch (error) {
        if ((error as Error).name === 'AbortError') {
          // Stream was intentionally aborted
          return;
        }
        options?.onError?.(error as Error);
      }
    })();
  } catch (error) {
    options?.onError?.(error as Error);
  }

  return abortController;
}
