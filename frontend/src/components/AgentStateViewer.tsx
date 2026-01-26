/**
 * Main component for visualizing agent pipeline state.
 * 
 * Shows real-time progress as the RAG pipeline processes a query through
 * the multi-agent system: Retrieve → Distribute → Analyze → Synthesize
 */
import { PipelineState } from '../api/ragStream';
import PipelineProgress from './agent/PipelineProgress';
import AnalystGrid from './agent/AnalystGrid';
import MetricsPanel from './agent/MetricsPanel';
import SharedDocumentViewer from './agent/SharedDocumentViewer';
import { Loader2, XCircle } from 'lucide-react';

interface AgentStateViewerProps {
  state: PipelineState;
  isStreaming: boolean;
  onAbort?: () => void;
}

export default function AgentStateViewer({
  state,
  isStreaming,
  onAbort,
}: AgentStateViewerProps) {
  const showAnalystGrid =
    state.status === 'analyzing' ||
    (state.status === 'synthesizing' && state.analysis.analysts.length > 0) ||
    state.status === 'complete';

  const showMetrics =
    state.currentStep >= 1 && state.status !== 'idle';

  if (state.status === 'idle') {
    return null;
  }

  return (
    <div className="bg-charcoal border border-slate/50 rounded-2xl p-4 space-y-4 animate-slide-up">
      {/* Header with abort button */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isStreaming && (
            <Loader2 className="w-4 h-4 text-azure animate-spin" />
          )}
          <span className="text-sm font-medium text-pearl">
            {getStatusLabel(state.status)}
          </span>
        </div>
        {isStreaming && onAbort && (
          <button
            onClick={onAbort}
            className="flex items-center gap-1 text-xs text-steel hover:text-ruby transition-colors"
          >
            <XCircle className="w-4 h-4" />
            Cancel
          </button>
        )}
      </div>

      {/* Pipeline Progress */}
      <PipelineProgress state={state} />

      {/* Error State */}
      {state.status === 'error' && state.error && (
        <div className="p-3 bg-ruby/10 border border-ruby/30 rounded-lg">
          <p className="text-sm text-ruby">{state.error}</p>
        </div>
      )}

      {/* Current Step Details */}
      {state.status === 'retrieving' && (
        <div className="p-3 bg-midnight/50 rounded-lg">
          <p className="text-sm text-steel">
            Searching documents for relevant information...
          </p>
        </div>
      )}

      {state.status === 'distributing' && (
        <div className="p-3 bg-midnight/50 rounded-lg">
          <p className="text-sm text-steel">
            Distributing chunks to analyst agents...
          </p>
        </div>
      )}

      {/* Analyst Grid - show during and after analysis */}
      {showAnalystGrid && state.analysis.analysts.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-steel">
            Analyzing {state.analysis.totalAnalysts} chunks in parallel
          </p>
          <AnalystGrid
            analysts={state.analysis.analysts}
            totalAnalysts={state.analysis.totalAnalysts}
            completedAnalysts={state.analysis.completedAnalysts}
          />
        </div>
      )}

      {/* Synthesis State */}
      {state.status === 'synthesizing' && (
        <div className="p-3 bg-midnight/50 rounded-lg flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-electric animate-spin" />
          <p className="text-sm text-steel">
            Synthesizing final response from {state.analysis.completedAnalysts} analyzed chunks...
          </p>
        </div>
      )}

      {/* Metrics Panel */}
      {showMetrics && <MetricsPanel state={state} />}

      {/* Shared Document Viewer - only show when complete */}
      {state.status === 'complete' && state.result?.sharedDocument && (
        <SharedDocumentViewer content={state.result.sharedDocument} />
      )}
    </div>
  );
}

function getStatusLabel(status: PipelineState['status']): string {
  switch (status) {
    case 'idle':
      return 'Ready';
    case 'retrieving':
      return 'Retrieving Documents...';
    case 'distributing':
      return 'Distributing to Analysts...';
    case 'analyzing':
      return 'Analyzing Chunks...';
    case 'synthesizing':
      return 'Synthesizing Response...';
    case 'complete':
      return 'Complete';
    case 'error':
      return 'Error';
    default:
      return 'Processing...';
  }
}
