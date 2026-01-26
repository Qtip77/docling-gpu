/**
 * Real-time metrics display panel.
 * 
 * Shows:
 * - Chunks retrieved / analyzed
 * - Relevant vs non-relevant ratio
 * - Elapsed time
 */
import { FileSearch, CheckCircle, Clock, BarChart3 } from 'lucide-react';
import { PipelineState } from '../../api/ragStream';

interface MetricsPanelProps {
  state: PipelineState;
}

export default function MetricsPanel({ state }: MetricsPanelProps) {
  const { retrieval, analysis, totalDurationMs } = state;
  
  const relevantCount = analysis.analysts.filter((a) => a.isRelevant).length;
  const relevantPercentage = analysis.completedAnalysts > 0
    ? Math.round((relevantCount / analysis.completedAnalysts) * 100)
    : 0;
  
  const avgRelevance = analysis.analysts.length > 0
    ? analysis.analysts
        .filter((a) => a.relevanceScore !== undefined)
        .reduce((sum, a) => sum + (a.relevanceScore || 0), 0) /
      analysis.analysts.filter((a) => a.relevanceScore !== undefined).length
    : 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* Chunks Retrieved */}
      <MetricCard
        icon={<FileSearch className="w-4 h-4 text-azure" />}
        label="Retrieved"
        value={retrieval.chunksRetrieved.toString()}
        subValue={retrieval.durationMs ? `${retrieval.durationMs}ms` : undefined}
      />

      {/* Relevant Chunks */}
      <MetricCard
        icon={<CheckCircle className="w-4 h-4 text-emerald" />}
        label="Relevant"
        value={`${relevantCount}/${analysis.completedAnalysts}`}
        subValue={`${relevantPercentage}%`}
      />

      {/* Average Relevance */}
      <MetricCard
        icon={<BarChart3 className="w-4 h-4 text-electric" />}
        label="Avg Score"
        value={avgRelevance > 0 ? avgRelevance.toFixed(1) : '-'}
        subValue="/10"
      />

      {/* Duration */}
      <MetricCard
        icon={<Clock className="w-4 h-4 text-amber" />}
        label="Elapsed"
        value={formatDuration(totalDurationMs || calculateElapsed(state))}
        subValue={state.status === 'complete' ? 'total' : 'running'}
      />
    </div>
  );
}

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subValue?: string;
}

function MetricCard({ icon, label, value, subValue }: MetricCardProps) {
  return (
    <div className="p-3 bg-midnight/50 rounded-lg border border-slate/20">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-[10px] text-steel uppercase tracking-wide">
          {label}
        </span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-lg font-semibold text-pearl">{value}</span>
        {subValue && (
          <span className="text-xs text-steel">{subValue}</span>
        )}
      </div>
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms === 0) return '-';
  if (ms < 1000) {
    return `${ms}ms`;
  }
  if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function calculateElapsed(state: PipelineState): number {
  // Calculate approximate elapsed time from step durations
  let elapsed = 0;
  if (state.retrieval.durationMs) elapsed += state.retrieval.durationMs;
  if (state.analysis.durationMs) elapsed += state.analysis.durationMs;
  if (state.synthesis.durationMs) elapsed += state.synthesis.durationMs;
  return elapsed;
}
