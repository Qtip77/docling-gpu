/**
 * Grid of analyst cards showing parallel chunk analysis status.
 * 
 * Each card shows:
 * - Chunk ID
 * - Status (spinner/check/x)
 * - Relevance score bar (0-10)
 * - Confidence badge
 * - Expandable summary
 */
import { useState } from 'react';
import { Check, X, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { AnalystState, ConfidenceLevel } from '../../api/ragStream';

interface AnalystGridProps {
  analysts: AnalystState[];
  totalAnalysts: number;
  completedAnalysts: number;
}

export default function AnalystGrid({
  analysts,
  totalAnalysts,
  completedAnalysts,
}: AnalystGridProps) {
  const relevantCount = analysts.filter((a) => a.isRelevant).length;
  const relevantPercentage = completedAnalysts > 0 
    ? Math.round((relevantCount / completedAnalysts) * 100)
    : 0;

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-steel">
          Analysts: {completedAnalysts}/{totalAnalysts} complete
        </span>
        <span className="text-steel">
          Relevant: {relevantCount}/{completedAnalysts} ({relevantPercentage}%)
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 bg-slate/30 rounded-full overflow-hidden">
        <div
          className="h-full bg-azure transition-all duration-300 ease-out rounded-full"
          style={{ width: `${(completedAnalysts / totalAnalysts) * 100}%` }}
        />
      </div>

      {/* Analyst cards grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {analysts.slice(0, 20).map((analyst) => (
          <AnalystCard key={analyst.id} analyst={analyst} />
        ))}
      </div>

      {/* Show more indicator */}
      {analysts.length > 20 && (
        <p className="text-xs text-steel text-center">
          ... and {analysts.length - 20} more
        </p>
      )}
    </div>
  );
}

interface AnalystCardProps {
  analyst: AnalystState;
}

function AnalystCard({ analyst }: AnalystCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`
        p-2 rounded-lg border transition-all cursor-pointer
        ${analyst.status === 'complete'
          ? analyst.isRelevant
            ? 'bg-emerald/5 border-emerald/30'
            : 'bg-slate/5 border-slate/30'
          : analyst.status === 'analyzing'
          ? 'bg-azure/5 border-azure/30 animate-pulse'
          : 'bg-slate/5 border-slate/20'
        }
      `}
      onClick={() => analyst.status === 'complete' && setExpanded(!expanded)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-steel truncate max-w-[60px]">
          {analyst.chunkId}
        </span>
        <StatusIcon status={analyst.status} isRelevant={analyst.isRelevant} />
      </div>

      {/* Relevance score bar */}
      {analyst.status === 'complete' && analyst.relevanceScore !== undefined && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-steel">
              {analyst.relevanceScore}/10
            </span>
            {analyst.confidence && (
              <ConfidenceBadge confidence={analyst.confidence} />
            )}
          </div>
          <div className="h-1 bg-slate/30 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${getScoreColor(
                analyst.relevanceScore
              )}`}
              style={{ width: `${analyst.relevanceScore * 10}%` }}
            />
          </div>
        </div>
      )}

      {/* Pending/analyzing state */}
      {analyst.status === 'pending' && (
        <div className="h-4 flex items-center justify-center">
          <span className="text-[10px] text-slate/50">Waiting</span>
        </div>
      )}
      {analyst.status === 'analyzing' && (
        <div className="h-4 flex items-center justify-center gap-1">
          <Loader2 className="w-3 h-3 text-azure animate-spin" />
          <span className="text-[10px] text-azure">Analyzing</span>
        </div>
      )}

      {/* Expandable summary */}
      {expanded && analyst.summary && (
        <div className="mt-2 pt-2 border-t border-slate/20">
          <p className="text-[10px] text-silver leading-relaxed line-clamp-4">
            {analyst.summary}
          </p>
        </div>
      )}

      {/* Expand indicator */}
      {analyst.status === 'complete' && analyst.summary && (
        <div className="flex justify-center mt-1">
          {expanded ? (
            <ChevronUp className="w-3 h-3 text-steel" />
          ) : (
            <ChevronDown className="w-3 h-3 text-steel" />
          )}
        </div>
      )}
    </div>
  );
}

function StatusIcon({
  status,
  isRelevant,
}: {
  status: AnalystState['status'];
  isRelevant?: boolean;
}) {
  if (status === 'analyzing') {
    return <Loader2 className="w-3 h-3 text-azure animate-spin" />;
  }
  if (status === 'complete') {
    if (isRelevant) {
      return <Check className="w-3 h-3 text-emerald" />;
    }
    return <X className="w-3 h-3 text-steel" />;
  }
  return <div className="w-3 h-3 rounded-full bg-slate/30" />;
}

function ConfidenceBadge({ confidence }: { confidence: ConfidenceLevel }) {
  const colors = {
    high: 'bg-emerald/20 text-emerald',
    medium: 'bg-amber/20 text-amber',
    low: 'bg-slate/20 text-steel',
  };

  return (
    <span
      className={`px-1 py-0.5 rounded text-[8px] font-medium uppercase ${colors[confidence]}`}
    >
      {confidence}
    </span>
  );
}

function getScoreColor(score: number): string {
  if (score >= 7) return 'bg-emerald';
  if (score >= 4) return 'bg-amber';
  return 'bg-slate';
}
