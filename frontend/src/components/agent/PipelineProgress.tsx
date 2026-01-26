/**
 * Horizontal progress indicator with 4 steps.
 * Active step pulses, completed steps show checkmark, shows step timing on completion.
 */
import { Check, Loader2, Circle } from 'lucide-react';
import { PipelineState, StepStatus } from '../../api/ragStream';

interface PipelineProgressProps {
  state: PipelineState;
}

interface StepConfig {
  name: string;
  label: string;
  getStatus: (state: PipelineState) => StepStatus;
  getDuration: (state: PipelineState) => number | undefined;
}

const steps: StepConfig[] = [
  {
    name: 'retrieve',
    label: 'Retrieve',
    getStatus: (state) => state.retrieval.status,
    getDuration: (state) => state.retrieval.durationMs,
  },
  {
    name: 'distribute',
    label: 'Distribute',
    getStatus: (state) => {
      if (state.currentStep < 2) return 'pending';
      if (state.currentStep === 2 && state.status === 'distributing') return 'active';
      return 'complete';
    },
    getDuration: () => undefined, // Distribution is instant
  },
  {
    name: 'analyze',
    label: 'Analyze',
    getStatus: (state) => state.analysis.status,
    getDuration: (state) => state.analysis.durationMs,
  },
  {
    name: 'synthesize',
    label: 'Synthesize',
    getStatus: (state) => state.synthesis.status,
    getDuration: (state) => state.synthesis.durationMs,
  },
];

export default function PipelineProgress({ state }: PipelineProgressProps) {
  return (
    <div className="flex items-center justify-between">
      {steps.map((step, index) => {
        const status = step.getStatus(state);
        const duration = step.getDuration(state);
        const isLast = index === steps.length - 1;

        return (
          <div key={step.name} className="flex items-center flex-1">
            <div className="flex flex-col items-center">
              {/* Step indicator */}
              <div
                className={`
                  w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300
                  ${status === 'complete' 
                    ? 'bg-emerald/20 text-emerald' 
                    : status === 'active'
                    ? 'bg-azure/20 text-azure animate-pulse'
                    : 'bg-slate/20 text-steel'
                  }
                `}
              >
                {status === 'complete' ? (
                  <Check className="w-4 h-4" />
                ) : status === 'active' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Circle className="w-3 h-3" />
                )}
              </div>

              {/* Step label */}
              <span
                className={`
                  mt-1 text-xs font-medium transition-colors
                  ${status === 'complete' 
                    ? 'text-emerald' 
                    : status === 'active'
                    ? 'text-azure'
                    : 'text-steel'
                  }
                `}
              >
                {step.label}
              </span>

              {/* Duration */}
              {status === 'complete' && duration !== undefined && (
                <span className="text-[10px] text-steel">
                  {formatDuration(duration)}
                </span>
              )}
              {status === 'active' && (
                <span className="text-[10px] text-steel">
                  Running...
                </span>
              )}
              {status === 'pending' && (
                <span className="text-[10px] text-slate/50">
                  Waiting
                </span>
              )}
            </div>

            {/* Connector line */}
            {!isLast && (
              <div
                className={`
                  flex-1 h-0.5 mx-2 transition-colors duration-500
                  ${status === 'complete' 
                    ? 'bg-emerald/50' 
                    : 'bg-slate/30'
                  }
                `}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}
