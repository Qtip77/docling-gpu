/**
 * Expandable panel showing analyst contributions (shared document).
 * Markdown formatted with collapsible sections.
 */
import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface SharedDocumentViewerProps {
  content: string;
}

export default function SharedDocumentViewer({ content }: SharedDocumentViewerProps) {
  const [expanded, setExpanded] = useState(false);

  if (!content) {
    return null;
  }

  // Count analyst contributions (separated by ---)
  const contributions = content.split('---').filter((c) => c.trim()).length;

  return (
    <div className="border border-slate/30 rounded-lg overflow-hidden">
      {/* Header - always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 bg-midnight/50 hover:bg-midnight/70 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-electric" />
          <span className="text-sm font-medium text-pearl">
            Analyst Contributions
          </span>
          <span className="text-xs text-steel">
            ({contributions} {contributions === 1 ? 'entry' : 'entries'})
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-steel" />
        ) : (
          <ChevronDown className="w-4 h-4 text-steel" />
        )}
      </button>

      {/* Content - expandable */}
      {expanded && (
        <div className="p-4 border-t border-slate/20 max-h-96 overflow-y-auto">
          <div className="prose prose-invert prose-sm max-w-none prose-pre:bg-midnight prose-pre:border prose-pre:border-slate/50 prose-code:before:content-none prose-code:after:content-none prose-headings:text-pearl prose-h3:text-sm prose-h3:font-semibold prose-p:text-silver prose-ul:text-silver prose-li:text-silver">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
