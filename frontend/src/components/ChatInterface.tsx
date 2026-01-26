import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { chat, ChatResponse, SearchResult } from '../api/client';
import { useAgentStream } from '../hooks/useAgentStream';
import AgentStateViewer from './AgentStateViewer';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SearchResult[];
  sourcesUsed?: string[];
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Agent streaming hook
  const { state: agentState, isStreaming, startQuery, abort, reset } = useAgentStream();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, agentState]);

  // Handle streaming completion
  useEffect(() => {
    if (agentState.status === 'complete' && agentState.result) {
      setMessages(prev => [
        ...prev,
        { 
          role: 'assistant', 
          content: agentState.result!.finalResponse,
          sourcesUsed: agentState.result!.sourcesUsed 
        }
      ]);
      // Reset agent state after adding message
      setTimeout(() => reset(), 500);
    }
    if (agentState.status === 'error') {
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${agentState.error || 'An unknown error occurred'}` }
      ]);
      setTimeout(() => reset(), 500);
    }
  }, [agentState.status, agentState.result, agentState.error, reset]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading || isStreaming) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    if (useStreaming) {
      // Use streaming RAG endpoint
      await startQuery(userMessage);
    } else {
      // Use original chat endpoint
      setLoading(true);
      try {
        const response: ChatResponse = await chat(userMessage);
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: response.answer, sources: response.sources }
        ]);
      } catch (err) {
        setMessages(prev => [
          ...prev,
          { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }
        ]);
      } finally {
        setLoading(false);
      }
    }
  };

  const toggleSources = (index: number) => {
    setExpandedSources(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Sparkles className="w-12 h-12 text-electric mb-4" />
            <h3 className="text-pearl text-lg font-medium mb-2">Ask anything about your documents</h3>
            <p className="text-steel text-sm max-w-md">
              Upload documents first, then ask questions. I'll search through your indexed content to provide accurate answers.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 animate-slide-up ${msg.role === 'user' ? 'justify-end' : ''}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-lg bg-electric/20 flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-electric" />
              </div>
            )}
            
            <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-first' : ''}`}>
              <div
                className={`p-4 rounded-2xl ${
                  msg.role === 'user'
                    ? 'bg-azure text-midnight rounded-br-md'
                    : 'bg-charcoal border border-slate/50 text-pearl rounded-bl-md'
                }`}
              >
                {msg.role === 'assistant' ? (
                  <div className="prose prose-invert prose-sm max-w-none prose-pre:bg-midnight prose-pre:border prose-pre:border-slate/50 prose-code:before:content-none prose-code:after:content-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>

              {/* Sources (from original chat) */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleSources(idx)}
                    className="flex items-center gap-1 text-xs text-steel hover:text-silver transition-colors"
                  >
                    {expandedSources.has(idx) ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                    {msg.sources.length} sources
                  </button>
                  
                  {expandedSources.has(idx) && (
                    <div className="mt-2 space-y-2">
                      {msg.sources.map((source, sIdx) => (
                        <div
                          key={sIdx}
                          className="p-3 bg-midnight/50 rounded-lg border border-slate/30 text-xs"
                        >
                          <p className="text-silver line-clamp-3">{source.content}</p>
                          <p className="text-steel mt-1">Score: {source.score.toFixed(3)}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Sources used (from streaming RAG) */}
              {msg.sourcesUsed && msg.sourcesUsed.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleSources(idx)}
                    className="flex items-center gap-1 text-xs text-steel hover:text-silver transition-colors"
                  >
                    {expandedSources.has(idx) ? (
                      <ChevronUp className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                    {msg.sourcesUsed.length} sources
                  </button>
                  
                  {expandedSources.has(idx) && (
                    <div className="mt-2 space-y-1">
                      {msg.sourcesUsed.map((source, sIdx) => (
                        <div
                          key={sIdx}
                          className="p-2 bg-midnight/50 rounded-lg border border-slate/30 text-xs"
                        >
                          <p className="text-silver">{source}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-lg bg-azure/20 flex items-center justify-center flex-shrink-0">
                <User className="w-5 h-5 text-azure" />
              </div>
            )}
          </div>
        ))}

        {/* Agent State Viewer - shows during streaming */}
        {(isStreaming || agentState.status !== 'idle') && agentState.status !== 'complete' && agentState.status !== 'error' && (
          <div className="flex gap-3 animate-slide-up">
            <div className="w-8 h-8 rounded-lg bg-electric/20 flex items-center justify-center flex-shrink-0">
              <Bot className="w-5 h-5 text-electric" />
            </div>
            <div className="flex-1 max-w-[90%]">
              <AgentStateViewer 
                state={agentState} 
                isStreaming={isStreaming}
                onAbort={abort}
              />
            </div>
          </div>
        )}

        {/* Simple loading indicator for non-streaming */}
        {loading && !useStreaming && (
          <div className="flex gap-3 animate-slide-up">
            <div className="w-8 h-8 rounded-lg bg-electric/20 flex items-center justify-center">
              <Bot className="w-5 h-5 text-electric" />
            </div>
            <div className="p-4 bg-charcoal border border-slate/50 rounded-2xl rounded-bl-md">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-azure rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-azure rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-azure rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-slate/30">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={loading || isStreaming}
            className="flex-1 bg-charcoal border border-slate/50 rounded-xl px-4 py-3 text-pearl placeholder:text-steel focus:outline-none focus:border-azure/50 transition-colors"
          />
          {/* Streaming toggle button */}
          <button
            type="button"
            onClick={() => setUseStreaming(!useStreaming)}
            disabled={loading || isStreaming}
            className={`px-3 py-3 rounded-xl border transition-all ${
              useStreaming
                ? 'bg-electric/20 border-electric/50 text-electric'
                : 'bg-charcoal border-slate/50 text-steel'
            } hover:opacity-80 disabled:opacity-50`}
            title={useStreaming ? 'Multi-Agent RAG (streaming)' : 'Simple RAG'}
          >
            <Zap className="w-5 h-5" />
          </button>
          <button
            type="submit"
            disabled={loading || isStreaming || !input.trim()}
            className="px-4 py-3 bg-azure text-midnight font-medium rounded-xl hover:bg-azure/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        {/* Mode indicator */}
        <p className="mt-2 text-xs text-steel text-center">
          {useStreaming ? (
            <span className="flex items-center justify-center gap-1">
              <Zap className="w-3 h-3 text-electric" />
              Multi-Agent RAG with real-time progress
            </span>
          ) : (
            'Simple RAG mode'
          )}
        </p>
      </form>
    </div>
  );
}
