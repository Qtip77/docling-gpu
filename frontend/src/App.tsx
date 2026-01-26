import { useState } from 'react';
import { FileStack, MessageSquare, Zap } from 'lucide-react';
import DocumentUpload from './components/DocumentUpload';
import DocumentList from './components/DocumentList';
import ChatInterface from './components/ChatInterface';

export default function App() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadComplete = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen text-pearl font-sans">
      {/* Header */}
      <header className="border-b border-slate/30 bg-midnight/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-azure to-electric flex items-center justify-center">
            <Zap className="w-6 h-6 text-midnight" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Docling RAG</h1>
            <p className="text-steel text-xs">Azure AI Search + Azure OpenAI</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[calc(100vh-120px)]">
          {/* Left Panel - Documents */}
          <div className="lg:col-span-1 space-y-6">
            {/* Upload Section */}
            <section className="bg-charcoal/30 backdrop-blur-sm rounded-2xl border border-slate/30 p-5">
              <div className="flex items-center gap-2 mb-4">
                <FileStack className="w-5 h-5 text-azure" />
                <h2 className="text-pearl font-medium">Upload Documents</h2>
              </div>
              <DocumentUpload onUploadComplete={handleUploadComplete} />
            </section>

            {/* Documents List */}
            <section className="bg-charcoal/30 backdrop-blur-sm rounded-2xl border border-slate/30 p-5">
              <DocumentList refreshTrigger={refreshTrigger} />
            </section>
          </div>

          {/* Right Panel - Chat */}
          <div className="lg:col-span-2 bg-charcoal/30 backdrop-blur-sm rounded-2xl border border-slate/30 flex flex-col overflow-hidden">
            <div className="flex items-center gap-2 px-5 py-4 border-b border-slate/30">
              <MessageSquare className="w-5 h-5 text-electric" />
              <h2 className="text-pearl font-medium">Chat with Documents</h2>
            </div>
            <ChatInterface />
          </div>
        </div>
      </main>
    </div>
  );
}
