import { useState, useEffect } from 'react';
import { FileText, Trash2, RefreshCw, Database } from 'lucide-react';
import { listDocuments, deleteDocument, DocumentInfo } from '../api/client';

interface Props {
  refreshTrigger: number;
}

export default function DocumentList({ refreshTrigger }: Props) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshTrigger]);

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      setDocuments(prev => prev.filter(d => d.doc_id !== docId));
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-azure" />
          <span className="text-silver text-sm font-medium">Indexed Documents</span>
        </div>
        <button
          onClick={fetchDocuments}
          disabled={loading}
          className="p-1.5 text-steel hover:text-azure transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {documents.length === 0 ? (
        <div className="text-center py-8 text-steel text-sm">
          No documents indexed yet
        </div>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {documents.map(doc => (
            <div
              key={doc.doc_id}
              className="flex items-center gap-3 p-3 bg-charcoal/30 rounded-lg border border-slate/30 group hover:border-slate/50 transition-colors"
            >
              <FileText className="w-4 h-4 text-electric" />
              <div className="flex-1 min-w-0">
                <p className="text-pearl text-sm truncate">{doc.filename}</p>
                <p className="text-steel text-xs">{doc.chunks_count} chunks</p>
              </div>
              <button
                onClick={() => handleDelete(doc.doc_id)}
                className="p-1.5 text-steel opacity-0 group-hover:opacity-100 hover:text-coral transition-all"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
