const API_BASE = '/api';

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  chunks_count: number;
}

export interface ProcessingStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  filename: string;
  chunks_count?: number;
  error?: string;
}

export interface SearchResult {
  chunk_id: string;
  content: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: SearchResult[];
}

export async function uploadDocument(file: File): Promise<{ job_id: string; filename: string; status: string }> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) throw new Error('Upload failed');
  return res.json();
}

export async function getStatus(jobId: string): Promise<ProcessingStatus> {
  const res = await fetch(`${API_BASE}/documents/status/${jobId}`);
  if (!res.ok) throw new Error('Failed to get status');
  return res.json();
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error('Failed to list documents');
  return res.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete document');
}

export async function search(query: string, topK = 5): Promise<{ results: SearchResult[] }> {
  const res = await fetch(`${API_BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error('Search failed');
  return res.json();
}

export async function chat(message: string, topK = 5): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, top_k: topK }),
  });
  if (!res.ok) throw new Error('Chat failed');
  return res.json();
}
