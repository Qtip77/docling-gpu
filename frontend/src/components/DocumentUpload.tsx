import { useState, useCallback } from 'react';
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from 'lucide-react';
import { uploadDocument, getStatus, ProcessingStatus } from '../api/client';

interface Props {
  onUploadComplete: () => void;
}

export default function DocumentUpload({ onUploadComplete }: Props) {
  const [isDragging, setIsDragging] = useState(false);
  const [jobs, setJobs] = useState<ProcessingStatus[]>([]);

  const pollStatus = useCallback(async (jobId: string) => {
    const poll = async () => {
      try {
        const status = await getStatus(jobId);
        setJobs(prev => prev.map(j => j.job_id === jobId ? status : j));
        
        if (status.status === 'completed') {
          onUploadComplete();
        } else if (status.status !== 'failed') {
          setTimeout(poll, 2000);
        }
      } catch {
        // Keep polling on error
        setTimeout(poll, 2000);
      }
    };
    poll();
  }, [onUploadComplete]);

  const handleFile = async (file: File) => {
    try {
      const result = await uploadDocument(file);
      const newJob: ProcessingStatus = {
        job_id: result.job_id,
        filename: result.filename,
        status: 'pending',
      };
      setJobs(prev => [newJob, ...prev]);
      pollStatus(result.job_id);
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-mint" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-coral" />;
      default:
        return <Loader2 className="w-5 h-5 text-azure animate-spin" />;
    }
  };

  return (
    <div className="space-y-4">
      <label
        className={`
          flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl cursor-pointer
          transition-all duration-300
          ${isDragging 
            ? 'border-azure bg-azure/10 glow-pulse' 
            : 'border-slate hover:border-azure/50 hover:bg-charcoal/50'
          }
        `}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        <Upload className={`w-12 h-12 mb-3 transition-colors ${isDragging ? 'text-azure' : 'text-steel'}`} />
        <span className="text-silver text-sm">Drop PDF here or click to upload</span>
        <span className="text-steel text-xs mt-1">Supports tables, images, handwritten notes</span>
        <input type="file" accept=".pdf" onChange={handleChange} className="hidden" />
      </label>

      {jobs.length > 0 && (
        <div className="space-y-2">
          {jobs.map(job => (
            <div 
              key={job.job_id}
              className="flex items-center gap-3 p-3 bg-charcoal/50 rounded-lg border border-slate/50 animate-slide-up"
            >
              <FileText className="w-5 h-5 text-silver" />
              <div className="flex-1 min-w-0">
                <p className="text-pearl text-sm truncate">{job.filename}</p>
                <p className="text-steel text-xs">
                  {job.status === 'completed' && `${job.chunks_count} chunks indexed`}
                  {job.status === 'processing' && 'Processing with OCR...'}
                  {job.status === 'pending' && 'Waiting...'}
                  {job.status === 'failed' && (job.error || 'Failed')}
                </p>
              </div>
              {getStatusIcon(job.status)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
