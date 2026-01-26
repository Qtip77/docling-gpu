import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException, Query
import aiofiles

from app.config import settings
from app.models.schemas import DocumentUploadResponse, ProcessingStatus, DocumentInfo
from app.services import docling_parser, azure_search, azure_openai

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# In-memory job tracking (stateless - lost on restart)
jobs: Dict[str, ProcessingStatus] = {}


async def process_document(job_id: str, file_path: str, filename: str, use_vlm: bool = False):
    """Background task to process uploaded document."""
    try:
        logger.info(f"Starting processing job {job_id} for {filename} (VLM: {use_vlm})")
        jobs[job_id].status = "processing"
        
        # Parse and chunk document - now returns (chunk_id, content, metadata)
        logger.info(f"Parsing document: {file_path}")
        chunks = docling_parser.parse_document(file_path, use_vlm=use_vlm)
        logger.info(f"Parsed {len(chunks)} chunks from {filename}")
        
        # Generate embeddings and prepare for upload with metadata
        chunks_with_embeddings = []
        for i, (chunk_id, content, meta) in enumerate(chunks):
            logger.info(f"Generating embedding {i+1}/{len(chunks)}")
            embedding = azure_openai.embed_text(content)
            chunks_with_embeddings.append((chunk_id, content, embedding, meta))
        
        # Upload to Azure AI Search (with metadata)
        logger.info(f"Uploading {len(chunks_with_embeddings)} chunks to Azure Search")
        azure_search.upload_chunks(job_id, filename, chunks_with_embeddings)
        
        jobs[job_id].status = "completed"
        jobs[job_id].chunks_count = len(chunks)
        logger.info(f"Job {job_id} completed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        jobs[job_id].status = "failed"
        jobs[job_id].error = str(e)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_vlm: Optional[bool] = Query(
        None,
        description="Use VLM (Vision Language Model) pipeline for PDF processing. "
                    "If not specified, uses USE_VLM_PIPELINE environment variable setting."
    )
):
    """Upload and process a document.
    
    The VLM pipeline uses Azure OpenAI vision-capable models (like GPT-4o) to process
    PDF pages as images, which can be more accurate for complex layouts, handwritten
    content, and documents with mixed content types.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Ensure index exists
    azure_search.ensure_index_exists()
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{job_id}_{file.filename}"
    
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    # Initialize job status
    jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="pending",
        filename=file.filename
    )
    
    # Start background processing
    background_tasks.add_task(
        process_document, job_id, str(file_path), file.filename, use_vlm or False
    )
    
    return DocumentUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status="pending"
    )


@router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("", response_model=list[DocumentInfo])
async def list_documents():
    """List all processed documents."""
    try:
        azure_search.ensure_index_exists()
        docs = azure_search.get_documents()
        return [DocumentInfo(**d) for d in docs]
    except Exception:
        return []


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and its chunks from the index."""
    deleted_count = azure_search.delete_document_chunks(doc_id)
    
    # Clean up job status if exists
    if doc_id in jobs:
        del jobs[doc_id]
    
    return {"deleted_chunks": deleted_count}
