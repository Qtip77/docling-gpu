"""Document upload and status endpoints."""
import logging
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.models.schemas import (
    BatchStatusRequest,
    BatchStatusResponse,
    BulkUploadResponse,
    DocumentInfo,
    DocumentUploadResponse,
    ProcessingStatus,
)
from app.services import azure_search
from app.workers import queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a single document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    job_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{job_id}_{file.filename}"

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    status = await queue.enqueue_job(job_id, file.filename, str(file_path))
    logger.info(f"Uploaded {file.filename} -> job {job_id}")

    return DocumentUploadResponse(
        job_id=job_id,
        filename=file.filename,
        status=status.status,
    )


@router.post("/bulk", response_model=BulkUploadResponse)
async def bulk_upload(files: list[UploadFile] = File(...)):
    """Upload multiple documents (up to 20)."""
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per bulk upload")

    jobs = []
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename:
            continue

        job_id = str(uuid.uuid4())
        file_path = upload_dir / f"{job_id}_{file.filename}"

        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        status = await queue.enqueue_job(job_id, file.filename, str(file_path))
        jobs.append(
            DocumentUploadResponse(
                job_id=job_id,
                filename=file.filename,
                status=status.status,
            )
        )

    logger.info(f"Bulk uploaded {len(jobs)} files")
    return BulkUploadResponse(jobs=jobs)


@router.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_status(job_id: str):
    """Get processing status for a job."""
    status = await queue.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.post("/batch-status", response_model=BatchStatusResponse)
async def batch_status(request: BatchStatusRequest):
    """Get status for multiple jobs."""
    statuses = await queue.get_batch_status(request.job_ids)
    return BatchStatusResponse(statuses=statuses)


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
    """Delete a document and its chunks."""
    deleted = azure_search.delete_document_chunks(doc_id)
    return {"deleted_chunks": deleted}


@router.post("/compare-chunking/{job_id}")
async def compare_chunking(job_id: str):
    """
    Compare hierarchical vs agentic chunking on an uploaded document.
    Useful for evaluating which strategy works best for your document types.
    """
    from pathlib import Path
    from app.services.mistral_ocr import process_document as ocr_process
    from app.services.agentic_chunker import compare_chunking_strategies
    
    # Find the file
    upload_dir = Path(settings.upload_dir)
    matches = list(upload_dir.glob(f"{job_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = matches[0]
    filename = file_path.name.split("_", 1)[1] if "_" in file_path.name else file_path.name
    
    # Run OCR
    ocr_result = await ocr_process(str(file_path))
    
    # Compare strategies
    comparison = await compare_chunking_strategies(ocr_result, filename)
    
    return {
        "filename": filename,
        "pages": len(ocr_result.pages),
        "comparison": comparison,
    }
