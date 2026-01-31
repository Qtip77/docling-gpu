"""Async document processor worker."""
import asyncio
import logging
from typing import Optional

from app.models.schemas import JobStatus
from app.services import azure_openai, azure_search, hierarchical_chunker, mistral_ocr
from app.workers import queue

logger = logging.getLogger(__name__)

_running = False
_task: Optional[asyncio.Task] = None


async def process_document(job_data: dict):
    """Process a single document job."""
    job_id = job_data["job_id"]
    file_path = job_data["file_path"]
    filename = job_data["filename"]

    try:
        await queue.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info(f"Processing {filename} (job: {job_id})")

        # OCR with Mistral
        markdown, page_count = await mistral_ocr.process_document(file_path)
        logger.info(f"OCR complete: {page_count} pages, {len(markdown)} chars")

        # Chunk markdown
        chunks = hierarchical_chunker.chunk_markdown(markdown, filename)
        logger.info(f"Chunked into {len(chunks)} segments")

        if not chunks:
            await queue.update_job_status(job_id, JobStatus.COMPLETED, chunks_count=0)
            return

        # Batch embeddings
        texts = [c.content for c in chunks]
        embeddings = await azure_openai.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Ensure index exists
        azure_search.ensure_index_exists()

        # Bulk upload to Azure Search
        await azure_search.upload_chunks_bulk(job_id, filename, chunks)

        await queue.update_job_status(job_id, JobStatus.COMPLETED, chunks_count=len(chunks))
        logger.info(f"Job {job_id} completed: {len(chunks)} chunks indexed")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await queue.update_job_status(job_id, JobStatus.FAILED, error=str(e))


async def worker_loop():
    """Main worker loop - processes jobs from queue."""
    global _running
    logger.info("Worker started")

    while _running:
        try:
            job_data = await queue.dequeue_job()
            if job_data:
                await process_document(job_data)
        except Exception as e:
            logger.exception(f"Worker error: {e}")
            await asyncio.sleep(1)

    logger.info("Worker stopped")


async def start_workers(count: int = 1):
    """Start background workers."""
    global _running, _task
    if _running:
        return

    _running = True
    # For simplicity, run single worker. Can extend to multiple.
    _task = asyncio.create_task(worker_loop())
    logger.info(f"Started {count} worker(s)")


async def stop_workers():
    """Stop background workers."""
    global _running, _task
    _running = False
    if _task:
        _task.cancel()
        try:
            await _task
        except asyncio.CancelledError:
            pass
        _task = None
    await queue.close_redis()
    logger.info("Workers stopped")
