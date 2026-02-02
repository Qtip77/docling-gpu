"""Async document processor worker with concurrent job processing."""
import asyncio
import logging
from typing import Optional

from app.config import settings
from app.models.schemas import JobStatus
from app.services import azure_openai, azure_search, mistral_ocr
from app.services.hierarchical_chunker import chunk_ocr_result, ChunkConfig
from app.workers import queue

logger = logging.getLogger(__name__)

_running = False
_tasks: list[asyncio.Task] = []
_semaphore: Optional[asyncio.Semaphore] = None


def get_chunks(ocr_result, filename: str):
    """Get chunks using configured strategy."""
    if settings.chunking_strategy == "agentic":
        from app.services.agentic_chunker import chunk_ocr_result_agentic, AgenticChunkConfig
        config = AgenticChunkConfig(
            chunk_size=settings.agentic_chunk_size,
            model=settings.agentic_model,
            describe_images=settings.agentic_describe_images,
        )
        return chunk_ocr_result_agentic(ocr_result, filename, config)
    else:
        return chunk_ocr_result(ocr_result, filename, ChunkConfig())


def should_include_images() -> bool:
    """Check if OCR should include image base64 for processing."""
    return (
        settings.chunking_strategy == "agentic" 
        and settings.agentic_describe_images
    )


async def process_document(job_data: dict):
    """Process a single document job."""
    job_id = job_data["job_id"]
    file_path = job_data["file_path"]
    filename = job_data["filename"]

    try:
        await queue.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info(f"Processing {filename} (job: {job_id}) [strategy: {settings.chunking_strategy}]")

        # OCR with Mistral - returns full structured result
        # Include image base64 when using agentic chunking with image descriptions
        ocr_result = await mistral_ocr.process_document(file_path, include_images=should_include_images())
        page_count = len(ocr_result.pages)
        logger.info(f"OCR complete: {page_count} pages")

        # Chunk using configured strategy
        chunks = get_chunks(ocr_result, filename)
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


async def worker_loop(worker_id: int):
    """Worker loop - dequeues and processes jobs with concurrency control."""
    global _running, _semaphore
    logger.info(f"Worker {worker_id} started")

    while _running:
        try:
            job_data = await queue.dequeue_job()
            if job_data:
                # Semaphore limits concurrent OCR/embedding calls across all workers
                async with _semaphore:
                    await process_document(job_data)
        except Exception as e:
            logger.exception(f"Worker {worker_id} error: {e}")
            await asyncio.sleep(1)

    logger.info(f"Worker {worker_id} stopped")


async def start_workers(count: int = None):
    """Start background workers for concurrent document processing."""
    global _running, _tasks, _semaphore
    if _running:
        return

    count = count or settings.max_concurrent_docs
    _running = True
    _semaphore = asyncio.Semaphore(count)
    
    # Start multiple worker tasks
    _tasks = [asyncio.create_task(worker_loop(i)) for i in range(count)]
    logger.info(f"Started {count} worker(s) (max concurrent: {count})")


async def stop_workers():
    """Stop all background workers."""
    global _running, _tasks, _semaphore
    _running = False
    
    for task in _tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    _tasks = []
    _semaphore = None
    await queue.close_redis()
    logger.info("Workers stopped")
