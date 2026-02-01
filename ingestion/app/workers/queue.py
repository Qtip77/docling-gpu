"""Redis job queue for document processing (db=1)."""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis

from app.config import settings
from app.models.schemas import JobStatus, ProcessingStatus

logger = logging.getLogger(__name__)

_redis: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None


# Keys
def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _queue_key() -> str:
    return "queue:documents"


async def enqueue_job(job_id: str, filename: str, file_path: str) -> ProcessingStatus:
    """Add job to processing queue."""
    r = await get_redis()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    job_data = {
        "job_id": job_id,
        "filename": filename,
        "file_path": file_path,
        "status": JobStatus.PENDING.value,
        "created_at": now,
        "updated_at": now,
    }

    # Store job state
    await r.hset(_job_key(job_id), mapping=job_data)
    # Push to queue
    await r.lpush(_queue_key(), job_id)

    logger.info(f"Enqueued job {job_id} for {filename}")
    return ProcessingStatus(
        job_id=job_id,
        status=JobStatus.PENDING,
        filename=filename,
        created_at=datetime.fromisoformat(now.rstrip("Z")),
        updated_at=datetime.fromisoformat(now.rstrip("Z")),
    )


async def dequeue_job() -> Optional[dict]:
    """Get next job from queue (blocking)."""
    r = await get_redis()
    result = await r.brpop(_queue_key(), timeout=1)
    if not result:
        return None

    job_id = result[1]
    job_data = await r.hgetall(_job_key(job_id))
    return job_data if job_data else None


async def update_job_status(
    job_id: str,
    status: JobStatus,
    chunks_count: Optional[int] = None,
    error: Optional[str] = None,
):
    """Update job status."""
    r = await get_redis()
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    updates = {"status": status.value, "updated_at": now}
    if chunks_count is not None:
        updates["chunks_count"] = str(chunks_count)
    if error:
        updates["error"] = error

    await r.hset(_job_key(job_id), mapping=updates)
    logger.info(f"Job {job_id} status: {status.value}")


async def get_job_status(job_id: str) -> Optional[ProcessingStatus]:
    """Get job status."""
    r = await get_redis()
    job_data = await r.hgetall(_job_key(job_id))
    if not job_data:
        return None

    return ProcessingStatus(
        job_id=job_data["job_id"],
        status=JobStatus(job_data["status"]),
        filename=job_data["filename"],
        chunks_count=int(job_data["chunks_count"]) if job_data.get("chunks_count") else None,
        error=job_data.get("error"),
        created_at=datetime.fromisoformat(job_data["created_at"].rstrip("Z")) if job_data.get("created_at") else None,
        updated_at=datetime.fromisoformat(job_data["updated_at"].rstrip("Z")) if job_data.get("updated_at") else None,
    )


async def get_batch_status(job_ids: list[str]) -> list[ProcessingStatus]:
    """Get status for multiple jobs."""
    results = []
    for job_id in job_ids:
        status = await get_job_status(job_id)
        if status:
            results.append(status)
    return results
