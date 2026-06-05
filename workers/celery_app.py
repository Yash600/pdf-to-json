"""
Celery App + Worker Task
------------------------
Queue design:
  - Broker + backend: Redis (local Docker or Redis Cloud free tier)
  - Concurrency: MAX_WORKERS (default 3) — respects Groq 30 RPM free tier
  - Idempotency: PDF SHA-256 hash used as task ID prefix; result cached in Redis
  - Retries: 3 attempts, exponential backoff (60s, 120s, 240s)
  - Backpressure: task_queue_max_priority limits queue depth
  - Soft time limit: 80s warning, hard limit 100s (< 1.5 min SLA)

Scaling plan (documented in README):
  - Free tier (this config): 3 workers, ~1 PDF/min throughput
  - Paid Groq + more workers: 10 workers → ~100 PDFs in 8-10 min
  - Horizontal: run multiple Celery worker processes on separate machines
    all pointing to same Redis broker
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from celery import Celery
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))

# ── Celery app ────────────────────────────────────────────────────────────────

celery_app = Celery(
    "pdf_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Concurrency
    worker_concurrency=MAX_WORKERS,

    # Time limits — hard 100s to stay under 1.5 min SLA
    task_soft_time_limit=80,
    task_time_limit=100,

    # Result expiry — keep results for 24 hours
    result_expires=86400,

    # Retry settings
    task_acks_late=True,            # only ack after task completes (safer)
    task_reject_on_worker_lost=True,

    # Backpressure: limit prefetch so queue depth is visible
    worker_prefetch_multiplier=1,   # one task at a time per worker slot

    # Track task states
    task_track_started=True,
)

task_logger = get_task_logger(__name__)



# ── Idempotency cache ─────────────────────────────────────────────────────────

def _result_key(pdf_hash: str) -> str:
    return f"pipeline:result:{pdf_hash}"


def _get_cached_result(redis_client, pdf_hash: str):
    import json
    raw = redis_client.get(_result_key(pdf_hash))
    if raw:
        return json.loads(raw)
    return None


def _cache_result(redis_client, pdf_hash: str, result: dict):
    import json
    redis_client.setex(
        _result_key(pdf_hash),
        86400,  # TTL 24h
        json.dumps(result, ensure_ascii=False),
    )


# ── Celery task ───────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="pipeline.process_pdf",
    max_retries=3,
    default_retry_delay=60,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=240,
)
def process_pdf_task(self, pdf_path: str, job_id: str, pdf_hash: str):
    """
    Celery task: process a single PDF.

    Args:
        pdf_path: absolute path to the uploaded PDF
        job_id: unique job identifier (UUID)
        pdf_hash: SHA-256 of the PDF (for idempotency)
    """
    import redis as redis_lib
    import json

    r = redis_lib.from_url(REDIS_URL)

    # Idempotency check
    cached = _get_cached_result(r, pdf_hash)
    if cached:
        task_logger.info(f"[{job_id}] Cache hit for hash {pdf_hash[:8]}... — returning cached result")
        return cached

    task_logger.info(f"[{job_id}] Starting pipeline for {pdf_path}")

    try:
        from app.pipeline import run_pipeline
        result = run_pipeline(pdf_path, job_id)
        _cache_result(r, pdf_hash, result)
        task_logger.info(f"[{job_id}] Completed. {result['question_count']} questions extracted.")
        return result

    except Exception as exc:
        task_logger.error(f"[{job_id}] Task failed: {exc}")
        raise self.retry(exc=exc)
