"""
FastAPI Application
-------------------
Endpoints:
  POST /process          Upload PDF → returns job_id (queued via Celery)
  GET  /job/{job_id}     Poll job status + result
  GET  /images/{filename} Serve locally stored cropped images
  GET  /health           Health check
"""

import os
import uuid
import hashlib
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exam PDF → JSON Pipeline",
    description="PupilTree.ai intern assessment — PDF to structured JSON",
    version="1.0.0",
)

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
IMAGE_DIR = OUTPUT_DIR / "images"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Serve local images as static files
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")


# ── helpers ───────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF upload and enqueue it for processing.

    Returns:
        job_id: use this to poll /job/{job_id} for results
        status: "queued"
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = str(uuid.uuid4())

    # Save upload
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Compute hash for idempotency
    pdf_hash = _sha256_file(pdf_path)

    # Enqueue Celery task
    try:
        from workers.celery_app import process_pdf_task
        task = process_pdf_task.apply_async(
            args=[str(pdf_path), job_id, pdf_hash],
            task_id=job_id,
        )
        logger.info(f"Enqueued job {job_id} | hash={pdf_hash[:8]}...")
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")

    return JSONResponse({
        "job_id": job_id,
        "pdf_hash": pdf_hash,
        "status": "queued",
        "poll_url": f"/job/{job_id}",
    })


@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    """
    Poll job status and result.

    Returns:
        status: "queued" | "started" | "success" | "failure" | "retry"
        result: full JSON output (only when status == "success")
    """
    try:
        from workers.celery_app import celery_app as celery
        task = celery.AsyncResult(job_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")

    state = task.state  # PENDING, STARTED, SUCCESS, FAILURE, RETRY

    if state == "SUCCESS":
        return JSONResponse({
            "job_id": job_id,
            "status": "success",
            "result": task.result,
        })
    elif state == "FAILURE":
        return JSONResponse({
            "job_id": job_id,
            "status": "failure",
            "error": str(task.result),
        }, status_code=500)
    elif state == "STARTED":
        return JSONResponse({"job_id": job_id, "status": "started"})
    elif state == "RETRY":
        return JSONResponse({"job_id": job_id, "status": "retry"})
    else:
        return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/job/{job_id}/download")
def download_result(job_id: str):
    """Download the output JSON file directly."""
    out_path = OUTPUT_DIR / f"{job_id}.json"
    if not out_path.exists():
        raise HTTPException(status_code=404, detail="Result not ready yet.")
    return FileResponse(
        path=str(out_path),
        media_type="application/json",
        filename=f"{job_id}_output.json",
    )
