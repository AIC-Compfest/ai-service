"""
Background Worker for AI Processing Jobs
Processes queued jobs from Supabase
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set environment defaults
_ENV_DEFAULTS = {
    "PAGE_OUTPUT_DIR": "/tmp/ai-service/pages",
    "HF_HOME": "/tmp/huggingface",
    "XDG_CACHE_HOME": "/tmp/cache",
    "TORCH_HOME": "/tmp/torch",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
}
for _name, _value in _ENV_DEFAULTS.items():
    os.environ.setdefault(_name, _value)
if not os.access(os.path.expanduser("~"), os.W_OK):
    os.environ["HOME"] = "/tmp"

from app.core.logging import get_logger
from app.core.runtime import get_pipeline
from app.services import job_queue

logger = get_logger(__name__)


async def process_next_job() -> dict:
    """
    Process the next queued job from Supabase.
    This function is called by Appwrite Functions cron or manual trigger.
    """
    try:
        # Get next job
        job = await job_queue.get_next_queued_job()
        
        if not job:
            return {
                "status": "idle",
                "message": "No jobs in queue",
            }
        
        job_id = job["id"]
        document_id = job["document_id"]
        filename = job["filename"]
        storage_path = job["storage_path"]
        
        logger.info("worker.processing", job_id=job_id, document_id=document_id, filename=filename)
        
        # Update status to PROCESSING
        await job_queue.update_job_status(job_id, "PROCESSING")
        
        try:
            # Download file from storage
            file_bytes = await job_queue.download_file_from_storage("documents", storage_path)
            
            # Process document (this takes 60-120 seconds, but it's OK in async worker)
            pipeline = get_pipeline()
            evidence = await asyncio.to_thread(
                pipeline.process_document,
                file_bytes,
                filename=filename,
                document_id=document_id
            )
            
            # Store result
            await job_queue.update_job_status(
                job_id,
                "COMPLETED",
                result=evidence  # Will be serialized to JSONB
            )
            
            logger.info("worker.completed", job_id=job_id, document_id=document_id)
            
            return {
                "status": "completed",
                "job_id": job_id,
                "document_id": document_id,
                "message": "Job processed successfully",
            }
            
        except Exception as processing_error:
            error_msg = str(processing_error)
            error_detail = traceback.format_exc()
            
            logger.error("worker.processing.error", 
                        job_id=job_id, 
                        error=error_msg,
                        traceback=error_detail)
            
            # Check retry logic
            retry_count = await job_queue.increment_retry_count(job_id)
            max_retries = job.get("max_retries", 2)
            
            if retry_count < max_retries:
                # Requeue for retry
                await job_queue.update_job_status(
                    job_id,
                    "QUEUED",
                    error_message=f"Retry {retry_count}/{max_retries}: {error_msg}"
                )
                logger.info("worker.retry", job_id=job_id, retry_count=retry_count)
                
                return {
                    "status": "retrying",
                    "job_id": job_id,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "message": f"Job failed, retrying ({retry_count}/{max_retries})",
                }
            else:
                # Max retries reached, mark as FAILED
                await job_queue.update_job_status(
                    job_id,
                    "FAILED",
                    error_message=error_msg,
                    error_detail=error_detail
                )
                
                return {
                    "status": "failed",
                    "job_id": job_id,
                    "error": error_msg,
                    "message": "Job failed after max retries",
                }
    
    except Exception as e:
        error_msg = str(e)
        error_detail = traceback.format_exc()
        logger.error("worker.error", error=error_msg, traceback=error_detail)
        
        return {
            "status": "error",
            "error": error_msg,
            "message": "Worker encountered an error",
        }


def main(context: Any) -> Any:
    """
    Appwrite Functions entrypoint for background worker.
    Can be triggered by cron or manually.
    """
    try:
        # Run async function
        result = asyncio.run(process_next_job())
        
        # Return JSON response
        return context.res.json(result, 200)
        
    except Exception as e:
        error_msg = str(e)
        logger.error("worker.main.error", error=error_msg)
        
        return context.res.json({
            "status": "error",
            "error": error_msg,
            "message": "Worker failed to execute",
        }, 500)
