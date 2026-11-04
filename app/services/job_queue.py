"""
Job Queue Service for Async AI Processing
Integrates with Supabase for job management and storage
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from supabase import create_client, Client
from app.core.logging import get_logger

logger = get_logger(__name__)

# Singleton Supabase client
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client"""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        _supabase_client = create_client(url, key)
    return _supabase_client


async def create_job(
    job_id: str,
    document_id: str,
    filename: str,
    file_size: int,
    mime_type: str,
    storage_path: str,
    declared_document_type: Optional[str] = None,
    correlation_id: Optional[str] = None,
    ingest_batch_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new processing job record in Supabase.
    
    Args:
        job_id: Unique job identifier
        document_id: Document ID (may not exist yet in documents table)
        filename: Original filename
        file_size: File size in bytes
        mime_type: MIME type
        storage_path: Path in Supabase Storage
        declared_document_type: Optional document type hint
        correlation_id: Optional correlation ID for tracing
        ingest_batch_id: Optional batch ID
        
    Returns:
        Created job record
    """
    try:
        supabase = get_supabase_client()
        
        job_data = {
            "id": job_id,
            "document_id": document_id,
            "status": "QUEUED",
            "filename": filename,
            "file_size": file_size,
            "mime_type": mime_type,
            "storage_path": storage_path,
            "declared_document_type": declared_document_type,
            "correlation_id": correlation_id,
            "ingest_batch_id": ingest_batch_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        response = supabase.table("ai_processing_jobs").insert(job_data).execute()
        logger.info("job.created", job_id=job_id, document_id=document_id)
        return response.data[0] if response.data else job_data
        
    except Exception as e:
        logger.error("job.create.error", job_id=job_id, error=str(e))
        raise


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job by ID"""
    try:
        supabase = get_supabase_client()
        response = supabase.table("ai_processing_jobs").select("*").eq("id", job_id).single().execute()
        return response.data if response.data else None
    except Exception as e:
        logger.error("job.get.error", job_id=job_id, error=str(e))
        return None


async def get_next_queued_job() -> Optional[Dict[str, Any]]:
    """Get oldest queued job for processing"""
    try:
        supabase = get_supabase_client()
        response = (
            supabase.table("ai_processing_jobs")
            .select("*")
            .eq("status", "QUEUED")
            .order("created_at")
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except Exception as e:
        logger.error("job.get_next.error", error=str(e))
        return None


async def update_job_status(
    job_id: str,
    status: str,
    **kwargs
) -> None:
    """
    Update job status and optional fields.
    
    Args:
        job_id: Job ID
        status: New status (QUEUED, PROCESSING, COMPLETED, FAILED)
        **kwargs: Additional fields to update (started_at, completed_at, error_message, result, etc.)
    """
    try:
        supabase = get_supabase_client()
        
        update_data = {"status": status}
        
        # Auto-set started_at when status changes to PROCESSING
        if status == "PROCESSING" and "started_at" not in kwargs:
            update_data["started_at"] = datetime.utcnow().isoformat()
        
        # Merge additional fields
        update_data.update(kwargs)
        
        supabase.table("ai_processing_jobs").update(update_data).eq("id", job_id).execute()
        logger.info("job.status.updated", job_id=job_id, status=status)
        
    except Exception as e:
        logger.error("job.update.error", job_id=job_id, error=str(e))
        raise


async def upload_file_to_storage(bucket: str, path: str, file_bytes: bytes) -> str:
    """Upload file to Supabase Storage and return path"""
    try:
        supabase = get_supabase_client()
        supabase.storage.from_(bucket).upload(path, file_bytes, {"content-type": "application/octet-stream"})
        logger.info("storage.uploaded", bucket=bucket, path=path, size=len(file_bytes))
        return path
    except Exception as e:
        logger.error("storage.upload.error", bucket=bucket, path=path, error=str(e))
        raise


async def download_file_from_storage(bucket: str, path: str) -> bytes:
    """Download file from Supabase Storage"""
    try:
        supabase = get_supabase_client()
        data = supabase.storage.from_(bucket).download(path)
        logger.info("storage.downloaded", bucket=bucket, path=path, size=len(data))
        return data
    except Exception as e:
        logger.error("storage.download.error", bucket=bucket, path=path, error=str(e))
        raise


async def increment_retry_count(job_id: str) -> int:
    """Increment retry count and return new count"""
    try:
        supabase = get_supabase_client()
        
        # Get current job
        job = await get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        new_count = job.get("retry_count", 0) + 1
        
        supabase.table("ai_processing_jobs").update({"retry_count": new_count}).eq("id", job_id).execute()
        
        logger.info("job.retry.incremented", job_id=job_id, retry_count=new_count)
        return new_count
        
    except Exception as e:
        logger.error("job.retry.error", job_id=job_id, error=str(e))
        raise
