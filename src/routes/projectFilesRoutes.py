import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException

from src.config.index import appConfig
from src.models.index import (
    ConfirmFileUploadRequest,
    FileUploadRequest,
    ProcessingStatus,
    UrlRequest,
)
from src.services.awsS3 import s3_client
from src.services.celery import perform_rag_ingestion_task
from src.services.clerkAuth import get_current_user_clerk_id
from src.services.supabase import supabase
from src.utils.index import validate_url

router = APIRouter(tags=["projectFilesRoutes"])

CONFIRMED_PROCESSING_STATUSES = {
    ProcessingStatus.QUEUED.value,
    ProcessingStatus.PROCESSING.value,
    ProcessingStatus.PARTITIONING.value,
    ProcessingStatus.CHUNKING.value,
    ProcessingStatus.SUMMARISING.value,
    ProcessingStatus.VECTORIZATION.value,
    ProcessingStatus.COMPLETED.value,
}


def mark_upload_failed(document: dict, message: str):
    """Persist an actionable upload error without hiding the document."""
    processing_details = document.get("processing_details") or {}
    processing_details["error"] = {"message": message}
    supabase.table("project_documents").update(
        {
            "processing_status": ProcessingStatus.FAILED.value,
            "processing_details": processing_details,
        }
    ).eq("id", document["id"]).execute()


def expire_stale_uploads(documents: list[dict]) -> list[dict]:
    """Stop polling abandoned uploads once their one-hour signed URL has expired."""
    expiry_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
    for document in documents:
        if document.get("processing_status") not in {
            ProcessingStatus.UPLOADING.value,
            ProcessingStatus.PENDING.value,
        }:
            continue

        created_at = document.get("created_at")
        if not created_at:
            continue
        try:
            created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=timezone.utc)
        if created_time > expiry_cutoff:
            continue

        message = "Upload expired before it was confirmed"
        mark_upload_failed(document, message)
        document["processing_status"] = ProcessingStatus.FAILED.value
        details = document.get("processing_details") or {}
        details["error"] = {"message": message}
        document["processing_details"] = details
    return documents

"""
`/api/projects`

  - GET `/{project_id}/files` ~ List all project files
  - POST `/{project_id}/files/upload-url` ~ Generate presigned url for file upload for frontend
  - POST `/{project_id}/files/confirm` ~ Confirmation of file upload to S3
  - POST `/{project_id}/urls` ~ Add website URL to database
  - DELETE `/{project_id}/files/{file_id}` ~ Delete document from s3 and database
  - GET `/{project_id}/files/{file_id}/chunks` ~ Get project document chunks
"""


@router.get("/{project_id}/files")
async def get_project_files(
    project_id: str, current_user_clerk_id: str = Depends(get_current_user_clerk_id)
):
    """
    ! Logic Flow
    * 1. Get current user clerk_id
    * 2. Select all project documents from the project documents table for given project_id
    * 3. Return project documents data
    """
    try:
        project_files_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .order("created_at", desc=True)
            .execute()
        )

        # * If there are no project documents for the project, return an empty list
        # * A User may or may not have any project files.

        project_documents = expire_stale_uploads(project_files_result.data or [])
        return {
            "message": "Project files retrieved successfully",
            "data": project_documents,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrieving project {project_id} files: {str(e)}",
        )


@router.post("/{project_id}/files/upload-url")
async def get_upload_presigned_url(
    project_id: str,
    file_upload_request: FileUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify project exists and belongs to the current user
    * 2. Generate s3 key
    * 3. Generate upload presigned url (will expire in 1 hour)
    * 4. Create project document record with uploading status
    * 5. Return presigned url
    """
    try:
        # Verify project exists and belongs to the current user
        project_ownership_verification_result = (
            supabase.table("projects")
            .select("id")
            .eq("id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not project_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or you don't have permission to upload files to this project",
            )

        # Generate s3 key
        file_extension = Path(file_upload_request.filename).suffix.lower().lstrip(".")
        unique_file_id = uuid.uuid4()
        s3_key = (
            f"projects/{project_id}/documents/{unique_file_id}.{file_extension}"
            if file_extension
            else f"projects/{project_id}/documents/{unique_file_id}"
        )

        # Generate upload presigned url (will expire in 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": appConfig["s3_bucket_name"],
                "Key": s3_key,
                "ContentType": file_upload_request.file_type,
                "ContentLength": file_upload_request.file_size,
            },
            ExpiresIn=3600,  # 1 hour
        )

        if not presigned_url:
            raise HTTPException(
                status_code=422,
                detail="Failed to generate upload presigned url",
            )

        # Generate database record with uploading status
        document_creation_result = (
            supabase.table("project_documents")
            .insert(
                {
                    "project_id": project_id,
                    "filename": file_upload_request.filename,
                    "s3_key": s3_key,
                    "file_size": file_upload_request.file_size,
                    "file_type": file_upload_request.file_type,
                    "processing_status": ProcessingStatus.UPLOADING.value,
                    "clerk_id": current_user_clerk_id,
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project document - invalid data provided",
            )

        return {
            "message": "Upload presigned url generated successfully",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "content_type": file_upload_request.file_type,
                "document": document_creation_result.data[0],
            },
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while generating upload presigned url for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/files/confirm")
async def confirm_file_upload_to_s3(
    project_id: str,
    confirm_file_upload_request: ConfirmFileUploadRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify S3 key is provided
    * 2. Verify file exists in database
    * 3. Update file status to "queued"
    * 4. Perform Celery - RAG Ingestion Task
    * 5. Update the project document record with the task_id
    * 6. Return successfully confirmed file upload data
    """
    try:
        s3_key = confirm_file_upload_request.s3_key

        # Verify file exists in database
        document_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("s3_key", s3_key)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="File not found or you don't have permission to confirm upload to S3 for this file",
            )

        document = document_verification_result.data[0]
        if document["processing_status"] in CONFIRMED_PROCESSING_STATUSES:
            return {
                "message": "File upload was already confirmed",
                "data": document,
            }

        if document["processing_status"] == ProcessingStatus.FAILED.value:
            raise HTTPException(
                status_code=409,
                detail="This upload has failed. Remove it and upload the file again.",
            )

        if document["processing_status"] not in {
            ProcessingStatus.UPLOADING.value,
            ProcessingStatus.PENDING.value,
        }:
            raise HTTPException(
                status_code=409,
                detail="This file is not waiting for upload confirmation",
            )

        # Do not queue work until S3 confirms that the complete expected object exists.
        try:
            object_metadata = s3_client.head_object(
                Bucket=appConfig["s3_bucket_name"], Key=s3_key
            )
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404", "NotFound"}:
                message = "The uploaded file was not found in storage"
                mark_upload_failed(document, message)
                raise HTTPException(status_code=409, detail=message) from error
            raise

        actual_size = object_metadata.get("ContentLength")
        if actual_size != document["file_size"]:
            message = (
                f"Uploaded file size mismatch: expected {document['file_size']} bytes, "
                f"received {actual_size} bytes"
            )
            mark_upload_failed(document, message)
            raise HTTPException(status_code=422, detail=message)

        actual_type = (object_metadata.get("ContentType") or "").lower()
        if actual_type != document["file_type"].lower():
            message = (
                f"Uploaded file type mismatch: expected {document['file_type']}, "
                f"received {actual_type or 'no content type'}"
            )
            mark_upload_failed(document, message)
            raise HTTPException(status_code=422, detail=message)

        # Update file status to "queued"
        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": ProcessingStatus.QUEUED.value,
                }
            )
            .eq("s3_key", s3_key)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to queue the uploaded document",
            )

        # ! Celery - Starts Background Processing - RAG Ingestion Task
        document_id = document_update_result.data[0]["id"]
        try:
            task_result = perform_rag_ingestion_task.delay(document_id)
        except Exception as error:
            message = f"Failed to queue document processing: {str(error)}"
            mark_upload_failed(document_update_result.data[0], message)
            raise HTTPException(status_code=503, detail=message) from error
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not document_update_result.data:
            # The task is already queued, so do not turn a metadata-only failure
            # into a client retry that could enqueue the same work twice.
            confirmed_document = {
                **document_verification_result.data[0],
                "processing_status": ProcessingStatus.QUEUED.value,
                "task_id": task_id,
            }
        else:
            confirmed_document = document_update_result.data[0]

        return {
            "message": "File upload to S3 confirmed successfully And Started Background Pre-Processing of this file",
            "data": confirmed_document,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while confirming upload to S3 for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/urls")
async def process_url(
    project_id: str,
    url: UrlRequest,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Validate URL
    * 2. Add website URL to database
    * 3. Start background pre-processing of this URL
    * 4. Return successfully processed URL data
    """
    try:
        # Validate URL
        url = url.url
        if url.startswith("http://") or url.startswith("https://"):
            url = url
        else:
            url = f"https://{url}"

        if not validate_url(url):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL",
            )

        # Add website Url to database
        document_creation_result = (
            supabase.table("project_documents")
            .insert(
                {
                    "project_id": project_id,
                    "filename": url,
                    "s3_key": "",
                    "file_size": 0,
                    "file_type": "text/html",
                    "processing_status": ProcessingStatus.QUEUED,
                    "clerk_id": current_user_clerk_id,
                    "source_type": "url",
                    "source_url": url,
                }
            )
            .execute()
        )

        if not document_creation_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to create project document with URL Record - invalid data provided",
            )

        # ! Celery - Starts Background Processing - RAG Ingestion Task
        document_id = document_creation_result.data[0]["id"]
        task_result = perform_rag_ingestion_task.delay(document_id)
        task_id = task_result.id

        document_update_result = (
            supabase.table("project_documents")
            .update(
                {
                    "task_id": task_id,
                }
            )
            .eq("id", document_id)
            .execute()
        )

        if not document_update_result.data:
            raise HTTPException(
                status_code=422,
                detail="Failed to update project document record with task_id",
            )

        return {
            "message": "Website URL added to database successfully And Started Background Pre-Processing of this URL",
            "data": document_creation_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while processing urls for {project_id}: {str(e)}",
        )


@router.post("/{project_id}/files/{file_id}/retry")
async def retry_project_document_processing(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """Retry a failed ingestion job without uploading the source again."""
    try:
        document_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not document_result.data:
            raise HTTPException(status_code=404, detail="Document not found")

        document = document_result.data[0]
        if document["processing_status"] != ProcessingStatus.FAILED.value:
            raise HTTPException(
                status_code=409, detail="Only failed documents can be retried"
            )

        if document["source_type"] == "file":
            try:
                object_metadata = s3_client.head_object(
                    Bucket=appConfig["s3_bucket_name"], Key=document["s3_key"]
                )
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code")
                if error_code in {"NoSuchKey", "404", "NotFound"}:
                    raise HTTPException(
                        status_code=409,
                        detail="The original file is missing; upload it again",
                    ) from error
                raise

            if object_metadata.get("ContentLength") != document["file_size"]:
                raise HTTPException(
                    status_code=409,
                    detail="The stored file is incomplete; upload it again",
                )
            if (object_metadata.get("ContentType") or "").lower() != document[
                "file_type"
            ].lower():
                raise HTTPException(
                    status_code=409,
                    detail="The stored file type is invalid; upload it again",
                )

        # Remove any chunks written before the previous task failed.
        supabase.table("document_chunks").delete().eq(
            "document_id", file_id
        ).execute()

        queued_result = (
            supabase.table("project_documents")
            .update(
                {
                    "processing_status": ProcessingStatus.QUEUED.value,
                    "processing_details": {},
                    "task_id": None,
                }
            )
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )
        if not queued_result.data:
            raise HTTPException(status_code=422, detail="Failed to retry document")

        try:
            task_result = perform_rag_ingestion_task.delay(file_id)
        except Exception as error:
            message = f"Failed to queue document processing: {str(error)}"
            mark_upload_failed(queued_result.data[0], message)
            raise HTTPException(status_code=503, detail=message) from error

        updated_result = (
            supabase.table("project_documents")
            .update({"task_id": task_result.id})
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        return {
            "message": "Document processing queued again",
            "data": (
                updated_result.data[0]
                if updated_result.data
                else {
                    **queued_result.data[0],
                    "task_id": task_result.id,
                }
            ),
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while retrying document {file_id}: {str(e)}",
        )


@router.delete("/{project_id}/files/{file_id}")
async def delete_project_document(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify document exists and belongs to the current user and take complete project document record
    * 2. Delete file from S3 (only for actual files, not for URLs)
    * 3. Delete document from database
    * 4. Return successfully deleted document data
    """
    try:
        # Verify document exists and belongs to the current user and Take complete project document record
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        # Delete file from S3 (only for actual files, not for URLs)
        document = document_ownership_verification_result.data[0]
        s3_key = document["s3_key"]
        if s3_key:
            try:
                s3_client.delete_object(Bucket=appConfig["s3_bucket_name"], Key=s3_key)
            except ClientError as error:
                error_code = error.response.get("Error", {}).get("Code")
                is_failed_upload = document.get("processing_status") in {
                    ProcessingStatus.UPLOADING,
                    ProcessingStatus.PENDING,
                }
                if not is_failed_upload or error_code not in {
                    "AccessDenied",
                    "NoSuchKey",
                    "404",
                }:
                    raise

        # Delete document from database
        document_deletion_result = (
            supabase.table("project_documents")
            .delete()
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_deletion_result.data:
            raise HTTPException(
                status_code=404,
                detail="Failed to delete document",
            )

        return {
            "message": "Document deleted successfully",
            "data": document_deletion_result.data[0],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while deleting project document {file_id} for {project_id}: {str(e)}",
        )


@router.get("/{project_id}/files/{file_id}/chunks")
async def get_project_document_chunks(
    project_id: str,
    file_id: str,
    current_user_clerk_id: str = Depends(get_current_user_clerk_id),
):
    """
    ! Logic Flow:
    * 1. Verify document exists and belongs to the current user and Take complete project document record
    * 2. Get project document chunks
    * 3. Return project document chunks data
    """
    try:
        # Verify document exists and belongs to the current user and Take complete project document record
        document_ownership_verification_result = (
            supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("project_id", project_id)
            .eq("clerk_id", current_user_clerk_id)
            .execute()
        )

        if not document_ownership_verification_result.data:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete this document",
            )

        document_chunks_result = (
            supabase.table("document_chunks")
            .select("*")
            .eq("document_id", file_id)
            .order("chunk_index")
            .execute()
        )

        return {
            "message": "Project document chunks retrieved successfully",
            "data": document_chunks_result.data or [],
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while getting project document chunks for {file_id} for {project_id}: {str(e)}",
        )
