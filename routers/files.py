from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from storage import supabase, s3_client, BUCKET_NAME
from auth import get_current_user
import uuid

router = APIRouter(
    tags=["files"]
)

class FileUploadRequest(BaseModel):
    file_name: str
    file_size: int
    file_type: str

class UrlRequest(BaseModel):
    url: str

class FileConfirmRequest(BaseModel):
    s3_key: str

@router.get("/api/projects/{project_id}/files")
async def get_project_files(
    project_id: str, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        # Get all files for this project - FK constraints ensure project exists and belongs to the user
        result = supabase.table("project_documents").select("*").eq("project_id", project_id).eq("clerk_id", clerk_id).order("created_at", desc=True).execute()

        return {
            "message": "Project files retrieved successfully", 
            "data": result.data or []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to get project files: {str(e)}")

@router.post("/api/projects/{project_id}/files/upload-url")
async def get_upload_url(
    project_id: str,
    file_request: FileUploadRequest,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()
        if not result.data:
            raise HTTPException(status_code=400, detail="Project not found")
            
        file_extension = file_request.file_name.split('.')[-1] if '.' in file_request.file_name else ''
        unique_id = str(uuid.uuid4())
        s3_key = f"projects/{project_id}/documents/{unique_id}"
        if file_extension:
            s3_key = f"{s3_key}.{file_extension}"

        presigned_url = s3_client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": s3_key,
                "ContentType": file_request.file_type or "application/octet-stream",
            },
            ExpiresIn=3600
        )

        doc_result = supabase.table("project_documents").insert({
            "project_id": project_id,
            "filename": file_request.file_name,
            "s3_key": s3_key,
            "file_size": file_request.file_size,
            "file_type": file_request.file_type or "application/octet-stream",
            "processing_status": 'uploading',
            "clerk_id": clerk_id,
        }).execute()

        if not doc_result.data:
            raise HTTPException(status_code=500, detail="Failed to create document record")
        
        return {
            "message": "Uploaded URL generated",
            "data": {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "document": doc_result.data[0],
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to upload files: {str(e)}")

@router.post("/api/projects/{project_id}/files/confirm")
async def confirm_file_upload(
    project_id: str,
    confirm_request: FileConfirmRequest,
    clerk_id: str = Depends(get_current_user)
):
    try: 
        s3_key = confirm_request.s3_key

        if not s3_key:
            raise HTTPException(status_code=400, detail="s3_key required")
        result = supabase.table("project_documents").update({
            "processing_status": "queued"
        }).eq("s3_key", s3_key).eq("project_id", project_id).eq("clerk_id", clerk_id).execute()

        if not result.data:
            raise HTTPException(status_code=400, detail="Document not found")
        
        document = result.data[0]

        return {
            "message": "Upload confirmed, processing started with Celery", 
            "data": document
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Failed to confirm upload: {str(e)}")


@router.post("/api/projects/{project_id}/urls")
async def add_website_url(
    project_id: str, 
    url_request: UrlRequest, 
    clerk_id: str = Depends(get_current_user)
):
    try:
        url = url_request.url.strip()
        if not url.startswith(('http://', 'https://')):
            url = "https://" + url

        result = supabase.table("project_documents").insert({
            "project_id": project_id,
            'filename': url,
            's3_key': "",
            'file_size': 0,
            'file_type': 'text/html',
            'processing_status': 'queued',
            'clerk_id': clerk_id, 
            "source_url": url, 
            "source_type": "url"
        }).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create URL record")

        return {
            "message": "URL added successfully, processing started", 
            "data": result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add URL: {str(e)}")

@router.delete("/api/projects/{project_id}/files/{file_id}")
async def delete_file(
    project_id: str,
    file_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        result = (supabase.table("project_documents")
            .select("*")
            .eq("id", file_id)
            .eq("clerk_id", clerk_id)
            .eq("project_id", project_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_record = result.data[0]
        s3_key = file_record["s3_key"]

        if s3_key:
            try:
                s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
            except Exception as e:
                print(f"Failed to delete from s3: {e}")
        
        result = (
            supabase.table("project_documents")
            .delete()
            .eq("id", file_id)
            .eq("clerk_id", clerk_id)
            .eq("project_id", project_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to delete")

        return {
            "message": "File deleted successfully",
            "data": result.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
