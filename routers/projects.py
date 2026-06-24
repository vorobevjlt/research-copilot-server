from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from database import supabase
from auth import get_current_user

class ProjectCreate(BaseModel):
    name: str
    description: str = ""

class ProjectSettings(BaseModel):
    embedding_model: str
    rag_strategy: str
    agent_type: str
    chunks_per_search: int
    final_context_size: int
    similarity_threshold: float
    number_of_queries: int
    reranking_enabled: bool
    reranking_model: str
    vector_weight: float
    keyword_weight: float

router = APIRouter(
    tags=["projects"]
)

@router.get("/api/projects")
def get_all_projects(clerk_id: str = Depends(get_current_user)):
    try:
        result = supabase.table('projects').select('*').eq('clerk_id', clerk_id).execute()

        return {
            "message": "Projects retrieved successfully",
            "data": result.data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed hard {str(e)}")

@router.post("/api/projects")
def create_project(
    project: ProjectCreate, 
    clerk_id: str = Depends(get_current_user)
    ):
    try:
        project_result = supabase.table("projects").insert({
            "name": project.name,
            "description": project.description,
            "clerk_id": clerk_id
        }).execute()
        if not project_result.data: 
            raise HTTPException(status_code=500, detail=f"Faled create project not project_result.data")

        created_project = project_result.data[0]
        project_id = created_project["id"]

        settings_result = supabase.table("project_settings").insert({
            "project_id": project_id,
            "embedding_model": "text_embedding-3-large",
            "rag_strategy": "basic",
            "agent_type": "agentic",
            "chunks_per_search": 10,
            "final_context_size": 5,
            "similarity_threshold": 0.3,
            "number_of_queries": 5,
            "reranking_enabled": True,
            "reranking_model": "rerank-english-v3.0",
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
        }).execute()

        if not settings_result.data:
            supabase.table("projects").delete().eq("id", project_id).execute()
            raise HTTPException(status_code=500, detail=f"Failed create project, not settings_result.data")

        return {
            "message": "Project created",
            "data": created_project
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Faled create project projects.py {str(e)}")

@router.delete("/api/projects/{project_id}")
def delete_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    try:
        project_result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not project_result.data:
            raise HTTPException(status_code=404, detail=f"Project not found or access denied")
        
        deleted_result = supabase.table("projects").delete().eq("id", project_id).eq("clerk_id", clerk_id).execute()

        if not deleted_result.data:
            raise HTTPException(status_code=404, detail=f"Failed to delete project")

        return {
            "message": "Project deleted",
            "data": project_result.data[0]
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Faled delete project")

# Project API router
@router.get("/api/projects/{project_id}")
async def get_project(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):

    try:
        result = supabase.table("projects").select("*").eq("id", project_id).eq("clerk_id", clerk_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or don't have permission"
            )
        return {
            "success": True,
            "message": "Project retrieved",
            "data": result.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"While retrieving projects: {str(e)}"
        )

# Settings API router
@router.get("/api/projects/{project_id}/settings")
async def get_project_settings(
    project_id: str,
    clerk_id: str = Depends(get_current_user)
):
    
    try:
        project_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()
        if not project_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or don't have permission"
            )

        result = supabase.table("project_settings").select("*").eq("project_id", project_id).execute()
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Settings not found"
            )
        return {
            "success": True,
            "message": "Settings retrieved",
            "data": result.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while retrieving settings: {str(e)}"
        )

@router.put("/api/projects/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    settings: ProjectSettings,
    clerk_id: str = Depends(get_current_user)
):
    
    try:
        p_result = supabase.table("projects").select("id").eq("id", project_id).eq("clerk_id", clerk_id).execute()
        if not p_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or haven`t permission to update"
            )
        s_result = supabase.table("project_settings").update(settings.model_dump()).eq("project_id",project_id).execute()
        if not s_result.data:
            raise HTTPException(
                status_code=404,
                detail="Project not found or haven`t permision to update settings"
            )
        
        return {
            "success": True,
            "message": "Settings updated",
            "data": s_result.data[0]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred while updating project settings: {str(e)}"
        )
