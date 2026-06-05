from fastapi import APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import supabase

router = APIRouter(
    tags=["users"]
)

@router .post("/create-user")
async def clerk_hook(webhook_data: dict):
    try:
        event_type = webhook_data.get("type")
        if event_type == "user.created":
            user_data = webhook_data.get("data", {})
            clerk_id = webhook_data.get("id")
        if not clerk_id:
            raise HTTPException(status_code=400, detail=f"no user id {str(e)}")
        
        result = supabase.table('users').insert({
            "clerk_id": clerk_id
        }).execute()

        return {
            "message": "user created",
            "data": result.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed {str(e)}")

