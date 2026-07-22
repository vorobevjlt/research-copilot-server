from fastapi import APIRouter, HTTPException
from storage import supabase

router = APIRouter(
    tags=["users"]
)

@router.post("/create-user")
async def clerk_hook(webhook_data: dict):
    event_type = webhook_data.get("type")

    if event_type != "user.created":
        return {"message": "event ignored", "type": event_type}

    clerk_id = webhook_data.get("data", {}).get("id")

    if not clerk_id:
        raise HTTPException(status_code=400, detail="No user ID")

    try:
        result = (
            supabase.table("users")
            .upsert(
                {"clerk_id": clerk_id},
                on_conflict="clerk_id",
            )
            .execute()
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Supabase insert failed: {exc}",
        ) from exc

    return {
        "message": "user created",
        "data": result.data[0] if result.data else None,
    }
