from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from routers import users, database
import os

load_dotenv()

app = FastAPI(
    title="Server",
    description="Server for the project",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)

@app.get("/")
def read_root():
    return {"message": "Server is running!"}

@app.get("/health")
def read_health():
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

@app.post("/create-user")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
