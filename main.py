from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import users, projects, files, chats

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
app.include_router(projects.router)
app.include_router(files.router)
app.include_router(chats.router)

@app.get("/")
def read_root():
    return {"message": "Server is running!"}

@app.get("/health")
def read_health():
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
