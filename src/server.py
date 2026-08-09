from sys import version
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.userRoutes import router as userRoutes
from src.routes.projectRoutes import router as projectRoutes
from src.routes.projectFilesRoutes import router as projectFilesRoutes
from src.routes.chatRoutes import router as chatRoutes
from src.config.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__ + " line 10 server.py")

logger.info("init_app", version="1.0.0.0")

# Create FastAPI app
app = FastAPI(
    title="AI Engineering API",
    description="Backend API",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("middleware_config_done")

app.include_router(userRoutes, prefix="/api/user")
app.include_router(projectRoutes, prefix="/api/projects")
app.include_router(projectFilesRoutes, prefix="/api/projects")
app.include_router(chatRoutes, prefix="/api/chats")

logger.info("all_routes_registred")

"""
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
"""
