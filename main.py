from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from routes.chatbot_routes import router as chatbot_router
from routes.auth_routes import router as auth_router
from stateful_services.database import check_db_health
from services.qdrant_service import qdrant_healthcheck
from fastapi.openapi.docs import get_redoc_html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Create app once ---
app = FastAPI(
    title="Generic ChatBot API",
    description="API for creating custom chatbots with document embeddings",
    version="1.0.0",
    docs_url='/api/v1/docs',
    redoc_url=None,
    openapi_url='/api/v1/openapi.json'
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, restrict to your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# --- Startup event ---
@app.on_event("startup")
def on_startup():
    logger.info("Starting application...")

    # Database health check
    if not check_db_health():
        logger.warning("⚠️ Database connection failed! Some features may not work.")
    else:
        logger.info("Database is healthy.")

    # # Qdrant health check
    # if not qdrant_healthcheck():
    #     logger.warning("⚠️ Qdrant health check failed! Some features may not work.")
    # else:
    #     logger.info("Qdrant is healthy.")

    # logger.info("Application startup complete.")

# --- Routes ---
@app.get("/")
def read_root():
    return {
        "message": "Generic ChatBot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/v1/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js",
    )

@app.get("/health")
def health_check():
    """Check health of both database and Qdrant"""
    db_status = "healthy" if check_db_health() else "unhealthy"
    qdrant_status = "healthy" if qdrant_healthcheck() else "unhealthy"
    return {
        "database": db_status,
        "qdrant": qdrant_status
    }

# Include routers
app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# --- Run Uvicorn ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
