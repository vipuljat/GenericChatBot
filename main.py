"""FastAPI application entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html

from routes.chatbot_routes import router as chatbot_router
from routes.chatbot import router as chatbot
from routes.auth_routes import router as auth_router
from routes.employee_routes import router as employee_router
from stateful_services.question import router as question_router
from stateful_services.answer import router as answer_router
from routes.people_analyzer_routes import router as people_analyzer_router
from stateful_services.database import check_all_health
from utils.logging import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    log.info("Starting application...")
    
    health_status = check_all_health()
    
    for service, is_healthy in health_status.items():
        if not is_healthy:
            log.warning(f"⚠️ {service.capitalize()} health check failed!")
        else:
            log.info(f"{service.capitalize()} is healthy")
    
    log.info("Application startup complete")
    
    yield
    
    # Shutdown
    log.info("Shutting down application...")


def create_app() -> FastAPI:
    """Application factory for creating FastAPI instance."""
    application = FastAPI(
        title="Generic ChatBot API",
        description="API for creating custom chatbots with document embeddings",
        version="1.0.0",
        docs_url="/api/v1/docs",
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan
    )
    
    # Configure CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    application.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])
    application.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    application.include_router(chatbot, prefix="/chatbot/v2", tags=["Chatbot v2"])
    application.include_router(question_router, prefix="/quiz", tags=["Questions"])
    application.include_router(answer_router, prefix="/quiz", tags=["Answers"])
    application.include_router(people_analyzer_router)
    application.include_router(employee_router, prefix="/employees", tags=["Employees"])

    
    @application.get("/")
    def read_root():
        """Root endpoint returning API information."""
        return {
            "message": "Generic ChatBot API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @application.get("/health")
    def health_check():
        """Health check endpoint for monitoring."""
        health = check_all_health()
        all_healthy = all(health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": health
        }
    
    @application.get("/api/v1/redoc", include_in_schema=False)
    async def redoc_html():
        """ReDoc documentation endpoint."""
        return get_redoc_html(
            openapi_url=application.openapi_url or "/api/v1/openapi.json",
            title=application.title + " - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js",
        )
    
    return application


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use your custom logging configuration
    )