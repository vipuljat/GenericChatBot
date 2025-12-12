from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn  # Add this import
from routes.chatbot_routes import router as chatbot_router
from routes.auth_routes import router as auth_router
from stateful_services.database import check_db_health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Generic ChatBot API",
    description="API for creating custom chatbots with document embeddings",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    logging.info("Starting application...")
    if not check_db_health():
        logging.error("Database connection failed! Application startup aborted.")
    else:
        logging.info("Application started successfully and database is healthy.")

@app.get("/")
def read_root():
    return {
        "message": "Generic ChatBot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(chatbot_router, prefix="/chatbot", tags=["Chatbot"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])


if __name__ == "__main__":
    # Runs the FastAPI app with uvicorn when executing: python3 main.py
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True, )
