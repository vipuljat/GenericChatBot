from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import Base, engine  
from routes.chatbot_routes import router as chatbot_router
from routes.auth_routes import router as auth_router
import logging

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
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables created")
    logging.info("Application started successfully")

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

