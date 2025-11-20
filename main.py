from fastapi import FastAPI
from core.database import Base, engine  
from chatbot.controller.chatbot_controller import router as chatbot_router
app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(chatbot_router, prefix="/chatbot")

