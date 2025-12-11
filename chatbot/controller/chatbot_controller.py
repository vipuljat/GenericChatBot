from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from chatbot.schema.request import ChatbotCreate
from chatbot.services.chabot_services import create_chatbot_with_documents
from fastapi import UploadFile, File, Form
from agent.agent import render_questions_from_file
from chatbot.services.chabot_services import UPLOAD_DIR
import shutil

from typing import List,Optional
import os
router = APIRouter()

# @router.post("/chatbots")
# def create_chatbot_route(data: ChatbotCreate, db: Session = Depends(get_db)):
#     create_chatbot(db, data)
#     return {"message": "Chatbot created successfully"}

@router.post("/chatbots")
def create_chatbot_route(
    chatbot_name: str = Form(...),
    description: str = Form(...),
    instructions: str = Form(...),
    conversation_starters: Optional[List[str]] = Form(None),
    is_quiz_mode: bool = Form(False),
    is_active: bool = Form(True),
    recommended_model: Optional[str] = Form(None),
    file: UploadFile = File(...),
    quiz_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    chatbot_id = create_chatbot_with_documents(
        db=db,
        chatbot_name=chatbot_name,
        description=description,
        instructions=instructions,
        conversation_starters=conversation_starters,
        is_quiz_mode=is_quiz_mode,
        is_active=is_active,
        recommended_model=recommended_model,
        file=file,
        quiz_file=quiz_file,
    )

    return {
        "message": "Chatbot created successfully",
        "chatbot_id": chatbot_id
    }

@router.post("/quiz/render")
async def upload_and_render_quiz_document(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call the agent
    try:
        rendered_questions = render_questions_from_file(file_path)
    except Exception as e:
        raise Exception(status_code=500, detail=f"Error processing document: {str(e)}")

    return {
        "file_name": file.filename,
        "total_questions": len(rendered_questions),
        "questions": rendered_questions
    }