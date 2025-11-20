from sqlalchemy.orm import Session
from models.chatbot import Chatbot
from models.documents import ChatbotDocument
from chatbot.schema.request import ChatbotCreate
import os
from fastapi import UploadFile, File, Form
from typing import Optional
# def create_chatbot(db: Session, data: ChatbotCreate):
#     chatbot = Chatbot(
#         chatbot_name=data.chatbot_name,
#         description=data.description,
#         instructions=data.instructions,
#         conversation_starters=data.conversation_starters,
#         is_quiz_mode=data.is_quiz_mode,
#         is_active=data.is_active,
#         recommended_model=data.recommended_model
#     )
#     db.add(chatbot)
#     db.commit()
#     db.refresh(chatbot)
#     return {"data":"Chatbot created"}

UPLOAD_DIR = "uploaded_chatbots"

def save_file(upload_file):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    filename = f"{upload_file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())

    return file_path

def create_chatbot_with_documents(
    db: Session,
    chatbot_name: str,
    description: str,
    instructions: str,
    conversation_starters: list,
    is_quiz_mode: bool,
    is_active: bool,
    recommended_model: str,
    file: UploadFile,
    quiz_file: Optional[UploadFile]
):
    # 1️⃣ Create chatbot entry
    chatbot = Chatbot(
        chatbot_name=chatbot_name,
        description=description,
        instructions=instructions,
        conversation_starters=conversation_starters,
        is_quiz_mode=is_quiz_mode,
        is_active=is_active,
        recommended_model=recommended_model
    )
    db.add(chatbot)
    db.commit()
    db.refresh(chatbot)

    chatbot_id = chatbot.id

    # 2️⃣ Save main document
    if file:
        normal_path = save_file(file)
        normal_doc = ChatbotDocument(
            chatbot_id=chatbot_id,
            doc_name=file.filename,
            file_path=normal_path,
            is_quiz_document=False
        )
        db.add(normal_doc)

    # 3️⃣ Save quiz document (only if enabled)
    if is_quiz_mode and quiz_file:
        quiz_path = save_file(quiz_file)
        quiz_doc = ChatbotDocument(
            chatbot_id=chatbot_id,
            doc_name=quiz_file.filename,
            file_path=quiz_path,
            is_quiz_document=True
        )
        db.add(quiz_doc)

    db.commit()
    return chatbot_id
