# routes/role.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from stateful_services.database import get_db

from services.role import (
    # Admin Question Services
    get_all_questions,
    get_question_by_id,
    update_question,
    delete_question,

    # Admin Answer View Service
    get_all_answers,

    # User Chatbot Service
    get_chat_response,

    # User Quiz Services
    get_quiz_questions,
    submit_answer,
    skip_question,
    get_user_answers
)

router = APIRouter()

# ======================================================
#               ADMIN QUESTION ROUTES
# ======================================================

@router.get("/admin/questions/{chatbot_id}")
def route_get_all_questions(chatbot_id: str, db: Session = Depends(get_db)):
    return get_all_questions(chatbot_id, db)


@router.get("/admin/question/{chatbot_id}/{question_id}")
def route_get_question(chatbot_id: str, question_id: str, db: Session = Depends(get_db)):
    return get_question_by_id(chatbot_id, question_id, db)


@router.put("/admin/question/{chatbot_id}/{question_id}")
def route_update_question(chatbot_id: str, question_id: str, body: dict, db: Session = Depends(get_db)):
    return update_question(chatbot_id, question_id, body, db)


@router.delete("/admin/question/{chatbot_id}/{question_id}")
def route_delete_question(chatbot_id: str, question_id: str, db: Session = Depends(get_db)):
    return delete_question(chatbot_id, question_id, db)


# ======================================================
#               ADMIN ANSWER ROUTES
# ======================================================

@router.get("/admin/chatbot/{chatbot_id}/attempts")
def route_get_all_attempts(chatbot_id: str, db: Session = Depends(get_db)):
    return get_all_answers(chatbot_id, db)


# ======================================================
#               USER CHAT ROUTES
# ======================================================

@router.post("/user/chat/{chatbot_id}")
def route_user_chat(chatbot_id: str, body: dict, db: Session = Depends(get_db)):
    return get_chat_response(chatbot_id, body["question"], db)


# ======================================================
#               USER QUIZ ROUTES
# ======================================================

@router.get("/user/quiz/{chatbot_id}/questions")
def route_quiz_questions(chatbot_id: str, db: Session = Depends(get_db)):
    return get_quiz_questions(chatbot_id, db)


@router.post("/user/quiz/{chatbot_id}/answer")
def route_submit_answer(chatbot_id: str, body: dict, db: Session = Depends(get_db)):
    return submit_answer(chatbot_id, body, db)


@router.post("/user/quiz/{chatbot_id}/skip")
def route_skip(chatbot_id: str, body: dict, db: Session = Depends(get_db)):
    return skip_question(chatbot_id, body, db)


@router.get("/user/quiz/{chatbot_id}/answers")
def route_get_user_answers(chatbot_id: str, db: Session = Depends(get_db)):
    return get_user_answers(chatbot_id, db)
