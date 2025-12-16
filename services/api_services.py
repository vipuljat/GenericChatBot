# services/api_services.py

from fastapi import HTTPException
from sqlalchemy.orm import Session
from stateful_services.db_schema import Question, Answer
from services.pipeline import classify_intent, Mode, Intent
import uuid

# ======================================================
#               ADMIN QUESTION SERVICES
# ======================================================

def get_all_questions(chatbot_id: str, db: Session):
    row = db.query(Question).filter(Question.chatbot_id == chatbot_id).first()
    if not row:
        raise HTTPException(404, "No questions found")
    return {"questions": row.question_data}


def get_question_by_id(chatbot_id: str, question_id: str, db: Session):
    row = db.query(Question).filter(Question.chatbot_id == chatbot_id).first()
    if not row:
        raise HTTPException(404, "Questions not found")

    for q in row.question_data:
        if q["id"] == question_id:
            return q

    raise HTTPException(404, "Question not found")


def update_question(chatbot_id: str, question_id: str, update: dict, db: Session):
    row = db.query(Question).filter(Question.chatbot_id == chatbot_id).first()
    if not row:
        raise HTTPException(404, "Questions not found")

    updated_list = []
    found = False

    for q in row.question_data:
        if q["id"] == question_id:
            q.update(update)
            found = True
        updated_list.append(q)

    if not found:
        raise HTTPException(404, "Question not found")

    row.question_data = updated_list
    db.commit()
    db.refresh(row)

    return {"message": "Question updated"}


def delete_question(chatbot_id: str, question_id: str, db: Session):
    row = db.query(Question).filter(Question.chatbot_id == chatbot_id).first()
    if not row:
        raise HTTPException(404, "Questions not found")

    new_list = [q for q in row.question_data if q["id"] != question_id]

    if len(new_list) == len(row.question_data):
        raise HTTPException(404, "Question not found")

    row.question_data = new_list
    db.commit()

    return {"message": "Question deleted"}


# ======================================================
#               ADMIN ANSWER SERVICES
# ======================================================

def get_all_answers(chatbot_id: str, db: Session):
    answers = db.query(Answer).filter(Answer.chatbot_id == chatbot_id).all()

    return {
        "attempts": [
            {
                "id": a.id,
                "attempter_id": a.attempter_id,
                "attempter_name": a.attempter_name,
                "answer_data": a.answer_data,
                "created_at": a.created_at
            }
            for a in answers
        ]
    }


# ======================================================
#               USER CHAT SERVICE
# ======================================================

def get_chat_response(chatbot_id: str, question: str, db: Session):
    """
    Handle user chat using RAG service with intent classification.
    Intent: ASKED → ANSWERED
    """
    from services.chatbots_services import rag_query_service
    
    # Classify intent as ASKED in chatbot mode
    intent = classify_intent(question, Mode.CHATBOT)
    
    # Get RAG response
    result = rag_query_service(
        chatbot_name=str(chatbot_id),
        query=question,
        chatbot_mode=Mode.CHATBOT
    )
    
    # Intent transitions to ANSWERED after successful response
    return {
        "chatbot_id": chatbot_id,
        "question": question,
        "answer": result.get("response"),
        "intent": Intent.ANSWERED,
        "sources": result.get("sources", [])
    }


# ======================================================
#               USER QUIZ SERVICES
# ======================================================

def get_quiz_questions(chatbot_id: str, db: Session):
    """
    Get quiz questions for a user.
    Intent: ASKED (when presenting questions)
    """
    row = db.query(Question).filter(Question.chatbot_id == chatbot_id).first()
    if not row:
        return {"questions": [], "intent": Intent.ASKED}
    
    return {
        "questions": row.question_data,
        "intent": Intent.ASKED
    }


def submit_answer(chatbot_id: str, body: dict, db: Session):
    """
    Submit a quiz answer.
    Intent: ANSWERED
    """
    new = Answer(
        id=uuid.uuid4(),
        attempter_id=body.get("attempter_id"),
        attempter_name=body.get("attempter_name"),
        chatbot_id=chatbot_id,
        answer_data=body.get("answer")
    )
    db.add(new)
    db.commit()
    db.refresh(new)

    return {
        "message": "Answer submitted",
        "id": new.id,
        "intent": Intent.ANSWERED
    }


def skip_question(chatbot_id: str, body: dict, db: Session):
    """
    Skip a quiz question.
    Intent: SKIPPED
    """
    new = Answer(
        id=uuid.uuid4(),
        attempter_id=body.get("attempter_id"),
        attempter_name=body.get("attempter_name"),
        chatbot_id=chatbot_id,
        answer_data={"question_id": body["question_id"], "status": "skipped"}
    )
    db.add(new)
    db.commit()

    return {
        "message": "Question skipped",
        "intent": Intent.SKIPPED
    }


def get_user_answers(chatbot_id: str, db: Session):
    """
    Get all user answers (can indicate quiz completion).
    Intent: SUBMITTED (if quiz is complete)
    """
    answers = db.query(Answer).filter(Answer.chatbot_id == chatbot_id).all()
    
    return {
        "answers": [a.answer_data for a in answers],
        "intent": Intent.SUBMITTED if answers else Intent.ASKED
    }
