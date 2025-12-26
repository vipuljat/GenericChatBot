"""
People Analyzer Services
"""

# ===================== IMPORTS =====================

from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import HTTPException
from typing import Optional, Dict, Any
import uuid

from stateful_services.db_schema import Employee, PeopleAnalyzer, Chatbot, Question, ChatbotPermission
from utils.logging import log

# ==================================================
# ===================== SERVICES ===================
# ==================================================

# --------------------------------------------------
# Service: Get People Analyzer Chatbot
# --------------------------------------------------
def get_people_analyzer_chatbot_service(
    db: Session,
    department: Optional[str] = None,
):
    """
    Get active people analyzer chatbot.
    Optionally filter by department from metadata.
    """
    try:
        # Query for active people analyzer chatbots
        query = db.query(Chatbot).filter(
            Chatbot.mode == "people_analyzer",
            Chatbot.status == "active"
        )
        
        # If department is provided, filter by metadata
        if department:
            # Check if metadata contains department filter
            chatbot = query.filter(
                Chatbot.meta_data.contains({"department": department})
            ).first()
            
            # If no department-specific chatbot, get general one
            if not chatbot:
                chatbot = query.filter(
                    or_(
                        Chatbot.meta_data == None,
                        ~Chatbot.meta_data.has_key("department")
                    )
                ).first()
        else:
            chatbot = query.first()
        
        if not chatbot:
            return None
        
        # Get question count
        question_record = db.query(Question).filter(
            Question.chatbot_id == chatbot.chatbot_id,
            Question.status == "active"
        ).first()
        
        question_count = 0
        if question_record and question_record.question_data:
            question_count = len(question_record.question_data)
        
        return {
            "chatbot_id": str(chatbot.chatbot_id),
            "chatbot_name": chatbot.chatbot_name,
            "description": chatbot.description,
            "instruction": chatbot.instruction,
            "question_count": question_count,
            "mode": chatbot.mode,
            "status": chatbot.status,
        }
    except Exception as e:
        log.error(f"Error fetching people analyzer chatbot: {e}")
        return None


def get_employees_service(
    db: Session,
    department: Optional[str] = None,
    role: Optional[str] = None,
    search: Optional[str] = None,
    include_chatbot: bool = False,
    chatbot_id: Optional[str] = None,
    user_id: Optional[str] = None,
):

    log.info("==== get_employees_service START ====")

    response = {
        "employees": [],
        "people_analyzer_chatbot": None,
    }

    # ---------- People Analyzer Mode ----------
    if include_chatbot and chatbot_id:
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == chatbot_id
        ).first()

        if not chatbot:
            log.warning("Chatbot not found")
            return response

        response["people_analyzer_chatbot"] = {
            "chatbot_id": str(chatbot.chatbot_id),
            "name": chatbot.chatbot_name,
            "mode": chatbot.mode,
            "status": chatbot.status,
        }

        if chatbot.mode != "people_analyzer" or not user_id:
            log.info("Not people_analyzer or user_id missing — returning empty")
            return response

        permission = db.query(ChatbotPermission).filter(
            ChatbotPermission.chatbot_id == chatbot_id,
            ChatbotPermission.reviewer_id == user_id,
        ).first()

        if not permission or not permission.can_review_users:
            log.info("No reviewer permission or empty can_review_users")
            return response

        # 🔑 START FROM PERMISSION SCOPE
        query = db.query(Employee).filter(
            Employee.id.in_(permission.can_review_users)
        )

        log.info(
            f"Base employees from can_review_users: {query.count()}"
        )

    else:
        # ---------- Normal Mode ----------
        query = db.query(Employee)

    # ---------- Apply UI Filters ----------
    if department:
        log.info(f"Applying department filter: {department}")
        query = query.filter(Employee.department == department)

    if role:
        log.info(f"Applying role filter: {role}")
        query = query.filter(Employee.employee_role == role)

    if search:
        log.info(f"Applying search filter: {search}")
        query = query.filter(
            or_(
                Employee.employee_name.ilike(f"%{search}%"),
                Employee.employee_email.ilike(f"%{search}%"),
            )
        )

    final_count = query.count()
    log.info(f"FINAL employees count: {final_count}")

    response["employees"] = query.order_by(Employee.created_at.asc()).all()

    log.info("==== get_employees_service END ====")
    return response

# --------------------------------------------------
# Service: Submit Review
# --------------------------------------------------
def submit_review_service(
    db: Session,
    chatbot_id: uuid.UUID,
    reviewer_employee_id: uuid.UUID,
    reviewed_employee_id: uuid.UUID,
    answers: Dict[str, Any],
):
    log.info(
        f"Review submitted by {reviewer_employee_id} for employee {reviewed_employee_id}"
    )

    review = PeopleAnalyzer(
        chatbot_id=chatbot_id,
        employee_id=reviewed_employee_id,
        created_by=reviewer_employee_id,
        answers=answers,
    )

    db.add(review)
    db.commit()
    db.refresh(review)

    return {
        "status": "success",
        "message": "Review submitted successfully",
        "review_id": str(review.id),
        "submitted_by": str(reviewer_employee_id),
        "reviewed_employee_id": str(reviewed_employee_id),
        "submitted_at": review.created_at,
    }


# --------------------------------------------------
# Service: View My Review Summary
# --------------------------------------------------
def get_my_review_summary_service(
    db: Session,
    chatbot_id: uuid.UUID,
    reviewer_employee_id: uuid.UUID,
    reviewed_employee_id: uuid.UUID,
):
    review = (
        db.query(PeopleAnalyzer)
        .filter(
            PeopleAnalyzer.chatbot_id == chatbot_id,
            PeopleAnalyzer.created_by == reviewer_employee_id,
            PeopleAnalyzer.employee_id == reviewed_employee_id,
        )
        .order_by(PeopleAnalyzer.created_at.desc())
        .first()
    )

    if not review:
        raise HTTPException(
            status_code=404,
            detail="No review found for this employee by you",
        )

    return {
        "review_id": str(review.id),
        "chatbot_id": str(review.chatbot_id),
        "reviewed_employee_id": str(review.employee_id),
        "reviewer_employee_id": str(review.created_by),
        "answers": review.answers,
        "submitted_at": review.created_at,
    }



