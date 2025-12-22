"""
People Analyzer Services
"""

# ===================== IMPORTS =====================

from sqlalchemy.orm import Session
from sqlalchemy import or_
from fastapi import HTTPException
from typing import Optional, Dict, Any
import uuid

from stateful_services.db_schema import Employee, PeopleAnalyzer, Chatbot, Question
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


# --------------------------------------------------
# Service: Get Employees (Filtered + Search)
# --------------------------------------------------
def get_employees_service(
    db: Session,
    department: Optional[str] = None,
    role: Optional[str] = None,
    search: Optional[str] = None,
    include_chatbot: bool = False,
):
    """
    Fetch employees with optional filters and chatbot integration.
    
    Args:
        db: Database session
        department: Filter by department (optional)
        role: Filter by employee role (optional)
        search: Search by name or email (optional)
        include_chatbot: Include people analyzer chatbot info (default: False)
    
    Returns:
        Dict with employees list and optional chatbot info
    """
    try:
        query = db.query(Employee)

        if department:
            query = query.filter(Employee.department == department)

        if role:
            query = query.filter(Employee.employee_role == role)

        if search:
            query = query.filter(
                or_(
                    Employee.employee_name.ilike(f"%{search}%"),
                    Employee.employee_email.ilike(f"%{search}%"),
                )
            )

        employees = query.order_by(Employee.employee_name.asc()).all()

        response = {
            "employees": employees
        }
        
        # Include people analyzer chatbot if requested
        if include_chatbot:
            try:
                chatbot_info = get_people_analyzer_chatbot_service(db, department)
                response["people_analyzer_chatbot"] = chatbot_info
            except Exception as e:
                log.error(f"Error fetching people analyzer chatbot: {e}")
                response["people_analyzer_chatbot"] = None
        
        log.info(f"Retrieved {len(response['employees'])} employees")
        return response
    
    except Exception as e:
        log.error(f"Error in get_employees_service: {e}")
        # Return empty result instead of raising exception
        return {
            "employees": []
        }


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



