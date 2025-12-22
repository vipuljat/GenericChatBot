"""
People Analyzer Routes
Handles employee listing and review submission/viewing
"""

# ===================== IMPORTS =====================

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import uuid

from stateful_services.database import get_db
from routes.employee_routes import get_current_employee_from_token
from utils.logging import log

from services.people_analyzer_service import (
    get_employees_service,
    submit_review_service,
    get_my_review_summary_service,
)

# ===================== ROUTER =====================

router = APIRouter(
    prefix="/api/people-analyzer",
    tags=["People Analyzer"]
)

@router.get("/employees")
def get_employees(
    request: Request,
    department: Optional[str] = None,
    role: Optional[str] = None,
    search: Optional[str] = None,
    include_chatbot: bool = False,
    db: Session = Depends(get_db),
):
    """
    Fetch employees with filters and search.
    
    Parameters:
    - department: Filter by department (optional)
    - role: Filter by role (optional)
    - search: Search by name or email (optional)
    - include_chatbot: If True, includes people analyzer chatbot info for the department (default: False)
    
    Returns:
    - employees: List of employee objects
    - people_analyzer_chatbot: Chatbot info (only if include_chatbot=True)
    """
    try:
        get_current_employee_from_token(request, db)

        result = get_employees_service(
            db=db,
            department=department,
            role=role,
            search=search,
            include_chatbot=include_chatbot,
        )
        
        log.info(f"Returning {len(result.get('employees', []))} employees")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in get_employees endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch employees: {str(e)}"
        )


# --------------------------------------------------
# 2️⃣ Submit Review Answers
# --------------------------------------------------
@router.post("/review/submit", summary="Submit people analyzer review")
def submit_review(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
):
    reviewer = get_current_employee_from_token(request, db)

    try:
        chatbot_id = uuid.UUID(payload["chatbot_id"])
        reviewed_employee_id = uuid.UUID(payload["employee_id"])
        answers = payload["answers"]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid request payload")

    return submit_review_service(
        db=db,
        chatbot_id=chatbot_id,
        reviewer_employee_id=reviewer.id,
        reviewed_employee_id=reviewed_employee_id,
        answers=answers,
    )


# --------------------------------------------------
# 3️⃣ View My Review Summary (Reviewer Only)
# --------------------------------------------------
@router.get("/review/summary", summary="View my review summary")
def view_my_review_summary(
    request: Request,
    chatbot_id: str,
    reviewed_employee_id: str,
    db: Session = Depends(get_db),
):
    reviewer = get_current_employee_from_token(request, db)

    try:
        chatbot_uuid = uuid.UUID(chatbot_id)
        reviewed_uuid = uuid.UUID(reviewed_employee_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")

    return get_my_review_summary_service(
        db=db,
        chatbot_id=chatbot_uuid,
        reviewer_employee_id=reviewer.id,
        reviewed_employee_id=reviewed_uuid,
    )
