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
) -> Dict[str, Any]:
    """
    Fetch employees with optional filters.
    
    When include_chatbot=True and chatbot_id is provided:
    - If mode == "people_analyzer" and user is a reviewer:
      → First apply department/role/search filters
      → Then intersect with employees the current user is allowed to review
    - Otherwise → return empty list for safety/privacy in people analyzer context
    
    This endpoint is specifically useful for People Analyzer flow.
    """
    try:
        # Base query with standard filters
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

        # Apply base filters first (department, role, search)
        base_employees = query.order_by(Employee.employee_name.asc()).all()
        log.debug(f"Base filtered employees count: {len(base_employees)}")

        response: Dict[str, Any] = {
            "employees": base_employees,  # default
            "people_analyzer_chatbot": None
        }

        # People analyzer specific logic - applied on top
        if include_chatbot and chatbot_id:
            try:
                # Parse chatbot_id
                try:
                    parsed_chatbot_id = uuid.UUID(chatbot_id)
                except ValueError:
                    log.warning(f"Invalid chatbot_id format: {chatbot_id}")
                    response["people_analyzer_chatbot"] = {"error": "Invalid chatbot ID"}
                    response["employees"] = []  # strict
                    return response

                # Get chatbot
                chatbot = db.query(Chatbot).filter(
                    Chatbot.chatbot_id == parsed_chatbot_id
                ).first()

                if not chatbot:
                    response["people_analyzer_chatbot"] = {"error": "Chatbot not found"}
                    response["employees"] = []
                    return response

                chatbot_info = {
                    "chatbot_id": str(chatbot.chatbot_id),
                    "name": chatbot.chatbot_name,
                    "mode": chatbot.mode,
                    "status": chatbot.status,
                }
                response["people_analyzer_chatbot"] = chatbot_info

                # Only apply reviewer filtering in people_analyzer mode
                if chatbot.mode != "people_analyzer" or not user_id:
                    log.info(f"People analyzer filtering skipped: mode={chatbot.mode}, user_id={user_id}")
                    return response

                # Parse reviewer (user_id)
                try:
                    reviewer_uuid = uuid.UUID(user_id)
                except ValueError:
                    log.warning(f"Invalid user_id format: {user_id}")
                    response["employees"] = []
                    return response

                # Find permission for this reviewer
                permission = db.query(ChatbotPermission).filter(
                    ChatbotPermission.chatbot_id == parsed_chatbot_id,
                    ChatbotPermission.reviewer_id == reviewer_uuid
                ).first()

                if not permission or not permission.can_review_users:
                    log.info(f"No review permission for user {user_id} in chatbot {chatbot_id}")
                    response["employees"] = []
                    return response

                # Convert allowed subjects (str IDs → UUID)
                try:
                    allowed_subject_uuids = [
                        uuid.UUID(str_sid.strip())
                        for str_sid in permission.can_review_users
                        if str_sid and str_sid.strip()
                    ]
                except ValueError as ve:
                    log.error(f"Invalid UUID in can_review_users: {ve}")
                    response["employees"] = []
                    return response

                if not allowed_subject_uuids:
                    response["employees"] = []
                    log.info(f"No subjects defined for reviewer {user_id}")
                    return response

                # Now filter the already filtered base_employees with allowed subjects
                allowed_employee_ids = {emp.id for emp in base_employees}  # set of UUIDs
                final_ids = allowed_employee_ids.intersection(set(allowed_subject_uuids))

                if not final_ids:
                    response["employees"] = []
                else:
                    response["employees"] = [
                        emp for emp in base_employees
                        if emp.id in final_ids
                    ]
                    # or alternatively re-query for consistency:
                    # response["employees"] = db.query(Employee).filter(
                    #     Employee.id.in_(final_ids)
                    # ).order_by(Employee.employee_name.asc()).all()

                log.info(
                    f"People analyzer filter applied: {len(response['employees'])} "
                    f"employees remain after reviewer permission"
                )

            except Exception as e:
                log.error(f"People analyzer logic error: {e}", exc_info=True)
                response["people_analyzer_chatbot"] = {"error": str(e)}
                response["employees"] = []  # fail-safe

        log.info(f"Final response contains {len(response['employees'])} employees")
        return response

    except Exception as e:
        log.error(f"Critical error in get_employees_service: {e}", exc_info=True)
        return {
            "employees": [],
            "people_analyzer_chatbot": None
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



