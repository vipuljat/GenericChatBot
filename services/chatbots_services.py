"""
Refactored Chatbot Service - FUNCTIONAL APPROACH
"""

# ============================================================================
# services/chatbot_service.py - FUNCTIONAL CHATBOT CRUD
# ============================================================================

"""Functional chatbot CRUD service."""

import uuid
from typing import List, Optional, Dict, Any
from fastapi.encoders import jsonable_encoder
from fastapi.params import Depends
from sqlalchemy.orm import Session
from fastapi import HTTPException, BackgroundTasks, Request
from qdrant_client.models import PointStruct
from sqlalchemy.orm import aliased

from stateful_services.database import get_db
from stateful_services.db_schema import Answer, Chatbot, ChatbotAccess, ChatbotPermission, Employee, PeopleAnalyzer, Question
from utils.document_service import extract_text_from_document, get_file_extension
from utils.logging import log

# Import functional services
from utils.embedding import process_document_for_embedding
from services.vectore_store_service import (
    create_collection,
    upsert_points,
    delete_collection
)
from utils.token_decode import get_current_employee_from_token


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_questions(db: Session, chatbot_id: uuid.UUID, questions: dict):
    """Save or update questions for quiz mode."""
    existing = db.query(Question).filter(
        Question.chatbot_id == chatbot_id
    ).first()
    
    if existing:
        existing.question_data = questions
    else:
        new_questions = Question(
            chatbot_id=chatbot_id,
            question_data=questions
        )
        db.add(new_questions)
    
    db.commit()
    log.info(f"✓ Questions saved for chatbot {chatbot_id}")


def process_documents_background(
    chatbot_name: str,
    doc_contents: List[bytes],
    doc_names: List[str]
):
    """
    Background task: Process documents and store embeddings.
    
    KEY: Embeddings generated ONCE per chunk with cost logging.
    """
    try:
        log.info(f"📄 Processing {len(doc_contents)} documents for '{chatbot_name}'")
        
        all_points = []
        point_id = 0
        
        # Process each document
        for doc_content, doc_name in zip(doc_contents, doc_names):
            try:
                log.info(f"Processing: {doc_name}")
                
                # Extract text
                text = extract_text_from_document(doc_content, doc_name)
                
                if not text or not text.strip():
                    log.warning(f"⚠️  No text from {doc_name}")
                    continue
                
                log.info(f"✓ Extracted {len(text)} chars from {doc_name}")
                
                doc_type = get_file_extension(doc_name) or 'unknown'
                
                # Chunk and generate embeddings (costs logged inside)
                chunks_with_embeddings = process_document_for_embedding(
                    text=text,
                    metadata={
                        'chatbot_name': chatbot_name,
                        'source_file': doc_name,
                        'document_type': doc_type
                    }
                )
                
                if not chunks_with_embeddings:
                    log.warning(f"⚠️  No chunks from {doc_name}")
                    continue
                
                log.info(f"✓ {len(chunks_with_embeddings)} chunks with embeddings")
                
                # Prepare Qdrant points
                for chunk in chunks_with_embeddings:
                    point = PointStruct(
                        id=point_id,
                        vector=chunk['embedding'],
                        payload={
                            'text': chunk['text'],
                            'chatbot_name': chatbot_name,
                            'source_file': doc_name,
                            'document_type': doc_type,
                            'chunk_index': chunk['chunk_index'],
                            'total_chunks': chunk['total_chunks'],
                            'char_count': chunk['char_count']
                        }
                    )
                    all_points.append(point)
                    point_id += 1
            
            except Exception as e:
                log.error(f"❌ Error processing {doc_name}: {e}", exc_info=True)
                continue
        
        if not all_points:
            log.error(f"❌ No embeddings generated for '{chatbot_name}'")
            return
        
        # Create collection and upload
        vector_size = len(all_points[0].vector)
        create_collection(
            collection_name=chatbot_name,
            vector_size=vector_size,
            force_recreate=True
        )
        
        upsert_points(
            collection_name=chatbot_name,
            points=all_points
        )
        
        log.info(f"✅ Stored {len(all_points)} embeddings for '{chatbot_name}'")
    
    except Exception as e:
        log.error(f"❌ Background processing failed: {e}", exc_info=True)


# ============================================================================
# MAIN CRUD FUNCTIONS
# ============================================================================

def create_chatbot(
    request: Request,
    chatbot_name: str,
    status: str,
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    doc_contents: Optional[List[bytes]] = None,
    doc_names: Optional[List[str]] = None,
    generated_by: Optional[uuid.UUID] = None,
    meta_data: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = "general",
    questions: Optional[dict] = None,
    employee_ids: Optional[List[str]] = None,
    access_list: Optional[List[str]] = None,
    background_tasks: Optional[BackgroundTasks] = None,
    db: Session=Depends(get_db)
) -> Chatbot:
    """Create chatbot and schedule document processing."""
    try:
        user_info = get_current_employee_from_token(request, db)
        
        user_id = user_info.id
        # Check uniqueness
        existing = db.query(Chatbot).filter(
            Chatbot.chatbot_name == chatbot_name
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Chatbot '{chatbot_name}' already exists"
            )
        
        # Create chatbot record
        chatbot = Chatbot(
            chatbot_name=chatbot_name,
            description=description,
            instruction=instruction,
            pdf_names=doc_names or [],
            status=status,
            mode=mode,
            generated_by=user_id,
            meta_data=meta_data or {}
        )
        
        db.add(chatbot)
        db.commit()
        db.refresh(chatbot)
        
        log.info(f" Chatbot created: {chatbot_name} (ID: {chatbot.chatbot_id})")
        
        if questions and mode in ("quiz", "people_analyzer"):
            save_questions(db, chatbot.chatbot_id, questions)

        if employee_ids:
            # Create access record
            access_record = ChatbotAccess(
                chatbot_id=chatbot.chatbot_id,
                created_by=user_id,
                allowed_users=employee_ids
            )
            db.add(access_record)
            db.commit()
            log.info(f"Allowed users set for chatbot {chatbot_name}")
        
        if access_list and mode == "people_analyzer":
            for entry in access_list:
                employee_id = entry.get("employee_id")
                allowed_to_review = entry.get("allowed_users", [])

                if not employee_id:
                    continue

                permission = ChatbotPermission(
                    chatbot_id=chatbot.chatbot_id,
                    reviewer_id=employee_id,               # reviwer
                    can_review_users=allowed_to_review,    # people who reviewer can give feedback
                )
                db.add(permission)
                db.commit()
                log.info(f"Allowed feedback givers set for chatbot {chatbot_name}")
            
        # Schedule document processing
        if doc_contents and doc_names and background_tasks:
            background_tasks.add_task(
                process_documents_background,
                chatbot_name=chatbot_name,
                doc_contents=doc_contents,
                doc_names=doc_names
            )
            log.info(f"Scheduled: {len(doc_names)} documents")
        
        return chatbot
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f" Create failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def update_chatbot(
    db: Session,
    chatbot_id: uuid.UUID,
    chatbot_name: Optional[str] = None,
    status: Optional[str] = None,
    description: Optional[str] = None,
    instruction: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    doc_contents: Optional[List[bytes]] = None,
    doc_names: Optional[List[str]] = None,
    questions: Optional[dict] = None,
    employee_ids: Optional[List[str]] = None,
    access_list: Optional[List[Dict[str, Any]]] = None,  # [{"reviewer_id": "...", "can_review_users": [...]}]
    background_tasks: Optional[BackgroundTasks] = None
) -> Chatbot:

    try:
        chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Name uniqueness check
        if chatbot_name and chatbot_name != chatbot.chatbot_name:
            if db.query(Chatbot).filter(
                Chatbot.chatbot_name == chatbot_name,
                Chatbot.chatbot_id != chatbot_id
            ).first():
                raise HTTPException(status_code=400, detail="Chatbot name already exists")
            chatbot.chatbot_name = chatbot_name

        # Basic scalar field updates
        if status is not None:
            chatbot.status = status
        if description is not None:
            chatbot.description = description
        if instruction is not None:
            chatbot.instruction = instruction
        if meta_data is not None:
            chatbot.meta_data = meta_data
        if mode is not None:
            chatbot.mode = mode
        if doc_names is not None:
            chatbot.pdf_names = doc_names

        # ── Update ChatbotAccess (who can access the chatbot) ──────────────────
        if employee_ids is not None:
            # We usually expect one record per chatbot from creator
            access = db.query(ChatbotAccess).filter(
                ChatbotAccess.chatbot_id == chatbot_id,
                ChatbotAccess.created_by == chatbot.generated_by
            ).first()

            if access:
                access.allowed_users = employee_ids
            else:
                access = ChatbotAccess(
                    chatbot_id=chatbot_id,
                    created_by=chatbot.generated_by,
                    allowed_users=employee_ids
                )
                db.add(access)
            log.info(f"Updated chatbot access (visibility) for {chatbot.chatbot_name}")

        # ── Update ChatbotPermission (review permissions) ──────────────────────
        if access_list is not None:
            # Full replace strategy (common & simple)
            db.query(ChatbotPermission).filter(
                ChatbotPermission.chatbot_id == chatbot_id
            ).delete()

            for entry in access_list:
                reviewer_id = entry.get("reviewer_id") or entry.get("employee_id")
                can_review_users = entry.get("allowed_users", [])

                if not reviewer_id:
                    continue

                permission = ChatbotPermission(
                    id=uuid.uuid4(),  # explicit if you want
                    chatbot_id=chatbot.chatbot_id,
                    reviewer_id=reviewer_id,
                    can_review_users=can_review_users
                )
                db.add(permission)

            log.info(f"Replaced review permissions (who can review whom) for {chatbot.chatbot_name}")

        # Questions update - support both modes
        if questions is not None and chatbot.mode in ("quiz", "people_analyzer"):
            save_questions(db, chatbot.chatbot_id, questions)

        db.commit()
        db.refresh(chatbot)

        # New documents in background
        if doc_contents and doc_names and background_tasks:
            background_tasks.add_task(
                process_documents_background,
                chatbot_name=chatbot.chatbot_name,
                doc_contents=doc_contents,
                doc_names=doc_names
            )

        log.info(f"Chatbot updated successfully: {chatbot.chatbot_id}")
        return chatbot

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def delete_chatbot_by_id(db: Session, chatbot_id: uuid.UUID) -> bool:
    """Delete chatbot and its vector collection."""
    try:
        chatbot = db.query(Chatbot).filter(
            Chatbot.chatbot_id == chatbot_id
        ).first()
        
        if not chatbot:
            raise HTTPException(status_code=404, detail="Chatbot not found")
        
        chatbot_name = chatbot.chatbot_name
        
        # Delete vector collection
        delete_collection(chatbot_name)
        
        # Delete questions
        db.query(Question).filter(Question.chatbot_id == chatbot_id).delete()
        
        # Delete chatbot
        db.delete(chatbot)
        db.commit()
        
        log.info(f"🗑️  Chatbot deleted: {chatbot_name}")
        return True
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"❌ Delete failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def get_chatbot_responses_service(
    chatbot_id: uuid.UUID,
    department: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Main service to route based on mode."""
    chatbot = db.query(Chatbot).filter(Chatbot.chatbot_id == chatbot_id).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    if chatbot.mode == "quiz":
        return get_quiz_responses(chatbot_id, db)
    elif chatbot.mode == "people_analyzer":
        return get_people_analyzer_responses(chatbot_id, department, db)
    else:
        raise HTTPException(status_code=400, detail="Unsupported chatbot mode")


def get_quiz_responses(chatbot_id: uuid.UUID, db: Session) -> Dict[str, Any]:
    """Get all quiz attempts with question text and options in answers."""
    
    # Fetch active questions to get text and options
    questions_raw = db.query(Question)\
        .filter(Question.chatbot_id == chatbot_id, Question.status == "active")\
        .all()

    # Build question map: id -> {text, options}
    question_map = {}
    for q in questions_raw:
        data_list = q.question_data
        if isinstance(data_list, list):
            for item in data_list:
                qid = item.get("id")
                if qid is not None:
                    question_map[qid] = {
                        "text": item.get("text"),
                        "options": item.get("options", []),
                        "type": item.get("type", "mcq")
                    }

    # Fetch attempts
    attempts = db.query(Answer)\
        .filter(Answer.chatbot_id == chatbot_id)\
        .order_by(Answer.created_at.desc())\
        .all()

    if not attempts:
        return {
            "mode": "quiz",
            "chatbot_id": str(chatbot_id),
            "total_attempts": 0,
            "attempts": []
        }

    formatted_attempts = []

    for attempt in attempts:
        answers_data = attempt.answer_data or {}
        employee = db.query(Employee)\
            .filter(Employee.id == attempt.attempter_by_id)\
            .first()

        detailed_answers = []
        attempted_count = 0

        for q_id_str, ans_obj in answers_data.items():
            try:
                q_id = int(q_id_str)
            except:
                continue

            answer_value = ans_obj.get("answer") if isinstance(ans_obj, dict) else ans_obj
            is_skipped = answer_value == "skipped" or (isinstance(ans_obj, dict) and ans_obj.get("answer") == "skipped")

            question_info = question_map.get(q_id, {"text": f"Question {q_id}", "options": [], "type": "mcq"})

            detailed_answers.append({
                "question_id": q_id,
                "question_text": question_info["text"],
                "type": question_info["type"],
                "options": question_info["options"],
                "selected_answer": "Skipped" if is_skipped else answer_value
            })

            if not is_skipped:
                attempted_count += 1

        # Sort by question_id
        detailed_answers.sort(key=lambda x: x["question_id"])

        formatted_attempts.append({
            "attempt_id": str(attempt.id),
            "employee_id": attempt.attempter_by_id  ,
            "employee_name": employee.employee_name if employee else "Unknown",
            "department": employee.department if employee else None,
            "attempted_count": attempted_count,
            "total_questions": len(answers_data),
            "answers": detailed_answers,
            "created_at": attempt.created_at.isoformat()
        })

    return {
        "mode": "quiz",
        "chatbot_id": str(chatbot_id),
        "total_attempts": len(formatted_attempts),
        "attempts": formatted_attempts
    }
def get_people_analyzer_responses(
    chatbot_id: uuid.UUID,
    department: Optional[str],
    db: Session
) -> Dict[str, Any]:
    """Aggregated responses for people_analyzer with overall average rating."""
    
    # Fetch questions
    questions_raw = db.query(Question)\
        .filter(Question.chatbot_id == chatbot_id, Question.status == "active")\
        .all()

    questions = []
    for q in questions_raw:
        data_list = q.question_data
        if isinstance(data_list, list):
            for item in data_list:
                questions.append({
                    "id": item.get("id"),
                    "text": item.get("text"),
                    "category": item.get("category"),
                    "order": item.get("order", 999)
                })
        else:
            questions.append({
                "id": data_list.get("id"),
                "text": data_list.get("text"),
                "category": data_list.get("category"),
                "order": data_list.get("order", 999)
            })

    questions.sort(key=lambda x: x.get("order", 999))

    # Fetch ratings with both rated employee and rater information
    RatedEmployee = aliased(Employee)
    RaterEmployee = aliased(Employee)
    
    query = db.query(
        PeopleAnalyzer,
        RatedEmployee.employee_name.label('rated_employee_name'),
        RatedEmployee.department.label('rated_employee_department'),
        RaterEmployee.employee_name.label('rater_name')
    )\
        .join(RatedEmployee, PeopleAnalyzer.employee_id == RatedEmployee.id)\
        .outerjoin(RaterEmployee, PeopleAnalyzer.created_by == RaterEmployee.id)\
        .filter(PeopleAnalyzer.chatbot_id == chatbot_id)

    if department:
        query = query.filter(RatedEmployee.department.ilike(f"%{department}%"))

    entries = query.all()

    if not entries:
        return {
            "mode": "people_analyzer",
            "chatbot_id": str(chatbot_id),
            "questions": questions,
            "filters_applied": {"department": department},
            "aggregated_data": []
        }

    # Group by rated employee
    grouped: Dict[str, Dict] = {}

    for entry, rated_employee_name, rated_employee_department, rater_name in entries:
        emp_id = entry.employee_id
        if emp_id not in grouped:
            grouped[emp_id] = {
                "employee_id": emp_id,
                "employee_name": rated_employee_name or "Unknown",
                "department": rated_employee_department or "Unknown",
                "total_ratings": 0,
                "question_stats": {},
                "total_sum": 0,
                "total_count": 0,
                "responses": []
            }

        # Store individual response details with rater name
        response_detail = {
            "rater_id": entry.created_by,
            "rater_name": rater_name or "Anonymous",
            "answers": entry.answers or {},
            "created_at": entry.created_at.isoformat() if entry.created_at else None
        }
        grouped[emp_id]["responses"].append(response_detail)
        
        grouped[emp_id]["total_ratings"] += 1

        answers = entry.answers or {}
        for q_id_str, ans_obj in answers.items():
            answer = ans_obj.get("answer") if isinstance(ans_obj, dict) else ans_obj
            if answer == "skipped" or answer is None:
                continue

            if q_id_str not in grouped[emp_id]["question_stats"]:
                grouped[emp_id]["question_stats"][q_id_str] = {"sum": 0, "count": 0}

            score = {"+": 4, "-": 0, "+-": 2}.get(str(answer).strip(), 0)
            grouped[emp_id]["question_stats"][q_id_str]["sum"] += score
            grouped[emp_id]["question_stats"][q_id_str]["count"] += 1

            # Add to overall total
            grouped[emp_id]["total_sum"] += score
            grouped[emp_id]["total_count"] += 1

    # Final aggregation
    aggregated_data = []
    for emp_id, data in grouped.items():
        question_averages = {}
        for q_id, stats in data["question_stats"].items():
            avg = stats["sum"] / stats["count"] if stats["count"] > 0 else 0
            question_averages[q_id] = {
                "average": round(avg, 2),
                "count": stats["count"]
            }

        overall_average = (
            round(data["total_sum"] / data["total_count"], 2)
            if data["total_count"] > 0
            else 0
        )

        aggregated_data.append({
            "employee_id": data["employee_id"],
            "employee_name": data["employee_name"],
            "department": data["department"],
            "total_ratings": data["total_ratings"],
            "overall_average": overall_average,
            "question_averages": question_averages,
            "responses": data["responses"]
        })

    # Sort by overall_average descending (highest rated first)
    aggregated_data.sort(key=lambda x: x["overall_average"], reverse=True)

    return {
        "mode": "people_analyzer",
        "chatbot_id": str(chatbot_id),
        "questions": questions,
        "filters_applied": {"department": department},
        "aggregated_data": aggregated_data
    }