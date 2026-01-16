"""
Quiz Mode Service - Functional approach with navigation support.
Handles interactive quiz assessments with intelligent conversation flow.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import google.generativeai as genai
import config
from sqlalchemy.orm import Session
from fastapi import HTTPException
from stateful_services.db_schema import Chatbot, PeopleAnalyzer, Question, Answer
from utils.logging import log
import re

# UPDATED IMPORTS - Using refactored functional services
from services.rag_service import retrieve_context, generate_with_retry

genai.configure(api_key=config.GEMINI_API_KEY)


# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_user_intent(user_message: str, chatbot_mode: str = "quiz") -> str:
    """
    Detect user intent using Gemini for accurate classification.
    
    Args:
        user_message: The user's message
        chatbot_mode: "quiz" or "people_analyzer"
    
    Returns:
        "greeting" | "skip" | "end_quiz" | "clarification_question" | "answer" | "navigate_previous" | "navigate_next"
    """
    try:
        msg_lower = user_message.lower().strip()
        
        # Special handling for people_analyzer mode - accept both symbols and typed words
        if chatbot_mode == "people_analyzer":
            # Accept symbols: +, -, ±, +-
            if user_message.strip() in ["+", "-", "±", "+-"]:
                return "answer"
            # Accept typed words: positive, negative, average
            if msg_lower in ["positive", "negative", "average"]:
                return "answer"
        
        # Fast rule-based detection for obvious cases
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'start']
        if any(msg_lower == greet or msg_lower.startswith(greet + ' ') for greet in greetings) and len(msg_lower) < 30:
            return "greeting"
        
        # Navigation detection - MUST come before other checks
        previous_phrases = ['previous question', 'prev question', 'go back', 'last question', 'back']
        if any(phrase in msg_lower for phrase in previous_phrases):
            return "navigate_previous"
        
        next_phrases = ['next question', 'move to next']
        if any(phrase in msg_lower for phrase in next_phrases):
            return "navigate_next"
        
        # Direct question number navigation (e.g., "Question 1", "question 5")
        if re.match(r'^question\s+\d+$', msg_lower):
            return "navigate_direct"
        
        # End quiz detection
        end_phrases = ['end quiz', 'finish quiz', 'stop quiz', 'submit quiz', 'i am done', "i'm done", 'submit']
        if any(phrase in msg_lower for phrase in end_phrases):
            return "end_quiz"
        
        # Skip detection
        skip_phrases = ['skip', 'skip this', 'skip it', 'pass', 'next']
        if any(msg_lower == phrase or msg_lower.startswith(phrase + ' ') for phrase in skip_phrases):
            return "skip"
        
        # For ambiguous cases, use Gemini
        prompt = f"""Classify the user's intent in a quiz context.

User message: "{user_message}"

Classify as ONE of these intents:
1. "clarification_question" - User asking for help/explanation (e.g., "explain this", "what does this mean?", "I don't understand")
2. "skip" - User wants to skip (e.g., "skip", "don't know", "I don't know")
3. "answer" - User providing an answer (e.g., "A", "option B", "I think it's C", any actual answer)
4. "end_quiz" - User wants to end the quiz (e.g., "I'm done", "submit")
5. "navigate_previous" - User wants to go to previous question (e.g., "previous question", "go back")
6. "navigate_next" - User wants to move to next question (e.g., "next question")

Respond with ONLY ONE WORD: clarification_question, skip, answer, end_quiz, navigate_previous, or navigate_next.

NOTE: if user use +, _ or +-, it will be considered as answer.

Examples:
"explain" -> clarification_question
"i don't get the question" -> clarification_question
"can you clarify?" -> clarification_question
"previous question" -> navigate_previous
"go back" -> navigate_previous
"next question" -> navigate_next
"don't know" -> skip
"I don't know" -> skip
"A" -> answer
"option B" -> answer
"The answer is Paris" -> answer
"I'm done" -> end_quiz
"""
        
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        detected_intent = response.text.strip().lower()
        
        # Validate response
        valid_intents = ['clarification_question', 'skip', 'answer', 'end_quiz', 'navigate_previous', 'navigate_next']
        if detected_intent in valid_intents:
            log.info(f"Gemini detected intent: {detected_intent} for '{user_message[:50]}'")
            return detected_intent
        
        log.warning(f"Invalid intent from Gemini: {detected_intent}, defaulting to 'answer'")
        return "answer"
        
    except Exception as e:
        log.error(f"Intent detection failed: {e}, defaulting to 'answer'")
        return "answer"


def get_quiz_session_from_history(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Extract quiz state from conversation history.
    
    Returns:
        {
            "current_index": int,
            "attempted": [int],
            "skipped": [int],
            "answers": {str(index): answer},
            "completed": bool
        }
    """
    state = {
        "current_index": 0,
        "attempted": [],
        "skipped": [],
        "answers": {},
        "completed": False
    }
    
    current_index = 0
    attempted = []
    skipped = []
    answers = {}
    
    for i, msg in enumerate(conversation_history):
        if msg.get("role") in ["model", "assistant", "bot"]:
            content = msg.get("content", "")
            
            # Check for question markers to track current position
            match = re.search(r'Question (\d+) of (\d+)', content)
            if match:
                question_num = int(match.group(1))
                current_index = question_num - 1
            
            # Check for skip confirmation
            if "skipping that question" in content.lower() or "okay, skipping" in content.lower():
                if current_index > 0:
                    skipped_idx = current_index - 1
                    if skipped_idx not in skipped:
                        skipped.append(skipped_idx)
                        answers[str(skipped_idx)] = "skipped"
            
            # Check for answer recorded - MULTIPLE PATTERNS
            answer_recorded = (
                "answer recorded" in content.lower() or
                "got it!" in content.lower() or
                "thank you for your feedback" in content.lower()
            )
            
            if answer_recorded:
                if current_index > 0:
                    answered_idx = current_index - 1
                    if answered_idx not in attempted:
                        attempted.append(answered_idx)
                        # Get the actual answer from previous user message
                        if i > 0:
                            prev_msg = conversation_history[i-1]
                            if prev_msg.get("role") == "user":
                                answers[str(answered_idx)] = {
                                    "answer": prev_msg.get("content", "")
                                }
    
    state["current_index"] = current_index
    state["attempted"] = attempted
    state["skipped"] = skipped
    state["answers"] = answers
    
    log.info(f"Quiz state: index={current_index}, attempted={len(attempted)}, skipped={len(skipped)}, answers={list(answers.keys())}")
    return state


def build_quiz_progress_summary(state: Dict[str, Any], total_questions: int) -> str:
    """Build a progress summary string."""
    answered = len(state["attempted"])
    skipped_count = len(state["skipped"])
    current_index = state.get("current_index", 0)
    remaining = total_questions - current_index
    
    return (f"📊 Progress: {answered} answered, {skipped_count} skipped, "
            f"{remaining} remaining out of {total_questions} total questions.")


def format_question_for_display(question: Dict[str, Any], index: int, total: int) -> str:
    """Format a question for display to the user."""
    q_text = question.get('text', question.get('question', 'No question text'))
    q_type = question.get('type', 'text')
    
    log.info(f"Formatting question {index}: type={q_type}, text_length={len(q_text)}")
    
    formatted = f"\n**Question {index + 1} of {total}**\n\n{q_text}\n"
    
    # If multiple choice, show options
    if q_type in ['mcq', 'multiple_choice'] and 'options' in question:
        formatted += "\nOptions:\n"
        for i, option in enumerate(question['options']):
            formatted += f"{chr(65 + i)}. {option}\n"
    
    formatted += "\n💡 You can answer, skip, or ask for clarification about this question."
    return formatted


def handle_navigation(
    direction: str,  # "previous" or "next"
    questions: List[Dict[str, Any]],
    quiz_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle navigation to previous or next question.
    """
    current_index = quiz_state["current_index"]
    
    if direction == "previous":
        if current_index == 0:
            response = "You're already at the first question!"
            target_index = 0
        else:
            target_index = current_index - 1
            quiz_state["current_index"] = target_index
            question = questions[target_index]
            formatted_q = format_question_for_display(question, target_index, len(questions))
            
            # Check if this question was already answered
            answer_status = ""
            if str(target_index) in quiz_state["answers"]:
                prev_answer = quiz_state["answers"][str(target_index)]
                if prev_answer == "skipped":
                    answer_status = "\n⚠️ You skipped this question earlier."
                else:
                    answer_status = f"\n📝 Your previous answer: {prev_answer.get('answer', 'N/A')}"
            
            progress = build_quiz_progress_summary(quiz_state, len(questions))
            response = f"⬅️ Going back to the previous question.\n\n{progress}\n{formatted_q}{answer_status}"
    
    else:  # direction == "next"
        if current_index >= len(questions) - 1:
            response = "You're at the last question! Submit your quiz or answer this question."
            target_index = current_index
        else:
            target_index = current_index + 1
            quiz_state["current_index"] = target_index
            question = questions[target_index]
            formatted_q = format_question_for_display(question, target_index, len(questions))
            
            progress = build_quiz_progress_summary(quiz_state, len(questions))
            response = f"➡️ Moving to the next question.\n\n{progress}\n{formatted_q}"
    
    return {
        "response": response,
        "quiz_state": quiz_state,
        "metadata": {
            "total_questions": len(questions),
            "current_question_index": quiz_state["current_index"],
            "is_navigation": True
        }
    }


def handle_clarification_question(
    query: str,
    chatbot_name: str,
    questions: List[Dict[str, Any]],
    quiz_state: Dict[str, Any],
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Handle user questions about the quiz using RAG context.
    Uses functional retrieve_context() and generate_with_retry().
    """
    try:
        # Get relevant context using functional RAG service
        context, _ = retrieve_context(
            chatbot_name=chatbot_name,
            query=query,
            top_k=5,
            min_score=0.5,
            max_context_chars=2000
        )
        
        current_q_index = quiz_state["current_index"]
        current_question = questions[current_q_index] if current_q_index < len(questions) else None
        
        # Get question text
        q_text = ""
        if current_question:
            q_text = current_question.get('text', current_question.get('question', ''))
        
        prompt = f"""You are a helpful quiz assistant. A user is taking a quiz and has a question.

Current Quiz Question: {q_text if q_text else 'N/A'}

Relevant Context from Knowledge Base:
{context if context else "(No additional context available)"}

User's Question: {query}

Provide a helpful, friendly response that clarifies their doubt WITHOUT giving away the answer to the quiz question.
Be encouraging and guide them to think through the question themselves.
Keep your response concise (2-3 sentences max).
"""
        
        # Use functional generate_with_retry
        response_obj = generate_with_retry(
            prompt=prompt,
            conversation_history=conversation_history[-3:] if len(conversation_history) > 3 else conversation_history,
            model_name=config.GEMINI_MODEL,
            max_retries=2
        )
        
        clarification_response = response_obj.text.strip()
        
        # Add reminder
        if current_question:
            full_response = f"{clarification_response}\n\n💡 When you're ready, please provide your answer or type 'skip' to move on."
        else:
            full_response = clarification_response
        
        return {
            "response": full_response,
            "quiz_state": quiz_state,
            "metadata": {
                "total_questions": len(questions),
                "current_question_index": quiz_state["current_index"],
                "is_clarification": True
            }
        }
        
    except Exception as e:
        log.error(f"Clarification handling failed: {e}", exc_info=True)
        return {
            "response": "I'd be happy to help clarify! Could you rephrase your question?",
            "quiz_state": quiz_state,
            "metadata": {
                "total_questions": len(questions),
                "current_question_index": quiz_state["current_index"]
            }
        }

def handle_quiz_completion(
    db: Session,
    chatbot_id: str,
    chatbot_mode: str,
    user_id: Optional[uuid.UUID],
    employee_id: Optional[uuid.UUID],
    quiz_state: Dict[str, Any],
    questions: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Handle quiz completion - save answers to appropriate table.

    NOTE:
    - If an entry already exists, return early and DO NOT create a new entry.
    """
    try:
        quiz_state["completed"] = True

        # Mark unanswered questions as "skipped"
        for idx in range(len(questions)):
            if str(idx) not in quiz_state["answers"]:
                quiz_state["answers"][str(idx)] = {"answer": "skipped"}

        # Convert index-based answers to question ID-based answers
        id_based_answers = {}
        for idx_str, ans_data in quiz_state["answers"].items():
            idx = int(idx_str)
            if idx < len(questions):
                question_id = str(questions[idx]["id"])  # Convert to string for consistency
                id_based_answers[question_id] = ans_data

        # Prepare answer data
        answer_data = {
            "answers": id_based_answers,
            "attempted": quiz_state["attempted"],
            "skipped": quiz_state["skipped"],
            "total_questions": len(questions),
            "completion_time": datetime.utcnow().isoformat()
        }

        chatbot_uuid = uuid.UUID(chatbot_id)

        # Save based on mode
        if chatbot_mode == "people_analyzer":
            # ✅ CHECK IF ENTRY EXISTS
            existing = (
                db.query(PeopleAnalyzer)
                .filter(
                    PeopleAnalyzer.chatbot_id == chatbot_uuid,
                    PeopleAnalyzer.employee_id == employee_id,
                    PeopleAnalyzer.created_by == user_id
                )
                .first()
            )

            if existing:
                log.info(f"✓ People analyzer entry already exists: {existing.id}")
                return {
                    "response": "✅ Assessment already submitted. No new entry was created.",
                    "quiz_state": quiz_state,
                    "metadata": {
                        "completed": True,
                        "already_exists": True,
                        "record_id": str(existing.id),
                        "total_questions": len(questions),
                        "answered": len(quiz_state["attempted"]),
                        "skipped": len(quiz_state["skipped"])
                    }
                }

            # ✅ CREATE NEW ENTRY (only if not exists)
            new_record = PeopleAnalyzer(
                id=uuid.uuid4(),
                chatbot_id=chatbot_uuid,
                employee_id=employee_id,
                answers=id_based_answers,
                created_by=user_id
            )
            db.add(new_record)
            db.commit()
            db.refresh(new_record)

            log.info(f"✓ People analyzer completed and saved: {new_record.id}")
            response = "✅ Assessment completed! Your ratings have been submitted successfully. Thank you!"
            record = new_record

        else:
            # ✅ CHECK IF ENTRY EXISTS
            existing = (
                db.query(Answer)
                .filter(
                    Answer.chatbot_id == chatbot_uuid,
                    Answer.attempter_by_id == user_id
                )
                .first()
            )

            if existing:
                log.info(f"✓ Quiz entry already exists: {existing.id}")
                return {
                    "response": "✅ Quiz already submitted. No new entry was created.",
                    "quiz_state": quiz_state,
                    "metadata": {
                        "completed": True,
                        "already_exists": True,
                        "record_id": str(existing.id),
                        "total_questions": len(questions),
                        "answered": len(quiz_state["attempted"]),
                        "skipped": len(quiz_state["skipped"])
                    }
                }

            # ✅ CREATE NEW ENTRY (only if not exists)
            new_record = Answer(
                id=uuid.uuid4(),
                chatbot_id=chatbot_uuid,
                answer_data=answer_data.get("answers", []),
                attempter_by_id=user_id,
                chat_history=conversation_history or []
            )
            db.add(new_record)
            db.commit()
            db.refresh(new_record)

            log.info(f"✓ Quiz completed and saved: {new_record.id}")
            response = "✅ Quiz completed! Your answers have been submitted successfully. Thank you!"
            record = new_record

        return {
            "response": response,
            "quiz_state": quiz_state,
            "metadata": {
                "completed": True,
                "already_exists": False,
                "record_id": str(record.id),
                "total_questions": len(questions),
                "answered": len(quiz_state["attempted"]),
                "skipped": len(quiz_state["skipped"])
            }
        }

    except Exception as e:
        log.error(f"❌ Completion failed: {e}", exc_info=True)
        db.rollback()
        return {
            "response": "Completed, but there was an error saving your responses. Please contact support.",
            "quiz_state": quiz_state,
            "error": str(e)
        }

# ============================================================================
# MAIN QUIZ SERVICE FUNCTION
# ============================================================================

def quiz_query_service(
    db: Session,
    chatbot_id: str,
    chatbot_name: str,
    chatbot_mode: str,
    query: str,
    conversation_history: List[Dict[str, str]],
    user_id: Optional[uuid.UUID] = None,
    employee_id: Optional[uuid.UUID] = None
) -> Dict[str, Any]:
    """
    Handle quiz/people_analyzer mode queries with intelligent conversation flow.
    
    Args:
        db: Database session
        chatbot_id: Chatbot UUID
        chatbot_name: Name of the chatbot
        chatbot_mode: "quiz" or "people_analyzer"
        query: User's message
        conversation_history: Previous conversation
        user_id: UUID of the user taking quiz/filling assessment
        employee_id: UUID of employee being analyzed (people_analyzer mode only)
        
    Returns:
        Dict with response, quiz_state, and metadata
    """
    try:
        log.info(f"{chatbot_mode} query for '{chatbot_name}': {query[:60]}...")
        
        # Fetch questions
        question_record = db.query(Question).filter(
            Question.chatbot_id == chatbot_id,
            Question.status == "active"
        ).first()
        
        if not question_record or not question_record.question_data:
            return {
                "response": "Sorry, no active questions found for this quiz. Please contact the administrator.",
                "quiz_state": {"completed": True},
                "error": "No questions available"
            }
        
        questions = question_record.question_data
        if not questions:
            return {
                "response": "Sorry, this quiz has no questions configured.",
                "quiz_state": {"completed": True},
                "error": "Empty questions list"
            }
        
        # Extract quiz state from history
        quiz_state = get_quiz_session_from_history(conversation_history)
        
        # Detect user intent (pass chatbot_mode for people_analyzer special handling)
        intent = detect_user_intent(query, chatbot_mode=chatbot_mode)
        log.info(f"Intent: {intent}")
        
        # Check if starting
        is_starting = len(conversation_history) == 0
        
        # Handle greeting at start
        if is_starting and intent == "greeting":
            first_question = questions[0]
            formatted_q = format_question_for_display(first_question, 0, len(questions))
            
            response = (
                f"Hello! 👋 Welcome to the quiz!\n\n"
                f"I'll guide you through {len(questions)} questions. "
                f"You can answer, skip questions, or ask for clarification anytime.\n"
                f"{formatted_q}"
            )
            
            return {
                "response": response,
                "quiz_state": quiz_state,
                "metadata": {
                    "total_questions": len(questions),
                    "current_question_index": 0
                }
            }
        
        # Handle greeting mid-quiz
        if intent == "greeting" and not is_starting:
            current_q_index = quiz_state["current_index"]
            if current_q_index < len(questions):
                current_question = questions[current_q_index]
                formatted_q = format_question_for_display(
                    current_question,
                    current_q_index,
                    len(questions)
                )
                response = f"Hello! Let's continue with your quiz:\n{formatted_q}"
            else:
                response = "Hi! You've completed all questions. Would you like to submit your quiz?"
            
            return {
                "response": response,
                "quiz_state": quiz_state,
                "metadata": {
                    "total_questions": len(questions),
                    "current_question_index": quiz_state["current_index"]
                }
            }
        
        # Handle navigation
        if intent == "navigate_previous":
            return handle_navigation("previous", questions, quiz_state)
        
        if intent == "navigate_next":
            return handle_navigation("next", questions, quiz_state)
        
        # Handle direct question navigation (e.g., "Question 1", "Question 5")
        if intent == "navigate_direct":
            match = re.match(r'question\s+(\d+)', query.lower())
            if match:
                question_num = int(match.group(1))
                target_index = question_num - 1  # Convert to 0-based index
                
                if 0 <= target_index < len(questions):
                    quiz_state["current_index"] = target_index
                    target_question = questions[target_index]
                    formatted_q = format_question_for_display(
                        target_question,
                        target_index,
                        len(questions)
                    )
                    
                    return {
                        "response": formatted_q,
                        "quiz_state": quiz_state,
                        "metadata": {
                            "total_questions": len(questions),
                            "current_question_index": target_index
                        }
                    }
                else:
                    return {
                        "response": f"Invalid question number. Please choose between 1 and {len(questions)}.",
                        "quiz_state": quiz_state,
                        "metadata": {
                            "total_questions": len(questions),
                            "current_question_index": quiz_state["current_index"]
                        }
                    }
        
        # Handle end quiz
        if intent == "end_quiz":
            return handle_quiz_completion(
                db=db,
                chatbot_id=chatbot_id,
                chatbot_mode=chatbot_mode,
                user_id=user_id,
                employee_id=employee_id,
                quiz_state=quiz_state,
                questions=questions,
                conversation_history=conversation_history
            )
        
        # Handle clarification
        if intent == "clarification_question":
            return handle_clarification_question(
                query=query,
                chatbot_name=chatbot_name,
                questions=questions,
                quiz_state=quiz_state,
                conversation_history=conversation_history
            )
        
        # Handle skip
        if intent == "skip":
            current_q_index = quiz_state["current_index"]
            
            # Record skipped answer
            quiz_state["answers"][str(current_q_index)] = "skipped"
            if current_q_index not in quiz_state["skipped"]:
                quiz_state["skipped"].append(current_q_index)
            
            # Move to next question
            quiz_state["current_index"] += 1
            
            if quiz_state["current_index"] >= len(questions):
                return handle_quiz_completion(
                    db=db,
                    chatbot_id=chatbot_id,
                    chatbot_mode=chatbot_mode,
                    user_id=user_id,
                    employee_id=employee_id,
                    quiz_state=quiz_state,
                    questions=questions,
                    conversation_history=conversation_history
                )
            
            next_question = questions[quiz_state["current_index"]]
            formatted_q = format_question_for_display(
                next_question,
                quiz_state["current_index"],
                len(questions)
            )
            
            progress = build_quiz_progress_summary(quiz_state, len(questions))
            response = f"Okay, skipping that question. ⏭️\n\n{progress}\n{formatted_q}"
            
            return {
                "response": response,
                "quiz_state": quiz_state,
                "metadata": {
                    "total_questions": len(questions),
                    "current_question_index": quiz_state["current_index"]
                }
            }
        
        # Handle answer
        if intent == "answer":
            current_q_index = quiz_state["current_index"]
            
            # Validation
            if not query.strip():
                response = "Please provide an answer, or type 'skip' to move to the next question."
                return {
                    "response": response,
                    "quiz_state": quiz_state,
                    "metadata": {
                        "total_questions": len(questions),
                        "current_question_index": quiz_state["current_index"],
                        "validation_failed": True
                    }
                }
            
            # Special handling for people_analyzer mode
            if chatbot_mode == "people_analyzer":
                # Convert typed words to symbols
                word_to_symbol = {
                    "positive": "+",
                    "negative": "-",
                    "average": "+-",
                    "partial": "+-",
                    "partially": "+-",
                    "neutral": "+-",
                    "yes": "+",
                    "no": "-",
                }
                
                query_lower = query.strip().lower()
                if query_lower in word_to_symbol:
                    query = word_to_symbol[query_lower]
                
                # Validate answer is one of the valid symbols
                valid_answers = ["+", "-", "±", "+-", "-+"]
                if query.strip() not in valid_answers:
                    # Not a valid people analyzer answer - treat as clarification request
                    current_question = questions[current_q_index]
                    formatted_q = format_question_for_display(
                        current_question,
                        current_q_index,
                        len(questions)
                    )
                    
                    prompt = f"""You are a professional HR assistant helping employees understand evaluation questions during a People Analyzer assessment.

Current Question: {current_question.get('question', '')}

Employee Query: {query}

IMPORTANT GUIDELINES:
- Provide a clear, supportive response that helps clarify the question
- Be professional, neutral, and unbiased
- Help the employee understand what is being asked
- DO NOT suggest a rating (+, -, ±)
- DO NOT choose or recommend an answer
- DO NOT influence how they should evaluate
- Keep response brief and focused (2-4 sentences)

Respond in a helpful, clarifying manner:"""
                    
                    # Generate a conversational response using Gemini
                    try:
                        model = genai.GenerativeModel(config.GEMINI_MODEL)
                        response_obj = model.generate_content(prompt)
                        gemini_response = response_obj.text.strip()
                    except Exception as e:
                        log.error(f"Error generating Gemini response: {e}")
                        gemini_response = "I understand you have a question. Please feel free to provide your feedback using +, -, or ± based on your assessment."
                    
                    response = f"{gemini_response}\n\n---\n\n{formatted_q}\n\nPlease provide your feedback: + (Positive), - (Scope for improvement), or ± (Neutral)"
                    
                    # CRITICAL: Don't modify quiz_state - preserve all previous answers
                    return {
                        "response": response,
                        "quiz_state": quiz_state,
                        "metadata": {
                            "total_questions": len(questions),
                            "current_question_index": quiz_state["current_index"],
                            "query_handled": True,
                            "is_clarification": True
                        }
                    }
            
            # ========================================================================
            # STORE ANSWER - This is where answers are saved
            # ========================================================================
            # Normalize the answer format (convert -+ to +-)
            normalized_answer = query
            if query.strip() == "-+":
                normalized_answer = "+-"
            
            # Store the answer in quiz_state
            quiz_state["answers"][str(current_q_index)] = {
                "answer": normalized_answer
            }
            
            # Add to attempted list if not already there
            if current_q_index not in quiz_state["attempted"]:
                quiz_state["attempted"].append(current_q_index)
            
            # Move to next question
            quiz_state["current_index"] += 1
            
            log.info(f"✅ Answer stored: Q{current_q_index}='{query}', attempted={quiz_state['attempted']}, all_answers={list(quiz_state['answers'].keys())}")
            
            # Check completion
            if quiz_state["current_index"] >= len(questions):
                return handle_quiz_completion(
                    db=db,
                    chatbot_id=chatbot_id,
                    chatbot_mode=chatbot_mode,
                    user_id=user_id,
                    employee_id=employee_id,
                    quiz_state=quiz_state,
                    questions=questions,
                    conversation_history=conversation_history
                )
            
            # Next question
            next_question = questions[quiz_state["current_index"]]
            formatted_q = format_question_for_display(
                next_question,
                quiz_state["current_index"],
                len(questions)
            )
            
            progress = build_quiz_progress_summary(quiz_state, len(questions))
            
            # Different acknowledgment based on mode
            if chatbot_mode == "people_analyzer":
                response = f"Thank you for your feedback. ✅\n\n{progress}\n{formatted_q}"
            else:
                response = f"Got it! Your answer recorded. ✅\n\n{progress}\n{formatted_q}"
            
            return {
                "response": response,
                "quiz_state": quiz_state,
                "metadata": {
                    "total_questions": len(questions),
                    "current_question_index": quiz_state["current_index"]
                }
            }
        
        # Fallback - should rarely reach here now
        current_q_index = quiz_state["current_index"]
        if current_q_index < len(questions):
            current_question = questions[current_q_index]
            formatted_q = format_question_for_display(
                current_question,
                current_q_index,
                len(questions)
            )
            response = f"Let's continue with the current question:\n{formatted_q}"
        else:
            response = "It looks like we've covered all questions. Would you like to submit your quiz?"
        
        return {
            "response": response,
            "quiz_state": quiz_state,
            "metadata": {
                "total_questions": len(questions),
                "current_question_index": quiz_state["current_index"]
            }
        }
        
    except Exception as e:
        log.error(f"❌ Quiz query failed: {str(e)}", exc_info=True)
        return {
            "response": "Sorry, something went wrong. Please try again!",
            "quiz_state": {},
            "error": str(e)
        }