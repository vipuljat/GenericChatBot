"""
Quiz Mode Service - Handles interactive quiz assessments with intelligent conversation flow.
Functional approach without classes.
"""

import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import google.generativeai as genai
import config
from sqlalchemy.orm import Session
from fastapi import HTTPException
from stateful_services.db_schema import Chatbot, Question, Answer
from services.chatbots_services import get_chatbot_context, _generate_with_retry
from utils.logging import log
import json
import re

genai.configure(api_key=config.GEMINI_API_KEY)


def _detect_user_intent(user_message: str) -> str:
    """
    Detect user intent using Gemini for accurate classification.
    
    Returns:
        "greeting" | "skip" | "end_quiz" | "clarification_question" | "answer"
    """
    try:
        msg_lower = user_message.lower().strip()
        
        # Fast rule-based detection for obvious cases
        # Greeting detection
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(msg_lower == greet or msg_lower.startswith(greet + ' ') for greet in greetings) and len(msg_lower) < 30:
            return "greeting"
        
        # End quiz detection
        end_phrases = ['end quiz', 'finish quiz', 'stop quiz', 'submit quiz', 'i am done', "i'm done", 'submit']
        if any(phrase in msg_lower for phrase in end_phrases):
            return "end_quiz"
        
        # Skip detection - very clear cases
        skip_phrases = ['skip', 'skip this', 'skip it', 'next question', 'pass', 'next']
        if any(msg_lower == phrase or msg_lower.startswith(phrase + ' ') for phrase in skip_phrases):
            return "skip"
        
        # For ambiguous cases, use Gemini for classification
        prompt = f"""Classify the user's intent in a quiz context.

User message: "{user_message}"

Classify as ONE of these intents:
1. "clarification_question" - User is asking for help, explanation, or clarification about the question (e.g., "explain this", "what does this mean?", "I don't understand", "can you clarify?", "i don't get it")
2. "skip" - User wants to skip to next question (e.g., "skip", "don't know", "I don't know")
3. "answer" - User is providing an answer (e.g., "A", "option B", "I think it's C", any actual answer attempt)
4. "end_quiz" - User wants to end the quiz 

Respond with ONLY ONE WORD: clarification_question, skip, answer, or end_quiz.

Examples:
"explain" -> clarification_question
"i don't get the question" -> clarification_question
"can you explain the question" -> clarification_question
"clarify" -> clarification_question
"what does this mean?" -> clarification_question
"I don't understand" -> clarification_question
"don't know" -> skip
"I don't know" -> skip
"A" -> answer
"option B" -> answer
"I think it's C" -> answer
"The answer is Paris" -> answer
"I'm done" -> end_quiz

"""
        
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        detected_intent = response.text.strip().lower()
        
        # Validate response
        valid_intents = ['clarification_question', 'skip', 'answer', 'end_quiz']
        if detected_intent in valid_intents:
            log.info(f"Gemini detected intent: {detected_intent} for message: '{user_message[:50]}'")
            return detected_intent
        
        # Fallback to answer if invalid response
        log.warning(f"Invalid intent from Gemini: {detected_intent}, defaulting to 'answer'")
        return "answer"
        
    except Exception as e:
        log.error(f"Intent detection failed: {e}, defaulting to 'answer'")
        # Safe fallback
        return "answer"


def _get_quiz_session_from_history(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
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
    
    # Parse conversation history to rebuild state
    current_index = 0
    attempted = []
    skipped = []
    answers = {}
    
    for i, msg in enumerate(conversation_history):
        if msg.get("role") in ["model", "assistant", "bot"]:
            content = msg.get("content", "")
            
            # Check for question markers
            match = re.search(r'Question (\d+) of (\d+)', content)
            if match:
                question_num = int(match.group(1))
                current_index = question_num - 1
            
            # Check for skip confirmation
            if "skipping that question" in content.lower():
                # The question that was skipped is the one before current
                if current_index > 0:
                    skipped_idx = current_index - 1
                    if skipped_idx not in skipped:
                        skipped.append(skipped_idx)
            
            # Check for answer recorded
            if "answer recorded" in content.lower():
                # The question that was answered is the one before current
                if current_index > 0:
                    answered_idx = current_index - 1
                    if answered_idx not in attempted:
                        attempted.append(answered_idx)
                        # Try to find the user's answer in the previous message
                        if i > 0:
                            prev_msg = conversation_history[i-1]
                            if prev_msg.get("role") == "user":
                                answers[str(answered_idx)] = {
                                    "answer": prev_msg.get("content", ""),
                                    "timestamp": datetime.utcnow().isoformat()
                                }
    
    state["current_index"] = current_index
    state["attempted"] = attempted
    state["skipped"] = skipped
    state["answers"] = answers
    
    log.info(f"Reconstructed quiz state: index={current_index}, attempted={len(attempted)}, skipped={len(skipped)}")
    
    return state


def _build_quiz_progress_summary(state: Dict[str, Any], total_questions: int) -> str:
    """Build a progress summary string."""
    answered = len(state["attempted"])
    skipped_count = len(state["skipped"])
    current_index = state.get("current_index", 0)
    remaining = total_questions - current_index
    
    return (f"📊 Progress: {answered} answered, {skipped_count} skipped, "
            f"{remaining} remaining out of {total_questions} total questions.")


def _format_question_for_display(question: Dict[str, Any], index: int, total: int) -> str:
    """Format a question for display to the user."""
    # Support both 'question' and 'text' keys
    q_text = question.get('text', question.get('question', 'No question text'))
    q_type = question.get('type', 'text')
    
    log.info(f"Formatting question {index}: type={q_type}, text_length={len(q_text)}")
    
    formatted = f"\n📝 **Question {index + 1} of {total}**\n\n{q_text}\n"
    
    # If it's multiple choice (mcq or multiple_choice), show options
    if q_type in ['mcq', 'multiple_choice'] and 'options' in question:
        formatted += "\nOptions:\n"
        for i, option in enumerate(question['options']):
            formatted += f"{chr(65 + i)}. {option}\n"
    
    formatted += "\n💡 You can answer, skip, or ask for clarification about this question."
    
    return formatted


def quiz_query_service(
    db: Session,
    chatbot_id: str,
    chatbot_name: str,
    query: str,
    conversation_history: List[Dict[str, str]],
    user_id: Optional[uuid.UUID] = None
) -> Dict[str, Any]:
    """
    Handle quiz mode queries with intelligent conversation flow.
    
    Args:
        db: Database session
        chatbot_id: Chatbot UUID
        chatbot_name: Name of the chatbot
        query: User's message
        conversation_history: Previous conversation
        user_id: UUID of the user taking the quiz
        
    Returns:
        Dict with response, quiz_state, and metadata
    """
    try:
        log.info(f"Processing quiz query for chatbot '{chatbot_name}': {query[:60]}...")
        
        # Fetch questions for this chatbot
        question_record = db.query(Question).filter(
            Question.chatbot_id == chatbot_id,
            Question.status == "active"
        ).first()
        
        # if not question_record or not question_record.question_data:
        #     return {
        #         "response": "Sorry, no active questions found for this quiz. Please contact the administrator.",
        #         "quiz_state": {"completed": True},
        #         "error": "No questions available"
        #     }
        
        questions = question_record.get('question_data', []) if question_record else []
        # if not questions:
        #     return {
        #         "response": "Sorry, this quiz has no questions configured.",
        #         "quiz_state": {"completed": True},
        #         "error": "Empty questions list"
        #     }
        
        # Extract or initialize quiz state
        quiz_state = _get_quiz_session_from_history(conversation_history)
        
        # Detect user intent
        intent = _detect_user_intent(query)
        log.info(f"Detected intent: {intent}")
        
        # Check if quiz is starting (no history or first message)
        is_starting = len(conversation_history) == 0
        
        # Handle greeting at start only
        if is_starting and intent == "greeting":
            first_question = questions[0]
            formatted_q = _format_question_for_display(first_question, 0, len(questions))
            
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
        
        # If not starting but user greets, just acknowledge and continue
        if intent == "greeting" and not is_starting:
            current_q_index = quiz_state["current_index"]
            if current_q_index < len(questions):
                current_question = questions[current_q_index]
                formatted_q = _format_question_for_display(
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
        
        # Handle end quiz intent
        if intent == "end_quiz":
            return _handle_quiz_completion(
                db=db,
                chatbot_id=chatbot_id,
                user_id=user_id,
                quiz_state=quiz_state,
                questions=questions
            )
        
        # Handle clarification questions (user asking about the question/quiz)
        if intent == "clarification_question":
            return _handle_clarification_question(
                query=query,
                chatbot_name=chatbot_name,
                questions=questions,
                quiz_state=quiz_state,
                conversation_history=conversation_history
            )
        
        # Handle skip
        if intent == "skip":
            quiz_state["skipped"].append(quiz_state["current_index"])
            quiz_state["current_index"] += 1
            
            if quiz_state["current_index"] >= len(questions):
                return _handle_quiz_completion(
                    db=db,
                    chatbot_id=chatbot_id,
                    user_id=user_id,
                    quiz_state=quiz_state,
                    questions=questions
                )
            
            next_question = questions[quiz_state["current_index"]]
            formatted_q = _format_question_for_display(
                next_question,
                quiz_state["current_index"],
                len(questions)
            )
            
            progress = _build_quiz_progress_summary(quiz_state, len(questions))
            
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
            
            # Simple validation - just check not empty
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
            
            # Store the answer
            quiz_state["answers"][str(current_q_index)] = {
                "answer": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            quiz_state["attempted"].append(current_q_index)
            quiz_state["current_index"] += 1
            
            # Check if quiz is complete
            if quiz_state["current_index"] >= len(questions):
                return _handle_quiz_completion(
                    db=db,
                    chatbot_id=chatbot_id,
                    user_id=user_id,
                    quiz_state=quiz_state,
                    questions=questions
                )
            
            # Present next question
            next_question = questions[quiz_state["current_index"]]
            formatted_q = _format_question_for_display(
                next_question,
                quiz_state["current_index"],
                len(questions)
            )
            
            progress = _build_quiz_progress_summary(quiz_state, len(questions))
            
            response = f"Got it! ✅ Answer recorded.\n\n{progress}\n{formatted_q}"
            
            return {
                "response": response,
                "quiz_state": quiz_state,
                "metadata": {
                    "total_questions": len(questions),
                    "current_question_index": quiz_state["current_index"]
                }
            }
        
        # Fallback
        current_q_index = quiz_state["current_index"]
        if current_q_index < len(questions):
            current_question = questions[current_q_index]
            formatted_q = _format_question_for_display(
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
        log.error(f"Quiz query failed: {str(e)}", exc_info=True)
        return {
            "response": "Sorry, something went wrong. Please try again!",
            "quiz_state": {},
            "error": str(e)
        }


def _handle_clarification_question(
    query: str,
    chatbot_name: str,
    questions: List[Dict[str, Any]],
    quiz_state: Dict[str, Any],
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Handle user questions about the quiz or current question using RAG context.
    """
    try:
        # Get relevant context from chatbot's knowledge base
        context = get_chatbot_context(
            chatbot_name=chatbot_name,
            query=query,
            max_context_length=2000,
            min_score=0.5
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
        
        response_obj = _generate_with_retry(
            prompt=prompt,
            conversation_history=conversation_history[-3:] if len(conversation_history) > 3 else conversation_history,
            model_name=config.GEMINI_MODEL,
            max_retries=2
        )
        
        clarification_response = response_obj.text.strip()
        
        # Add reminder to answer when ready
        if current_question:
            full_response = f"{clarification_response}\n\nWhen you're ready, please provide your answer or type 'skip' to move on."
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


def _handle_quiz_completion(
    db: Session,
    chatbot_id: str,
    user_id: Optional[str],
    quiz_state: Dict[str, Any],
    questions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Handle quiz completion - save answers to database WITHOUT showing summary to user.
    """
    try:
        quiz_state["completed"] = True
        
        # Prepare answer data
        answer_data = {
            "answers": quiz_state["answers"],
            "attempted": quiz_state["attempted"],
            "skipped": quiz_state["skipped"],
            "total_questions": len(questions),
            "completion_time": datetime.utcnow().isoformat()
        }
       
        # Save to database
        new_answer = Answer(
            id=uuid.uuid4(),
            chatbot_id=uuid.UUID(chatbot_id),
            answer_data=answer_data,
            attempter_by_id=user_id,
            chat_history=[]  # Can store full conversation if needed
        )
        
        db.add(new_answer)
        db.commit()
        db.refresh(new_answer)
        
        log.info(f"Quiz completed and saved with ID: {new_answer.id}")
        
        # Simple completion message without detailed summary
        response = "Quiz completed! Your answers have been submitted successfully. Thank you!"
        
        return {
            "response": response,
            "quiz_state": quiz_state,
            "metadata": {
                "completed": True,
                "quiz_id": str(new_answer.id),
                "total_questions": len(questions),
                "answered": len(quiz_state["attempted"]),
                "skipped": len(quiz_state["skipped"])
            }
        }
        
    except Exception as e:
        log.error(f"Quiz completion failed: {e}", exc_info=True)
        db.rollback()
        return {
            "response": "Quiz completed, but there was an error saving your responses. Please contact support.",
            "quiz_state": quiz_state,
            "error": str(e)
        }