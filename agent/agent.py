import os
import sys
import json
import PyPDF2
import docx
from openai import OpenAI
from agent.constants import QUIZ_RENDERER_PROMPT
import config
from services.rag_service import generate_with_retry
from utils.document_service import extract_text_from_document, is_supported_document
from utils.logging import log
from fastapi import HTTPException, UploadFile
from typing import List, Dict, Any


def _llm_generate(prompt: str) -> str:
    """Single-turn LLM call via LiteLLM-compatible endpoint. Returns text string."""
    client = OpenAI(
        base_url=config.LITE_LLM_BASE_URL,
        api_key=config.LITE_LLM_API_KEY,
    )
    response = client.chat.completions.create(
        model=config.LITE_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()

def extract_text_from_file(file_path: str) -> str:
    if file_path.lower().endswith(".pdf"):
        return extract_from_pdf(file_path)
    if file_path.lower().endswith(".docx"):
        return extract_from_docx(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_from_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def render_questions_from_file(file_path: str):
    document_text = extract_text_from_file(file_path)

    raw = _llm_generate(QUIZ_RENDERER_PROMPT + "\n\n" + document_text)

    print("RAW OUTPUT FROM MODEL:\n", raw)

    # 🧹 Remove Markdown fenced code block if present
    if raw.startswith("```"):
        raw = raw.strip("`")            # removes ``` from both ends
        raw = raw.replace("json", "", 1).strip()  # remove json tag once

    # Make sure raw starts with [ or { for valid JSON
    raw = raw.strip()

    return json.loads(raw)

def clean_json_response(raw_text: str) -> str:
    """
    Clean LLM response to extract valid JSON.
    Removes markdown code blocks and other artifacts.
    """
    raw = raw_text.strip()
    
    log.info(f"Cleaning JSON response (length: {len(raw)})")
    
    # Remove markdown code blocks
    if raw.startswith("```"):
        # Remove opening ```
        raw = raw[3:]
        # Remove language identifier (json, JSON, etc.)
        if raw.lower().startswith("json"):
            raw = raw[4:]
        # Remove closing ```
        if raw.endswith("```"):
            raw = raw[:-3]
    
    raw = raw.strip()
    
    # Ensure it starts with [ or {
    if not raw.startswith('[') and not raw.startswith('{'):
        # Try to find the first [ or {
        start_bracket = raw.find('[')
        start_brace = raw.find('{')
        
        if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
            raw = raw[start_bracket:]
            log.info("Found JSON array start at position {start_bracket}")
        elif start_brace != -1:
            raw = raw[start_brace:]
            log.info(f"Found JSON object start at position {start_brace}")
        else:
            raise ValueError("No valid JSON structure found in response")
    
    # Ensure it ends properly
    if raw.startswith('['):
        if not raw.endswith(']'):
            end_bracket = raw.rfind(']')
            if end_bracket != -1:
                raw = raw[:end_bracket + 1]
    elif raw.startswith('{'):
        if not raw.endswith('}'):
            end_brace = raw.rfind('}')
            if end_brace != -1:
                raw = raw[:end_brace + 1]
    
    log.info(f"Cleaned JSON (length: {len(raw)})")
    return raw


# ============================================================================
# QUIZ RENDERING FUNCTIONS
# ============================================================================

def render_questions_from_text(document_text: str) -> List[Dict[str, Any]]:
    """
    Use LLM to extract quiz questions from document text.
    Returns structured question data.
    """
    try:
        log.info(f"🤖 Rendering quiz questions from {len(document_text)} chars")
        
        # Build full prompt
        full_prompt = QUIZ_RENDERER_PROMPT + "\n\n" + document_text
        

        response = generate_with_retry(
                prompt=full_prompt,
                conversation_history=[],  # No history needed for quiz rendering
                model_name=config.GEMINI_MODEL,
                max_retries=3
            )
        
        # Get raw output
        raw_output = response.text.strip()
        log.info(f"Raw LLM output length: {len(raw_output)}")
        
        # Clean the response
        cleaned_json = clean_json_response(raw_output)
        
        # Parse JSON
        try:
            questions = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            log.error(f"JSON parsing failed: {e}")
            log.error(f"Cleaned JSON: {cleaned_json[:500]}...")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM response as JSON: {str(e)}"
            )
        
        # Validate it's a list
        if not isinstance(questions, list):
            log.error(f"Expected list, got {type(questions)}")
            raise HTTPException(
                status_code=500,
                detail="LLM did not return a list of questions"
            )
        
        if not questions:
            log.warning("No questions extracted from document")
            return []
        
        log.info(f"Extracted {len(questions)} questions")
        return questions
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Question rendering failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render questions: {str(e)}"
        )


async def render_quiz_questions(file: UploadFile) -> List[Dict[str, Any]]:
    """
    Extract quiz questions from uploaded document.
    
    NO FILE STORAGE - works directly with file bytes.
    
    Args:
        file: Uploaded document file
        
    Returns:
        List of rendered quiz questions
    """
    try:
        # Validate file type
        if not is_supported_document(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Supported: PDF, DOCX, DOC, TXT, RTF"
            )
        
        log.info(f" Processing quiz document: {file.filename}")
        
        # Read file bytes (no storage)
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        log.info(f"File size: {len(file_content)} bytes")
        
        # Extract text using existing utility
        document_text = extract_text_from_document(file_content, file.filename)
        
        if not document_text or not document_text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from {file.filename}"
            )
        
        log.info(f"✓ Extracted {len(document_text)} chars from {file.filename}")
        
        # Render questions using LLM
        questions = render_questions_from_text(document_text)
        
        return questions
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Quiz rendering failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
