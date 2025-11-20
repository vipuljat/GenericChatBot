import os
import sys
import json
import PyPDF2
import docx
import google.generativeai as genai
from agent.constants import QUIZ_RENDERER_PROMPT
import config

# Correct configuration Block
genai.configure(api_key=config.GEMINI_API_KEY)
print(genai.list_models())

# Correct flash model init
model = genai.GenerativeModel(config.GEMINI_MODEL)

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

    response = model.generate_content(
        QUIZ_RENDERER_PROMPT + "\n\n" + document_text
    )

    raw = response.text.strip()

    print("RAW OUTPUT FROM MODEL:\n", raw)

    # 🧹 Remove Markdown fenced code block if present
    if raw.startswith("```"):
        raw = raw.strip("`")            # removes ``` from both ends
        raw = raw.replace("json", "", 1).strip()  # remove json tag once

    # Make sure raw starts with [ or { for valid JSON
    raw = raw.strip()

    return json.loads(raw)
