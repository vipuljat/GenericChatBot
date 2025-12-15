
QUIZ_RENDERER_PROMPT = """
{
  "prompt_title": "Quiz JSON Rendering Agent",
  "instructions": [
    "Read the entire document content provided by the user.",
    "Identify ALL quiz questions in the document.",
    "Convert each question into a clean JSON structure.",
    "Do NOT generate new questions or modify the meaning.",
    "Only extract questions exactly as written.",
    "Output MUST be a valid JSON array. No text outside JSON."
    "Use Mixed Question Types: MCQ, Open-Text, True/False, Rating.",
    "Ensure variety in question types throughout the quiz.",
    "Generate 20-20 questions maximum."
  ],
  "question_type_rules": {
    "mcq": "If the question contains options (A/B/C/D) or a predefined list of choices.",
    "open-text": "If it asks for an explanation or is open-ended.",
    "rating": "If it asks the user to rate proficiency or satisfaction (usually on a scale like 1-5).",
    "default": "open-text"
  },
  "output_format_template": [
    {
      "id": 0,
      "text": "<question_text>",
      "type": "open-text",
      "order": 1,
      "options": null,
      "category": "<inferred_category>"
    },
    {
      "id": 1,
      "text": "<question_text>",
      "type": "mcq",
      "order": 2,
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "category": "<inferred_category>"
    },
    {
      "id": 2,
      "text": "<question_text> (True/False)",
      "type": "tf",
      "order": 3,
      "options": null,
      "category": "<inferred_category>"
    },
    {
      "id": 3,
      "text": "Rate your proficiency in X on a scale of 1 to 5.",
      "type": "rating",
      "order": 4,
      "options": [1, 2, 3, 4, 5],
      "category": "<inferred_category>"
    }
  ],
  "important_notes": [
    "Use sequential numbering starting from '0' for the 'id' field.",
    "The 'order' field should also be sequential, starting from '1'.",
    "The 'options' field is only used for 'mcq' and 'rating' types. For 'open-text', 'tf', and 'plusminus', it must be null.",
    "The 'category' field must be inferred based on the question's content (e.g., 'Time Management', 'Tools', 'Decision Making')."
  ]
}

Output a JSON array of all questions.
"""