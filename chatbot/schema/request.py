from pydantic import BaseModel
from typing import List, Optional

class ChatbotCreate(BaseModel):
    chatbot_name: str
    description: str
    instructions: str
    conversation_starters: Optional[List[str]]
    is_quiz_mode: bool = False
    is_active: bool = True
    recommended_model: Optional[str]