from sqlalchemy import Column, Integer, String, Text, Boolean,ForeignKey
from core.database import Base

class ChatbotDocument(Base):
    __tablename__ = "chatbot_documents"
    id = Column(Integer, primary_key=True)
    chatbot_id = Column(Integer, ForeignKey("chatbots_details.id", ondelete="CASCADE"))
    doc_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    is_quiz_document = Column(Boolean, default=False)

