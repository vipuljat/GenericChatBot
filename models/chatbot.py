from sqlalchemy import Column, Integer, String, Text, Boolean, ARRAY
from stateful_services.database  import Base

class Chatbot(Base):
    __tablename__ = "chatbots_details"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    chatbot_name = Column(String, nullable=False)
    description = Column(String,nullable=False)
    instructions = Column(String, nullable=False)
    conversation_starters = Column(ARRAY(String),nullable=True)
    is_quiz_mode = Column(Boolean, default=False,nullable=True)
    is_active = Column(Boolean, default=True)
    recommended_model = Column(String(100), nullable=True)

