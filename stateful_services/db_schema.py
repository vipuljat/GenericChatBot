from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from stateful_services.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    employee_id = Column(String, nullable=False, unique=True)
    employee_name = Column(String, nullable=False)
    employee_role = Column(String, nullable=False)
    department = Column(String, nullable=True)
    meta_data = Column(JSONB, nullable=True)  # changed to JSONB
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Chatbot(Base):
    __tablename__ = "chatbots"

    chatbot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    instruction = Column(Text, nullable=True)
    pdf_name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="draft")
    generated_by = Column(String, nullable=True)
    meta_data = Column(JSONB, nullable=True)  # changed to JSONB
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Question(Base):
    __tablename__ = "questions"

    question_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.chatbot_id", ondelete="CASCADE"))
    status = Column(String, nullable=False, default="active")
    question_data = Column(JSONB, nullable=False)  # changed to JSONB
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Answer(Base):
    __tablename__ = "answers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempter_id = Column(UUID(as_uuid=True), ForeignKey("employees.id", ondelete="SET NULL"), nullable=True)
    attempter_name = Column(String, nullable=True)
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.chatbot_id", ondelete="CASCADE"))
    answer_data = Column(JSONB, nullable=False)  # changed to JSONB
    created_at = Column(DateTime(timezone=True), server_default=func.now())
