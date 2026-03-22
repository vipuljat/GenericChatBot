import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration - Use DATABASE_URL directly
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://vipul:vipul123@127.0.0.1:5432/GenericChatbotDB") 
# Google Gemini AI Configuration
# Use a stable default model name; avoid experimental unless explicitly configured.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
 
# Microsoft Azure AD SSO Configuration
MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
MICROSOFT_REDIRECT_URI = os.getenv("MICROSOFT_REDIRECT_URI")
MICROSOFT_AUTHORITY = os.getenv("MICROSOFT_AUTHORITY")
 
# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))
 
# OpenAI Configuration for Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
 
# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://ask-api-dev.47billion.com")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# RAG Retrieval Configuration
RAG_CONTEXT_SCORE_FLOOR = float(os.getenv("RAG_CONTEXT_SCORE_FLOOR", "0.4"))

# Multi-Query Expansion — use LLM to generate vocabulary-variant queries
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "gemini-2.0-flash")
QUERY_EXPANSION_MAX_VARIANTS = int(os.getenv("QUERY_EXPANSION_MAX_VARIANTS", "3"))

# HyDE (Hypothetical Document Embeddings) — embed a generated answer instead of the raw query
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"
HYDE_MODEL = os.getenv("HYDE_MODEL", "gemini-2.0-flash")

# Conversation history sent to LLM — keep only the last N messages.
# 10 messages = 5 user+bot turns. Older turns add tokens without value for RAG.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "20"))

