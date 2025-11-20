import os
from dotenv import load_dotenv

env = os.environ
load_dotenv()

POSTGRES_USER = env.get('POSTGRES_USER', 'root')
POSTGRES_PASSWORD = env.get('POSTGRES_PASSWORD', 'root')
POSTGRES_HOST = env.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = env.get('POSTGRES_PORT', '5432')
POSTGRES_DB = env.get('POSTGRES_DB', 'chatbot_db')

GEMINI_API_KEY=env.get("GEMINI_API_KEY","AIzaSyDvrU8JwghS-KvRNlJEH0hbSVosy77q4q8")
GEMINI_MODEL=env.get("GEMINI_MODEL","models/gemini-2.5-flash")