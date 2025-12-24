

echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting FastAPI app..."
# You can use uvicorn or gunicorn for production
exec uvicorn main:app --host 0.0.0.0 --port 8000
