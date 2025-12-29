#!/bin/sh
set -e

echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting FastAPI app..."
exec python3 main.py

