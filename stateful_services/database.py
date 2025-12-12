# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# import config

# DATABASE_URL = config.DATABASE_URL

# engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


from sqlalchemy import create_engine, text, text
from sqlalchemy.orm import sessionmaker, declarative_base
import config
import logging

DATABASE_URL = config.DATABASE_URL

# SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db():
    """Yield a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_db_health():
    """
    Perform a simple database health check using SQLAlchemy 2.x style.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logging.info("Database connection successful.")
        return True
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return False
