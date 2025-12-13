"""Database and Qdrant client configuration with health checks."""

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from qdrant_client import QdrantClient

import config
from utils.logging import log


# --------------------------
# DATABASE CONFIGURATION
# --------------------------

Base = declarative_base()


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""
    
    _instance: Optional['DatabaseManager'] = None
    
    def __new__(cls, database_url: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, database_url: str):
        if self._initialized:
            return
        
        self.engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._session_factory = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        self._initialized = True
        log.info("Database manager initialized")
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Create a new database session (for dependency injection)."""
        return self._session_factory()
    
    def check_health(self) -> bool:
        """Verify database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database health check passed")
            return True
        except Exception as e:
            log.error(f"Database health check failed: {e}")
            return False


class QdrantManager:
    """Manages Qdrant vector database connections and operations."""
    
    _instance: Optional['QdrantManager'] = None
    
    def __new__(cls, host: str, port: int):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, host: str, port: int):
        if self._initialized:
            return
        
        self.client: Optional[QdrantClient] = None
        self._initialize(host, port)
        self._initialized = True
    
    def _initialize(self, host: str, port: int) -> None:
        """Initialize Qdrant client connection."""
        try:
            self.client = QdrantClient(host=host, port=port)
        except Exception as e:
            log.error(f"Failed to initialize Qdrant client: {e}")
    
    def check_health(self) -> bool:
        """Verify Qdrant connectivity and list collections."""
        if self.client is None:
            log.error("Qdrant client not initialized")
            return False
        try:
            collections = self.client.get_collections()
            log.info(f"Qdrant health check passed, collections: {[col.name for col in collections.collections]}")
            return True
        except Exception as e:
            log.error(f"Qdrant health check failed: {e}")
            return False
    
    def get_client(self) -> Optional[QdrantClient]:
        """Get the Qdrant client instance."""
        return self.client


# --------------------------
# GLOBAL INSTANCES (Singletons)
# --------------------------

db_manager = DatabaseManager(config.DATABASE_URL)
qdrant_manager = QdrantManager(config.QDRANT_HOST, config.QDRANT_PORT)


# --------------------------
# DEPENDENCY INJECTION HELPERS
# --------------------------

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def get_qdrant() -> Optional[QdrantClient]:
    """Get Qdrant client instance."""
    return qdrant_manager.get_client()


# --------------------------
# HEALTH CHECK FUNCTIONS
# --------------------------
def check_db_health() -> bool:
    """Check database health status.""" 
    return db_manager.check_health()


def check_qdrant_health() -> bool:
    """Check Qdrant health status."""
    return qdrant_manager.check_health()


def check_all_health() -> dict[str, bool]:
    """Check health of all database services."""
    return {
        "database": check_db_health(),
        "qdrant": check_qdrant_health()
    }