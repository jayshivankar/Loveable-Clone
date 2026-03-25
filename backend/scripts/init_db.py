import sys
import os
import asyncio
from sqlmodel import SQLModel

# Ensure the root project directory is in PYTHONPATH so 'backend' is recognized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.core.database import engine
from backend.app.models.episodic_memory import EpisodicMemory
from sqlalchemy import text

async def init_db():
    print("Initializing Database...")
    # Using sync engine for table creation as currently configured in core.database
    with engine.begin() as conn:
        # Create pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # The rest of the setup (SQLModel) can safely run here
        SQLModel.metadata.create_all(conn)

        # Create indexes (if not created by SQLModel logic, we run raw SQL)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_episodic_user_time
                ON episodic_memory (user_id, created_at DESC);
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_episodic_embedding
                ON episodic_memory USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
        """))
    print("Database Initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())
