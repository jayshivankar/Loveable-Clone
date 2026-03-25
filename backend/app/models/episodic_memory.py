from datetime import UTC, datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column


class EpisodicMemory(SQLModel, table=True):
    __tablename__ = "episodic_memory"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True) # Assuming string ID for users based on user.py
    session_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    app_name: str = Field(default="")
    techstack: str = Field(default="")
    description: str = Field(default="")
    file_count: int = Field(default=0)
    feature_count: int = Field(default=0)
    retry_count: int = Field(default=0)

    quality_score: Optional[int] = Field(default=None)
    passed: Optional[bool] = Field(default=None)
    summary: str = Field(default="")
    issues_json: str = Field(default="[]")
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)

    # pgvector column — not a standard SQLModel field
    embedding: Optional[list] = Field(
        default=None,
        sa_column=Column(Vector(1536))
    )
