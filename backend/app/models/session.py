from sqlalchemy import Column, String, DateTime, ForeignKey, text
from sqlalchemy.orm import relationship
from backend.app.models.base import Base

class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    
    user = relationship("User", backref="sessions")
