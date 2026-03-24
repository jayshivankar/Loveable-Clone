from sqlalchemy import Column, String, DateTime, ForeignKey, text, JSON
from sqlalchemy.orm import relationship
from backend.app.models.base import Base

class Run(Base):
    __tablename__ = "runs"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="PENDING")
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    
    session = relationship("Session", backref="runs")
    user = relationship("User", backref="runs")
