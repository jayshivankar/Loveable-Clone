from sqlalchemy import Column, String, DateTime, ForeignKey, text
from sqlalchemy.orm import relationship
from backend.app.models.base import Base

class Approval(Base):
    __tablename__ = "approvals"

    id = Column(String, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False, index=True)
    thread_id = Column(String, nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    action = Column(String, nullable=False) # e.g. "approve" or "edit"
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))
    
    run = relationship("Run", backref="approvals")
    user = relationship("User", backref="approvals")
