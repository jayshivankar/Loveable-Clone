import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.session import Session as DBSession
from pydantic import BaseModel
from typing import List

router = APIRouter()

class SessionCreate(BaseModel):
    name: str

class SessionResponse(BaseModel):
    id: str
    name: str

    class Config:
        from_attributes = True

@router.get("/", response_model=List[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    return db.query(DBSession).all()

@router.post("/", response_model=SessionResponse)
def create_session(data: SessionCreate, db: Session = Depends(get_db)):
    # Placeholder user logic
    new_sess = DBSession(
        id=str(uuid.uuid4()), 
        user_id="test_user", 
        name=data.name
    )
    db.add(new_sess)
    db.commit()
    db.refresh(new_sess)
    return new_sess
