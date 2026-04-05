from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from src.database import get_db
from src.models import User

app = FastAPI()

@app.post("/users/")
def create_user(username: str, password: str, db: Session = Depends(get_db)):
    User.validate_username(username)
    User.validate_password(password)
    new_user = User(username=username, password=password)
    db.add(new_user)
    try:
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error occurred during user creation.")
    db.refresh(new_user)
    return new_user

@app.get("/users/{user_id}")
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
