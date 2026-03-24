from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class UserOut(BaseModel):
    id: str
    email: str
    display_name: str | None = None

@router.get("/me", response_model=UserOut)
def get_current_user():
    # Placeholder for auth logic
    return UserOut(id="test_user", email="test@example.com", display_name="Test User")
