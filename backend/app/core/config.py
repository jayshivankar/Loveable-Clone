import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env explicitly from the project root
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))
load_dotenv(dotenv_path=env_path)

class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "CodeForge")
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+psycopg2://postgres:postgres@localhost:5432/codeforge"
    )
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change_me")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
settings = Settings()
