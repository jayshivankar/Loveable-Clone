from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.database import engine
from backend.app.models.base import Base
from backend.app.api.v1 import auth, sessions, runs
import backend.app.models  # Ensures all models are imported before create_all

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME)

from backend.app.core.usage_metering import UsageMeteringMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(UsageMeteringMiddleware)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
app.include_router(runs.router, prefix="/api/v1/runs", tags=["Runs"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
