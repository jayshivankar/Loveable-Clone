from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.run import Run as DBRun
from pydantic import BaseModel
import uuid

router = APIRouter()

class RunCreate(BaseModel):
    session_id: str
    prompt: str

class RunResponse(BaseModel):
    id: str
    session_id: str
    status: str

    class Config:
        from_attributes = True

class TaskPlanUpdate(BaseModel):
    steps: list

@router.post("/", response_model=RunResponse)
def create_run(data: RunCreate, db: Session = Depends(get_db)):
    new_run = DBRun(
        id=str(uuid.uuid4()),
        session_id=data.session_id,
        user_id="test_user",
        status="IN_PROGRESS"
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    # Trigger graph execution background task
    return new_run

@router.post("/{run_id}/approve")
def approve_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.status = "IN_PROGRESS"
    db.commit()
    # Logic to resume graph run with ACCEPTED status
    return {"status": "accepted"}

@router.post("/{run_id}/reject")
def reject_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.status = "AWAITING_EDIT"
    db.commit()
    return {"status": "rejected"}

@router.put("/{run_id}/task-plan")
def edit_task_plan(run_id: str, plan_update: TaskPlanUpdate, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    # Logic to save the plan back to the LangGraph thread state
    return {"status": "plan_updated"}

@router.post("/{run_id}/resume")
def resume_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.status = "IN_PROGRESS"
    db.commit()
    # Resume graph execution with EDITED status
    return {"status": "resumed"}

from fastapi.responses import StreamingResponse
import asyncio

@router.get("/{run_id}/stream")
async def stream_run(run_id: str, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        # Placeholder for LangGraph state streaming
        for i in range(3):
            yield f"data: {{\"event\": \"node.started\", \"node\": \"step_{i}\"}}\n\n"
            await asyncio.sleep(1)
        yield f"data: {{\"event\": \"run.completed\"}}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")
