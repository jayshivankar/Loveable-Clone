from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.run import Run as DBRun
from backend.app.models.user import User
from backend.app.models.session import Session as DBSession
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

from pydantic import BaseModel, Field
from typing import Optional, Literal
from langgraph.types import Command
import uuid
import asyncio
from fastapi.responses import StreamingResponse
import json
from backend.app.core.langgraph.Workflow import get_compiled_app
from backend.app.models.approval import Approval

class TaskPlanUpdate(BaseModel):
    steps: list

class ResumePayload(BaseModel):
    action: Literal["approve", "edit"]
    edited_plan: Optional[dict] = None

@router.post("/", response_model=RunResponse)
def create_run(data: RunCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == "test_user").first()
    if not user:
        user = User(id="test_user", provider="test", provider_user_id="test_provider_id", email="test@example.com", display_name="Test User")
        db.add(user)
        db.commit()

    session = db.query(DBSession).filter(DBSession.id == data.session_id).first()
    if not session:
        session = DBSession(id=data.session_id, user_id="test_user", name="Test Session")
        db.add(session)
        db.commit()

    thread_id = str(uuid.uuid4())
    new_run = DBRun(
        id=str(uuid.uuid4()),
        session_id=data.session_id,
        user_id="test_user",
        thread_id=thread_id,
        status="IN_PROGRESS"
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)
    return new_run

@router.post("/{run_id}/resume")
async def resume_run(run_id: str, payload: ResumePayload, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status != "AWAITING_APPROVAL":
        # Maybe it's already IN_PROGRESS or DONE, but let's just proceed
        pass

    if payload.action == "edit" and payload.edited_plan is None:
        raise HTTPException(status_code=400, detail="edited_plan is required when action is edit")

    # Record Approval Audit trail
    approval = Approval(
        id=str(uuid.uuid4()),
        run_id=run.id,
        thread_id=run.thread_id,
        user_id=run.user_id,
        action=payload.action
    )
    db.add(approval)
    run.status = "IN_PROGRESS"
    db.commit()

    app = await get_compiled_app()
    config = {"configurable": {"thread_id": run.thread_id}}
    
    # We will forward streamed events as SSE
    async def event_generator():
        resume_data = {"action": payload.action}
        if payload.edited_plan:
            resume_data["edited_plan"] = payload.edited_plan
        
        yield f"data: {json.dumps({'event': 'hitl.resumed', 'action': payload.action})}\n\n"
        
        async for chunk in app.astream(Command(resume=resume_data), config=config, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                if "__interrupt__" in state_update:
                    yield f"data: {json.dumps({'event': 'hitl.required', 'task_plan': state_update.get('task_plan')})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'node.completed', 'node': node_name})}\n\n"
        
        yield f"data: {json.dumps({'event': 'run.completed'})}\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/{run_id}/stream")
async def stream_run(run_id: str, prompt: Optional[str] = None, db: Session = Depends(get_db)):
    run = db.query(DBRun).filter(DBRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    app = await get_compiled_app()
    config = {"configurable": {"thread_id": run.thread_id}}
    
    # Needs initial invoke parameters, we will fetch prompt from somewhere or assume it runs if no state
    # Actually wait. `create_run` didn't save the prompt to the thread yet via langgraph app.
    # In `stream_run` we should initialize if not run.
    
    async def event_generator():
        yield f"data: {json.dumps({'event': 'run.started', 'run_id': run.id})}\n\n"
        
        # Check if state exists, if not we start with user_prompt
        state = await app.aget_state(config)
        
        input_data = None
        if not state.values:
            input_data = {
                "user_prompt": prompt or "Continue working",
                "user_id": run.user_id,
                "session_id": run.session_id
            }

        interrupted = False
        async for chunk in app.astream(input_data, config=config, stream_mode="updates"):
            for node_name, state_update in chunk.items():
                if "__interrupt__" in state_update:
                    interrupted = True
                    task_plan = state_update[0].value.get("task_plan") if isinstance(state_update, tuple) else state_update.get("__interrupt__", [{}])[0].value.get("task_plan")
                    
                    full_state = await app.aget_state(config)
                    tp = full_state.values.get("task_plan")
                    
                    db.refresh(run)
                    if run.status != "AWAITING_APPROVAL":
                        run.status = "AWAITING_APPROVAL"
                        db.commit()
                        
                    yield f"data: {json.dumps({'event': 'hitl.required', 'task_plan': tp})}\n\n"
                else:
                    yield f"data: {json.dumps({'event': 'node.completed', 'node': node_name})}\n\n"
                    
        if not interrupted:
            yield f"data: {json.dumps({'event': 'run.completed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/{run_id}/approve")
def approve_run(run_id: str, db: Session = Depends(get_db)):
    # Legacy wrapper
    raise HTTPException(status_code=400, detail="Use /resume endpoint instead")

@router.post("/{run_id}/reject")
def reject_run(run_id: str, db: Session = Depends(get_db)):
    # Legacy wrapper
    raise HTTPException(status_code=400, detail="Use /resume endpoint instead")
