"""
CodeForge AI — FastAPI Application

Endpoints:
  POST /api/v1/generate          — start a generation job
  GET  /api/v1/stream/{job_id}   — SSE live log stream
  GET  /api/v1/status/{job_id}   — job status + results
  GET  /api/v1/download/local/{job_id} — fallback local download (if S3 not configured)
  GET  /api/v1/memory            — recent episodic memory episodes
  GET  /api/v1/metrics/history   — chart data for frontend
  GET  /                         — serves the frontend SPA
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from Agents.logger import setup_logging, get_logger

load_dotenv()
# Initialize JSON logging configuration
setup_logging("INFO")
log = get_logger("codeforge.api")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CodeForge AI",
    description="AI-powered multi-agent code generation platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (since we stream logs from memory for now) ────────────

jobs: dict[str, dict[str, Any]] = {}

from Agents.logger import register_log_callback

def _on_log(event_dict):
    sid = event_dict.get("session_id")
    msg = event_dict.get("msg") or event_dict.get("event")
    node = event_dict.get("node_name", "system")
    ts = event_dict.get("timestamp")
    if sid and sid in jobs and msg:
        if not ts:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
        jobs[sid]["logs"].append({
            "ts": ts,
            "msg": msg,
            "agent": node
        })

register_log_callback(_on_log)

# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    # Episodic memory
    try:
        from Agents.memory import init_episodic_memory
        init_episodic_memory()
    except Exception as exc:
        log.error("startup_memory_error", error=str(exc))

    # RAG retriever
    try:
        from Agents.RAG.rag import build_retriever
        from Agents.tools import set_retriever
        set_retriever(build_retriever())
    except Exception as exc:
        log.error("startup_rag_error", error=str(exc))

    # Pre-compile LangGraph
    try:
        from Agents.Workflow import get_app
        await get_app()
    except Exception as exc:
        log.error("startup_workflow_error", error=str(exc))

    log.info("CodeForge AI is ready.")

@app.on_event("shutdown")
async def _shutdown() -> None:
    # Need to close the pool when the application stops
    try:
        import Agents.Workflow as workflow
        if workflow._pool is not None:
            await workflow._pool.close()
    except Exception as exc:
        log.error("shutdown_error", error=str(exc))


# ── Workflow runner ────────────────────────────────────────────────────────────

async def _run_workflow_async(job_id: str, prompt: str, fixer_enabled: bool) -> None:
    try:
        from Agents.memory import store_episode
        from Agents.tools import init_project_root
        from Agents.Workflow import get_app

        init_project_root(job_id)

        flow = await get_app(fixer_enabled)
        
        # Async invoke
        config = {"configurable": {"thread_id": job_id}, "recursion_limit": 200}
        
        # Run workflow
        result = await flow.ainvoke(
            {
                "user_prompt":     prompt,
                "chat_session_id": job_id,
                "fixer_enabled":   fixer_enabled,
            },
            config
        )

        review = result.get("review_result")
        plan   = result.get("plan")

        if review and plan:
            try:
                # App name stored if available
                app_name = plan.name if plan else "project"
                store_episode(job_id, prompt, app_name, review)
            except Exception as exc:
                log.error("store_episode_error", session_id=job_id, error=str(exc))

        jobs[job_id].update({
            "status":            "done",
            "zip_url":           result.get("zip_url"),
            "project_structure": result.get("project_structure", []),
            "memory_context":    result.get("memory_context", ""),
            "plan_name":         plan.name        if plan   else "",
            "plan_techstack":    plan.techstack   if plan   else "",
            "review": {
                "score":   review.quality_score if review else None,
                "passed":  review.passed        if review else None,
                "summary": review.summary       if review else None,
                "issues":  len(review.issues)   if review else 0,
                "issue_list": [
                    {
                        "filepath":    i.filepath,
                        "severity":    i.severity.value,
                        "description": i.description,
                    }
                    for i in review.issues
                ] if review else [],
            } if review else None,
        })
        
        # Append termination marker
        jobs[job_id]["logs"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": f"__STATUS__done__",
            "agent": "system",
        })

    except Exception as exc:
        log.error("workflow_error", session_id=job_id, error=str(exc))
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(exc)
        jobs[job_id]["logs"].append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "msg": f"__STATUS__error__",
            "agent": "system",
        })
    finally:
        pass


# ── Schemas ────────────────────────────────────────────────────────────────────
import re
_UNSAFE_RE = re.compile(
    r"(rm\s+-rf|sudo\s|format\s+c:|drop\s+table|truncate\s+table|delete\s+from\s)",
    re.IGNORECASE,
)

class GenerateRequest(BaseModel):
    prompt:        str  = Field(..., min_length=10, max_length=2000)
    fixer_enabled: bool = Field(True)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    if _UNSAFE_RE.search(req.prompt):
        raise HTTPException(400, "Unsafe prompt detected.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "logs": [], "prompt": req.prompt}

    # Start the async workflow in the background
    background_tasks.add_task(_run_workflow_async, job_id, req.prompt, req.fixer_enabled)

    return {"job_id": job_id, "status": "running"}


@app.get("/api/v1/stream/{job_id}")
async def stream_logs(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def _gen():
        last = 0
        while True:
            job  = jobs.get(job_id, {})
            logs = job.get("logs", [])
            for entry in logs[last:]:
                yield f"data: {json.dumps(entry)}\n\n"
                last += 1
            status = job.get("status", "running")
            # If the job finishes we exit (the termination marker will be yielded above)
            if status in ("done", "error"):
                break
            await asyncio.sleep(0.5)
            
            # Send heartbeat keep-alive
            yield f": heartbeat\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "job_id":            job_id,
        "status":            job["status"],
        "project_structure": job.get("project_structure", []),
        "review":            job.get("review"),
        "memory_context":    job.get("memory_context", ""),
        "plan_name":         job.get("plan_name", ""),
        "plan_techstack":    job.get("plan_techstack", ""),
        "zip_url":           job.get("zip_url"),
        "error":             job.get("error"),
    }


@app.get("/api/v1/download/local/{job_id}")
async def download_local_zip(job_id: str):
    # This acts as a fallback if S3 is not configured
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Generation not complete.")
    
    app_name = job.get("plan_name", "project")
    zip_filename = f"{app_name}_{job_id}.zip"
    from Agents.tools import get_current_root
    
    zip_path = Path(get_current_root()).parent / zip_filename
    
    if not zip_path.exists():
        raise HTTPException(404, "ZIP not found locally. It might have been uploaded to S3.")
        
    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=zip_filename,
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@app.get("/api/v1/memory")
async def get_memory():
    try:
        from Agents.memory import get_recent_episodes
        return {"episodes": get_recent_episodes(limit=20)}
    except Exception as exc:
        return {"episodes": [], "error": str(exc)}
        
@app.get("/api/v1/metrics/history")
async def get_metrics_history():
    """Returns data for standard metrics dashboard."""
    try:
        from Agents.memory import get_recent_episodes
        episodes = get_recent_episodes(limit=20)
        # Sort ascending for chart viewing
        episodes_reversed = list(reversed(episodes))
        
        return {
            "labels": [f"{e['techstack']} {e['created_at'][:10]}" for e in episodes_reversed],
            "scores": [e.get('quality_score', 0) for e in episodes_reversed],
            "issues": [e.get('issue_count', 0) for e in episodes_reversed]
        }
    except Exception as exc:
        return {"error": str(exc)}

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "jobs": len(jobs)}


# ── Serve frontend SPA ─────────────────────────────────────────────────────────

_static = Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")

