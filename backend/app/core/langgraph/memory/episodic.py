from backend.app.core.database import SessionLocal
from backend.app.models.run import Run
import json

def save_episodic_memory(run_id: str, session_id: str, state: dict):
    """
    Persist memory to postgres after file_collector.
    """
    with SessionLocal() as db:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = state.get("status", "DONE")
            db.commit()

def load_project_history(user_id: str) -> str:
    """
    Load recent projects history for planner context.
    """
    with SessionLocal() as db:
        runs = db.query(Run).filter(Run.user_id == user_id).order_by(Run.created_at.desc()).limit(5).all()
        if not runs:
            return "No previous project runs found."
        history = [f"Run {r.id() if hasattr(r, 'id') else r}: {r.status} at {r.created_at}" for r in runs]
        return "Recent projects:\\n" + "\\n".join(history)

