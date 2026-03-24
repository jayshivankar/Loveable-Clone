from typing import Optional
from typing_extensions import TypedDict

from backend.app.core.langgraph.Structured_output import Plan, TaskPlan, CoderState, ReviewResult


class GraphState(TypedDict, total=False):
    # Input
    user_prompt: str
    chat_session_id: str

    # Planner
    plan: Plan

    # Architect
    task_plan: TaskPlan

    # Coder
    coder_state: CoderState

    # File Collector
    generated_files: dict        # {relative_path: file_content}
    project_structure: list      # sorted list of relative paths

    # Downloader
    zip_path: str

    # Reviewer
    review_result: ReviewResult    # score, issues, passed flag

    status: str
    
    # HITL and Memory
    approval_status: str # PENDING | ACCEPTED | REJECTED | EDITED
    awaiting_user_input: bool
    edited_task_plan: Optional[TaskPlan]
    session_id: str
    user_id: str
    thread_id: str
    past_review_lessons: str
    user_project_history: str
    retry_count: int
    error: Optional[str]