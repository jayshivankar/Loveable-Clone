from typing import Optional
from typing_extensions import TypedDict

from Agents.Structured_output import Plan, TaskPlan, CoderState, ReviewResult


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