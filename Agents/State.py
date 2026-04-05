from typing import Optional
from typing_extensions import TypedDict

from Agents.Structured_output import Plan, TaskPlan, ReviewResult


class GraphState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_prompt:     str
    chat_session_id: str
    fixer_enabled:   bool    # whether to run the fixer node after review

    # ── Episodic Memory ──────────────────────────────────────────────────────
    memory_context:  str     # raw text returned by recall_past_mistakes()

    # ── Planner ──────────────────────────────────────────────────────────────
    plan: Plan

    # ── Architect ────────────────────────────────────────────────────────────
    task_plan: TaskPlan

    # ── File Collector ───────────────────────────────────────────────────────
    generated_files:   dict   # {relative_path: file_content}
    project_structure: list   # sorted list of relative paths

    # ── Downloader ───────────────────────────────────────────────────────────
    zip_path: str

    # ── Reviewer ─────────────────────────────────────────────────────────────
    review_result: ReviewResult   # score, issues, passed flag

    # ── Status ───────────────────────────────────────────────────────────────
    status: str