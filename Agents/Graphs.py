"""
Flow: planner → architect → coder → reviewer → file_collector → downloader → END
"""

import zipfile
import pathlib
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from Agents.Prompts import (
    planner_prompt,
    architect_prompt,
    coder_system_prompt,
    reviewer_prompt,
)
from Agents.Structured_output import Plan, TaskPlan, CoderState, ReviewResult
from Agents.tools import write_file, read_file, get_current_directory, list_files, PROJECT_ROOT
from Agents.State import GraphState

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Helpers


def _read_all_project_files() -> dict[str, str]:
    """Returns {relative_path: content} for every file under PROJECT_ROOT."""
    root = pathlib.Path(PROJECT_ROOT)
    files: dict[str, str] = {}
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(root))
            try:
                files[rel] = p.read_text(encoding="utf-8")
            except Exception:
                files[rel] = "<binary — skipped>"
    return files



# Node 1 — Planner


def planner_agent(state: GraphState) -> GraphState:
    print("\n[PLANNER] Analysing prompt...")
    plan: Plan = llm.with_structured_output(Plan).invoke(
        planner_prompt(state["user_prompt"])
    )
    if plan is None:
        raise ValueError("[PLANNER] LLM returned None.")

    print(f"[PLANNER] App: {plan.name} | Stack: {plan.techstack}")
    for f in plan.files:
        print(f"  → {f.path}")
    return {"plan": plan}


# Node 2 — Architect


def architect_agent(state: GraphState) -> GraphState:
    plan: Plan = state["plan"]
    print(f"\n[ARCHITECT] Creating tasks for {len(plan.files)} files...")

    task_plan: TaskPlan = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json(indent=2))
    )
    if task_plan is None:
        raise ValueError("[ARCHITECT] LLM returned None.")

    print(f"[ARCHITECT] {len(task_plan.implementation_steps)} tasks:")

    return {"task_plan": task_plan}



# Node 3 — Coder


def coder_agent(state: GraphState) -> GraphState:
    coder_state: CoderState = state.get("coder_state") or CoderState(
        task_plan=state["task_plan"], current_step_idx=0
    )
    steps = coder_state.task_plan.implementation_steps

    if coder_state.current_step_idx >= len(steps):
        print("\n[CODER] All tasks done.")
        return {"coder_state": coder_state, "status": "DONE"}

    task = steps[coder_state.current_step_idx]
    print(f"\n[CODER] {coder_state.current_step_idx + 1}/{len(steps)}: {task.filepath}")

    existing = read_file.run(task.filepath) or ""
    user_msg = (
        f"File: {task.filepath}\n\n"
        f"Task:\n{task.task_description}\n\n"
        + (f"Dependencies: {task.depends_on}\n\n" if task.depends_on else "")
        + (f"Existing content:\n```\n{existing}\n```\n\n" if existing else "New file — create from scratch.\n\n")
        + "Write the COMPLETE file using write_file(path, content)."
    )

    agent = create_agent(
        model=llm,
        tools=[read_file, write_file, list_files, get_current_directory],
        system_prompt=coder_system_prompt(),
    )
    agent.invoke({"messages": [HumanMessage(content=user_msg)]})

    coder_state.current_step_idx += 1
    print(f"[CODER] : {task.filepath} done")
    return {"coder_state": coder_state, "status": "IN_PROGRESS"}



# Node 4 — Reviewer
# Reviews all generated files, flags issues, applies fixes via tools


def reviewer_agent(state: GraphState) -> GraphState:
    print("\n[REVIEWER] Reviewing all generated files...")

    all_files = _read_all_project_files()
    if not all_files:
        print("[REVIEWER] No files found to review.")
        return {"review_result": None}

    # Format all files for the LLM
    files_block = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in sorted(all_files.items())
    )

    # Step 1: structured review — get a report of issues
    review_result: ReviewResult = llm.with_structured_output(ReviewResult).invoke(
        reviewer_prompt(files_block=files_block)
    )

    if review_result is None:
        raise ValueError("[REVIEWER] LLM returned None.")

    print(f"[REVIEWER] Score   : {review_result.quality_score}/10")
    print(f"[REVIEWER] Passed  : {review_result.passed}")
    print(f"[REVIEWER] Issues  : {len(review_result.issues)}")
    for issue in review_result.issues:
        print(f"  [{issue.severity.upper()}] {issue.filepath}: {issue.description}")

    # Step 2: if issues exist, let the reviewer fix them using file tools
    if not review_result.passed and review_result.issues:
        print("\n[REVIEWER] Applying fixes...")

        fix_instructions = "\n".join(
            f"- {issue.filepath} ({issue.severity}): {issue.description}. Fix: {issue.suggested_fix}"
            for issue in review_result.issues
        )

        fix_msg = (
            f"You are reviewing and fixing a generated project.\n\n"
            f"Issues found:\n{fix_instructions}\n\n"
            f"For each issue:\n"
            f"1. Read the file with read_file()\n"
            f"2. Fix the issue\n"
            f"3. Write the corrected COMPLETE file with write_file()\n\n"
            f"Do not change files that have no issues."
        )

        fix_agent = create_agent(
            model=llm,
            tools=[read_file, write_file, list_files, get_current_directory],
            system_prompt="You are a senior code reviewer. Fix only the reported issues. Write complete files.",
        )
        fix_agent.invoke({"messages": [HumanMessage(content=fix_msg)]})
        print("[REVIEWER] : Fixes applied.")

    return {"review_result": review_result}



# Node 5 — File Collector
# Reads all files from disk


def file_collector(state: GraphState) -> GraphState:
    print("\n[FILE COLLECTOR] Collecting final project files...")

    generated_files = _read_all_project_files()
    project_structure = sorted(generated_files.keys())

    print(f"[FILE COLLECTOR] {len(generated_files)} files collected:")
    for path in project_structure:
        print(f"  {path}  ({len(generated_files[path])} chars)")


    return {
        "generated_files": generated_files,
        "project_structure": project_structure,
        "status": "READY",
    }


# Node 6 — Downloader
# Zips the project folder → zip_path in state

def downloader(state: GraphState) -> GraphState:
    print("\n[DOWNLOADER] Creating zip...")

    root = pathlib.Path(PROJECT_ROOT)
    plan: Plan = state.get("plan")
    app_name = plan.name if plan else "project"
    zip_path = root.parent / f"{app_name}.zip"

    if not root.exists():
        raise FileNotFoundError(f"[DOWNLOADER] Project root missing: {root}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(root.parent))

    size_kb = zip_path.stat().st_size / 1024
    print(f"[DOWNLOADER] : {zip_path.name}  ({size_kb:.1f} KB)")
    return {"zip_path": str(zip_path), "status": "DONE"}


