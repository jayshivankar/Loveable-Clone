"""
Agent node implementations.

Flow: planner → architect → coder (parallel) → reviewer → fixer* → file_collector → downloader → END
      *fixer runs only when review fails and fixer_enabled=True
"""

from __future__ import annotations

import pathlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from Agents.Prompts import (
    architect_prompt,
    coder_system_prompt,
    fixer_system_prompt,
    planner_prompt,
    reviewer_prompt,
)
from Agents.Structured_output import (
    CoderState,
    Plan,
    ReviewResult,
    TaskPlan,
    ImplementationTask,
)
from Agents.State import GraphState
from Agents.tools import (
    get_current_directory,
    get_current_root,
    list_files,
    rag_query,
    read_file,
    write_file,
)
from Agents.memory import recall_past_mistakes

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_all_project_files() -> dict[str, str]:
    """Returns {relative_path: content} for every file under the current project root."""
    root = pathlib.Path(get_current_root())
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


def _topological_levels(tasks: list[ImplementationTask]) -> list[list[ImplementationTask]]:
    """
    Group tasks into dependency levels so each level can be coded in parallel.
    Level 0 = no deps, Level 1 = depends on Level 0, etc.
    """
    levels: list[list[ImplementationTask]] = []
    done: set[str] = set()
    remaining = list(tasks)

    while remaining:
        ready = [t for t in remaining if all(d in done for d in t.depends_on)]
        if not ready:
            # Circular dependency fallback — add all remaining as one level
            ready = remaining
        levels.append(ready)
        done.update(t.filepath for t in ready)
        remaining = [t for t in remaining if t not in ready]

    return levels


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Planner  (queries episodic memory first)
# ─────────────────────────────────────────────────────────────────────────────


def planner_agent(state: GraphState) -> GraphState:
    print("\n[PLANNER] Querying episodic memory for past lessons…")
    memory_context = ""
    try:
        memory_context = recall_past_mistakes.invoke(
            {"current_prompt": state["user_prompt"]}
        )
        if "No similar" in memory_context or "not available" in memory_context:
            print("[PLANNER] No useful past memories found — starting fresh.")
        else:
            print("[PLANNER] 🧠 Past lessons loaded — incorporating into plan.")
    except Exception as exc:
        print(f"[PLANNER] Memory recall skipped: {exc}")

    print("[PLANNER] Analysing prompt…")
    plan: Plan = llm.with_structured_output(Plan).invoke(
        planner_prompt(state["user_prompt"], memory_context=memory_context)
    )
    if plan is None:
        raise ValueError("[PLANNER] LLM returned None.")

    print(f"[PLANNER] App: {plan.name} | Stack: {plan.techstack}")
    for f in plan.files:
        print(f"  → {f.path}")

    return {"plan": plan, "memory_context": memory_context}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Architect
# ─────────────────────────────────────────────────────────────────────────────


def architect_agent(state: GraphState) -> GraphState:
    plan: Plan = state["plan"]
    print(f"\n[ARCHITECT] Creating tasks for {len(plan.files)} files…")

    task_plan: TaskPlan = llm.with_structured_output(TaskPlan).invoke(
        architect_prompt(plan=plan.model_dump_json(indent=2))
    )
    if task_plan is None:
        raise ValueError("[ARCHITECT] LLM returned None.")

    print(f"[ARCHITECT] {len(task_plan.implementation_steps)} implementation tasks:")
    for task in task_plan.implementation_steps:
        deps = f" (deps: {task.depends_on})" if task.depends_on else ""
        print(f"  → {task.filepath}{deps}")

    return {"task_plan": task_plan}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Coder  (parallel execution by dependency level)
# ─────────────────────────────────────────────────────────────────────────────


def _code_single_task(task: ImplementationTask) -> None:
    """Code a single file inside a worker thread."""
    existing = read_file.invoke({"path": task.filepath}) or ""
    user_msg = (
        f"File: {task.filepath}\n\n"
        f"Task:\n{task.task_description}\n\n"
        + (f"Dependencies: {task.depends_on}\n\n" if task.depends_on else "")
        + (
            f"Existing content:\n```\n{existing}\n```\n\n"
            if existing
            else "New file — create from scratch.\n\n"
        )
        + "Write the COMPLETE file using write_file(path, content)."
    )

    agent = create_react_agent(
        model=llm,
        tools=[read_file, write_file, list_files, get_current_directory, rag_query],
        state_modifier=SystemMessage(content=coder_system_prompt()),
    )
    agent.invoke({"messages": [HumanMessage(content=user_msg)]})
    print(f"[CODER] ✓ {task.filepath}")


def coder_agent(state: GraphState) -> GraphState:
    task_plan: TaskPlan = state["task_plan"]
    levels = _topological_levels(task_plan.implementation_steps)

    total = len(task_plan.implementation_steps)
    print(f"\n[CODER] {total} files across {len(levels)} dependency level(s)")

    for lvl_idx, level in enumerate(levels):
        files = [t.filepath for t in level]
        print(f"[CODER] Level {lvl_idx + 1}/{len(levels)}: {files}")

        if len(level) == 1:
            _code_single_task(level[0])
        else:
            # Run dependency-free files concurrently (max 3 parallel LLM calls)
            with ThreadPoolExecutor(max_workers=min(len(level), 3)) as executor:
                futures = {executor.submit(_code_single_task, t): t for t in level}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[CODER] ✗ {task.filepath}: {exc}")

    print("[CODER] All files complete.")
    return {"status": "DONE"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Reviewer  (structured analysis only)
# ─────────────────────────────────────────────────────────────────────────────


def reviewer_agent(state: GraphState) -> GraphState:
    print("\n[REVIEWER] Reviewing all generated files…")

    all_files = _read_all_project_files()
    if not all_files:
        print("[REVIEWER] No files found to review.")
        return {"review_result": None}

    files_block = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in sorted(all_files.items())
    )

    review_result: ReviewResult = llm.with_structured_output(ReviewResult).invoke(
        reviewer_prompt(files_block=files_block)
    )
    if review_result is None:
        raise ValueError("[REVIEWER] LLM returned None.")

    print(f"[REVIEWER] Score  : {review_result.quality_score}/10")
    print(f"[REVIEWER] Passed : {review_result.passed}")
    print(f"[REVIEWER] Issues : {len(review_result.issues)}")
    for issue in review_result.issues:
        print(f"  [{issue.severity.value.upper()}] {issue.filepath}: {issue.description}")

    return {"review_result": review_result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Fixer  (applies fixes via file tools — separate from reviewer)
# ─────────────────────────────────────────────────────────────────────────────


def fixer_agent(state: GraphState) -> GraphState:
    review_result: ReviewResult = state.get("review_result")
    if not review_result or not review_result.issues:
        print("\n[FIXER] Nothing to fix.")
        return {}

    print(f"\n[FIXER] Applying fixes for {len(review_result.issues)} issue(s)…")

    fix_instructions = "\n".join(
        f"- {issue.filepath} ({issue.severity.value}): {issue.description}. "
        f"Fix: {issue.suggested_fix}"
        for issue in review_result.issues
    )

    fix_msg = (
        f"You are fixing issues in a generated project.\n\n"
        f"Issues to fix:\n{fix_instructions}\n\n"
        f"For each issue:\n"
        f"1. Read the file with read_file()\n"
        f"2. Apply the fix\n"
        f"3. Write the corrected COMPLETE file with write_file()\n\n"
        f"Do not change files that have no listed issues."
    )

    fixer = create_react_agent(
        model=llm,
        tools=[read_file, write_file, list_files, get_current_directory],
        state_modifier=SystemMessage(content=fixer_system_prompt()),
    )
    fixer.invoke({"messages": [HumanMessage(content=fix_msg)]})
    print("[FIXER] ✓ All fixes applied.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — File Collector
# ─────────────────────────────────────────────────────────────────────────────


def file_collector(state: GraphState) -> GraphState:
    print("\n[FILE COLLECTOR] Collecting final project files…")

    generated_files = _read_all_project_files()
    project_structure = sorted(generated_files.keys())

    print(f"[FILE COLLECTOR] {len(generated_files)} file(s) collected:")
    for path in project_structure:
        print(f"  {path}  ({len(generated_files[path])} chars)")

    return {
        "generated_files": generated_files,
        "project_structure": project_structure,
        "status": "READY",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Downloader
# ─────────────────────────────────────────────────────────────────────────────


def downloader(state: GraphState) -> GraphState:
    print("\n[DOWNLOADER] Creating ZIP archive…")

    root = pathlib.Path(get_current_root())
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
    print(f"[DOWNLOADER] ✓ {zip_path.name}  ({size_kb:.1f} KB)")
    return {"zip_path": str(zip_path), "status": "DONE"}
