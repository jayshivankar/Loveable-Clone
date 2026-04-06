"""
Agent node implementations.

Flow: planner → architect → coder (parallel) → reviewer → fixer* → file_collector → downloader → END
      *fixer runs only when review fails and fixer_enabled=True
"""

from __future__ import annotations

import os
import pathlib
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import boto3
from botocore.config import Config

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
from Agents.logger import get_logger
from Agents.metrics import AGENT_LATENCY

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
log = get_logger("codeforge.graphs")

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


# Tenacity retry decorator for LLM calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def _invoke_llm_structured(model, structure, prompt):
    return model.with_structured_output(structure).invoke(prompt)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def _invoke_agent(agent, messages):
    return agent.invoke(messages)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Planner  (queries episodic memory first)
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="planner").time()
def planner_agent(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="planner", step_index=1)
    
    node_log.info("querying_memory", msg="Querying episodic memory for past lessons...")
    memory_context = ""
    try:
        memory_context = recall_past_mistakes.invoke(
            {"current_prompt": state["user_prompt"]}
        )
        if "No similar" in memory_context or "not available" in memory_context:
            node_log.info("memory_empty", msg="No useful past memories found — starting fresh.")
        else:
            node_log.info("memory_found", msg="Past lessons loaded — incorporating into plan.")
    except Exception as exc:
        node_log.error("memory_error", error=str(exc))

    node_log.info("analysing_prompt", msg="Analysing prompt...")
    
    plan: Plan = _invoke_llm_structured(
        llm, Plan, planner_prompt(state["user_prompt"], memory_context=memory_context)
    )
    
    if plan is None:
        raise ValueError("[PLANNER] LLM returned None.")

    node_log.info("plan_created", app_name=plan.name, stack=plan.techstack)

    return {"plan": plan, "memory_context": memory_context}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Architect
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="architect").time()
def architect_agent(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="architect", step_index=2)
    plan: Plan = state["plan"]
    
    node_log.info("creating_tasks", files_count=len(plan.files))

    task_plan: TaskPlan = _invoke_llm_structured(
        llm, TaskPlan, architect_prompt(plan=plan.model_dump_json(indent=2))
    )
    
    if task_plan is None:
        raise ValueError("[ARCHITECT] LLM returned None.")

    node_log.info("tasks_created", tasks_count=len(task_plan.implementation_steps))

    return {"task_plan": task_plan}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Coder  (parallel execution by dependency level)
# ─────────────────────────────────────────────────────────────────────────────

def _code_single_task(task: ImplementationTask, session_id: str) -> None:
    """Code a single file inside a worker thread."""
    worker_log = log.bind(session_id=session_id, node_name="coder_worker", filepath=task.filepath)
    worker_log.info("coding_start", msg=f"Starting task: {task.filepath}")
    
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
    _invoke_agent(agent, {"messages": [HumanMessage(content=user_msg)]})
    worker_log.info("coding_done", msg=f"Completed task: {task.filepath}")


@AGENT_LATENCY.labels(node_name="coder").time()
def coder_agent(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="coder", step_index=3)
    
    task_plan: TaskPlan = state["task_plan"]
    levels = _topological_levels(task_plan.implementation_steps)

    total = len(task_plan.implementation_steps)
    node_log.info("coder_start", total_files=total, levels=len(levels))

    for lvl_idx, level in enumerate(levels):
        files = [t.filepath for t in level]
        node_log.info("level_start", level=lvl_idx + 1, files=files)

        if len(level) == 1:
            _code_single_task(level[0], session_id)
        else:
            # Run dependency-free files concurrently (max 3 parallel LLM calls)
            with ThreadPoolExecutor(max_workers=min(len(level), 3)) as executor:
                futures = {executor.submit(_code_single_task, t, session_id): t for t in level}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        node_log.error("worker_error", filepath=task.filepath, error=str(exc))

    node_log.info("coder_done", msg="All files complete.")
    return {"status": "DONE"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Reviewer  (structured analysis only)
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="reviewer").time()
def reviewer_agent(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="reviewer", step_index=4)
    
    node_log.info("reviewer_start", msg="Reviewing all generated files...")

    all_files = _read_all_project_files()
    if not all_files:
        node_log.warning("no_files", msg="No files found to review.")
        return {"review_result": None}

    files_block = "\n\n".join(
        f"### {path}\n```\n{content}\n```"
        for path, content in sorted(all_files.items())
    )

    review_result: ReviewResult = _invoke_llm_structured(
        llm, ReviewResult, reviewer_prompt(files_block=files_block)
    )
    
    if review_result is None:
        raise ValueError("[REVIEWER] LLM returned None.")

    node_log.info("review_done", score=review_result.quality_score, passed=review_result.passed, issues_count=len(review_result.issues))

    return {"review_result": review_result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — Fixer  (applies fixes via file tools — separate from reviewer)
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="fixer").time()
def fixer_agent(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="fixer", step_index=5)
    
    review_result: ReviewResult = state.get("review_result")
    if not review_result or not review_result.issues:
        node_log.info("fixer_skip", msg="Nothing to fix.")
        return {}

    node_log.info("fixer_start", issues_count=len(review_result.issues))

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
    _invoke_agent(fixer, {"messages": [HumanMessage(content=fix_msg)]})
    
    node_log.info("fixer_done", msg="All fixes applied.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — File Collector
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="file_collector").time()
def file_collector(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="file_collector", step_index=6)
    
    node_log.info("collector_start", msg="Collecting final project files...")

    generated_files = _read_all_project_files()
    project_structure = sorted(generated_files.keys())

    node_log.info("collector_done", files_count=len(generated_files))

    return {
        "generated_files": generated_files,
        "project_structure": project_structure,
        "status": "READY",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — Downloader (now uses S3/R2)
# ─────────────────────────────────────────────────────────────────────────────

@AGENT_LATENCY.labels(node_name="downloader").time()
def downloader(state: GraphState) -> GraphState:
    session_id = state.get("chat_session_id", "unknown")
    node_log = log.bind(session_id=session_id, node_name="downloader", step_index=7)
    
    node_log.info("downloader_start", msg="Creating ZIP archive for upload...")

    root = pathlib.Path(get_current_root())
    plan: Plan = state.get("plan")
    app_name = plan.name if plan else "project"
    zip_filename = f"{app_name}_{session_id}.zip"
    local_zip_path = root.parent / zip_filename

    if not root.exists():
        raise FileNotFoundError(f"[DOWNLOADER] Project root missing: {root}")

    # Create local zip temporarily
    with zipfile.ZipFile(local_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(root.parent))

    size_kb = local_zip_path.stat().st_size / 1024
    node_log.info("zip_created", size_kb=size_kb)

    # Context setup for boto3
    bucket_name = os.getenv("S3_BUCKET_NAME")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    # Upload to S3 if configured, otherwise fallback to a mock URL
    if all([bucket_name, aws_access_key_id, aws_secret_access_key]):
        node_log.info("s3_upload_start", bucket=bucket_name, file=zip_filename)
        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url if endpoint_url else None,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region,
                config=Config(signature_version='s3v4')
            )
            
            # Upload file
            s3_client.upload_file(str(local_zip_path), bucket_name, zip_filename)
            
            # Generate presigned URL
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': zip_filename},
                ExpiresIn=86400 # 24 hours
            )
            
            node_log.info("s3_upload_done", url=presigned_url)
            zip_url = presigned_url

            # Cleanup local zip
            local_zip_path.unlink(missing_ok=True)
            
        except Exception as e:
            node_log.error("s3_upload_error", error=str(e))
            # Fallback for dev without strict failing
            zip_url = f"/api/v1/download/local/{session_id}"
    else:
        node_log.warning("s3_not_configured", msg="S3 secrets missing. Emitting local stub URL.")
        zip_url = f"/api/v1/download/local/{session_id}"

    return {"zip_url": zip_url, "status": "DONE"}
