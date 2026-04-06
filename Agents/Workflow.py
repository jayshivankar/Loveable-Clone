"""
Workflow — LangGraph StateGraph definition.

get_app() is an async factory.
The graph is: planner → architect → coder → reviewer → [fixer?] → file_collector → downloader → END
"""

from __future__ import annotations
import os

from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from Agents.State import GraphState
from Agents.Graphs import (
    planner_agent,
    architect_agent,
    coder_agent,
    reviewer_agent,
    fixer_agent,
    file_collector,
    downloader,
)

# ── Cached app singleton ──────────────────────────────────────────────────────
_app = None
_pool = None
_saver = None

async def get_app(fixer_enabled: bool = True):
    """
    Build and return the compiled LangGraph application.
    Uses AsyncPostgresSaver for persistent checkponting.
    The result is cached; subsequent calls with different fixer_enabled values
    return the same graph (fixer routing uses runtime state).
    """
    global _app, _pool, _saver
    if _app is not None:
        return _app

    # Setup connection pool for checkpointing
    if _pool is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("[WORKFLOW] DATABASE_URL is not set in the environment.")
        # Replace async driver prefix if needed by psycopg
        url = url.replace("postgresql+psycopg2", "postgresql")
        _pool = AsyncConnectionPool(conninfo=url, max_size=20, kwargs={"autocommit": True})
        
    if _saver is None:
        _saver = AsyncPostgresSaver(_pool)
        await _saver.setup()

    # ── Conditional routing ───────────────────────────────────────────────────

    def route_after_reviewer(state: GraphState) -> str:
        review = state.get("review_result")
        enabled = state.get("fixer_enabled", True)
        if enabled and review and not review.passed and review.issues:
            return "fixer"
        return "file_collector"

    # ── Build graph ───────────────────────────────────────────────────────────

    graph = StateGraph(GraphState)

    graph.add_node("planner",        planner_agent)
    graph.add_node("architect",      architect_agent)
    graph.add_node("coder",          coder_agent)
    graph.add_node("reviewer",       reviewer_agent)
    graph.add_node("fixer",          fixer_agent)
    graph.add_node("file_collector", file_collector)
    graph.add_node("downloader",     downloader)

    graph.set_entry_point("planner")
    graph.add_edge("planner",   "architect")
    graph.add_edge("architect", "coder")
    graph.add_edge("coder",     "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"fixer": "fixer", "file_collector": "file_collector"},
    )
    graph.add_edge("fixer",          "file_collector")
    graph.add_edge("file_collector", "downloader")
    graph.add_edge("downloader",     END)

    _app = graph.compile(checkpointer=_saver)
    return _app


# ── CLI entry point (kept for direct testing) ─────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from Agents.tools import init_project_root, set_retriever
    from Agents.RAG.rag import build_retriever
    from Agents.memory import init_episodic_memory
    from Agents.Structured_output import ReviewResult

    async def main():
        TEST_JOB = "cli-test"
        init_project_root(TEST_JOB)
        init_episodic_memory()

        try:
            set_retriever(build_retriever())
        except Exception as e:
            print(f"[RAG] Skipping: {e}")

        flow = await get_app()
        
        config = {"configurable": {"thread_id": TEST_JOB}, "recursion_limit": 200}
        
        result = await flow.ainvoke(
            {
                "user_prompt":    "Build me a FastAPI webapp with Python",
                "chat_session_id": TEST_JOB,
                "fixer_enabled":  True,
            },
            config,
        )

        print("\n" + "=" * 50)
        print("STATUS :", result.get("status"))
        print("ZIP    :", result.get("zip_url"))
        print("FILES  :", result.get("project_structure"))
        review: ReviewResult = result.get("review_result")
        if review:
            print(f"REVIEW : {review.quality_score}/10 — passed={review.passed}")

    asyncio.run(main())
