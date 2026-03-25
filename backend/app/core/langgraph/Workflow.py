import sys
import os

# Add the project root (one level up) to sys.path so 'Agents' is recognized as a package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.langgraph.State import GraphState
from langgraph.constants import END
from langgraph.graph import StateGraph
from backend.app.core.langgraph.Graphs import *
from backend.app.core.langgraph.Structured_output import *
from backend.app.core.langgraph.tools import write_file, read_file, get_current_directory, list_files, PROJECT_ROOT, set_retriever, rag_query
from backend.app.core.langgraph.RAG.rag import build_retriever



# workflow

def workflow():
    # Conditional edges

    def route_coder(state: GraphState) -> str:
        return "reviewer" if state.get("status") == "DONE" else "coder"

    graph = StateGraph(GraphState)

    graph.add_node("planner",        planner_agent)
    graph.add_node("architect",      architect_agent)
    graph.add_node("coder",          coder_agent)
    graph.add_node("reviewer",       reviewer_agent)
    graph.add_node("file_collector", file_collector)
    graph.add_node("downloader",     downloader)

    graph.set_entry_point("planner")
    graph.add_edge("planner",        "architect")
    graph.add_edge("architect",      "coder")
    graph.add_conditional_edges("coder", route_coder, {"coder": "coder", "reviewer": "reviewer"})
    graph.add_edge("reviewer",       "file_collector")
    graph.add_edge("file_collector", "downloader")
    graph.add_edge("downloader",     END)

    # We return the uncompiled graph builder
    return graph

graph_builder = workflow()
app = graph_builder.compile() # Default uncheckpointed version for some scripts

async def get_compiled_app():
    from backend.app.core.config import settings
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql+"):
        db_url = "postgresql://" + db_url.split("://", 1)[1]
    
    pool = AsyncConnectionPool(db_url, kwargs={"autocommit": True})
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    
    return graph_builder.compile(checkpointer=checkpointer)

# Entry point



if __name__ == "__main__":
    from backend.app.core.langgraph.tools import init_project_root
    init_project_root()

    # rag
    try:
        set_retriever(build_retriever())
    except Exception as e:
        print(f"[RAG] Skipping RAG init: {e}")

    result = app.invoke(
        {"user_prompt": "Build me a fastapi webapp with python "},
        {"recursion_limit": 200},
    )

    print("\n" + "=" * 50)
    print("STATUS :", result.get("status"))
    print("ZIP    :", result.get("zip_path"))
    print("FILES  :", result.get("project_structure"))
    review: ReviewResult = result.get("review_result")
    if review:
        print(f"REVIEW : {review.quality_score}/10 — passed={review.passed}")

