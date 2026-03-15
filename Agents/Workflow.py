from Agents.State import GraphState
from langgraph.constants import END
from langgraph.graph import StateGraph
from Agents.Graphs import *
from Agents.Structured_output import *



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

    flow = graph.compile()

    return flow

app = workflow()


# Entry point


if __name__ == "__main__":
    from Agents.tools import init_project_root
    init_project_root()

    result = app.invoke(
        {"user_prompt": "Build a colourful modern todo app in HTML CSS and JS"},
        {"recursion_limit": 200},
    )

    print("\n" + "=" * 50)
    print("STATUS :", result.get("status"))
    print("ZIP    :", result.get("zip_path"))
    print("FILES  :", result.get("project_structure"))
    review: ReviewResult = result.get("review_result")
    if review:
        print(f"REVIEW : {review.quality_score}/10 — passed={review.passed}")

### add a RAG tool;