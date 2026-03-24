from backend.app.core.langgraph.State import GraphState

def hitl_gate(state: GraphState) -> GraphState:
    """
    Pause/resume checkpoint after architect node.
    If state is ACCEPTED or EDITED, it flows to coder.
    Otherwise, stops awaiting human approval.
    """
    status = state.get("approval_status", "PENDING")
    if status in ["ACCEPTED", "EDITED"]:
        state["awaiting_user_input"] = False
    else:
        state["approval_status"] = "PENDING"
        state["awaiting_user_input"] = True
    
    return state
