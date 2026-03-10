from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from Agents.Prompts import planner_prompt, architect_prompt, coder_system_prompt
from Agents.Structured_output import Plan, TaskPlan, CoderState, ProjectAction
from Agents.tools import write_file, read_file, get_current_directory, list_files
from langchain.agents import create_agent


load_dotenv()

# initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("graph-log")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GraphState(dict):
    """
    Keys used across the graph:
      user_prompt     : str
      plan            : Plan
      task_plan       : TaskPlan
      coder_state     : CoderState
      status          : str  ("IN_PROGRESS" | "DONE")
      error           : str | None
    """



# Node: Planner

def planner_agent(state: GraphState) -> GraphState:
    """
    Converts the user prompt into a structured Plan.
    Determines whether the user wants to BUILD or MODIFY a project.
    """
    user_prompt: str = state["user_prompt"]

    print("\n[PLANNER] Analysing prompt...")

    try:
        plan: Plan = llm.with_structured_output(Plan).invoke(
            planner_prompt(user_prompt)
        )
    except Exception as e:
        raise RuntimeError(f"[PLANNER] LLM call failed: {e}") from e



    #  Logging
    logger.info(f"[PLANNER] Action  : {plan.action}")
    logger.info(f"[PLANNER] App     : {plan.name}")
    logger.info(f"[PLANNER] Stack   : {plan.techstack}")
    logger.info(f"[PLANNER] Files   : {len(plan.files)}")
    for f in plan.files:
        logger.info(f"            → {f.path}  ({f.purpose})")

    return {"plan": plan}


# Node: Architect


def architect_agent(state: GraphState) -> GraphState:
    """
    Reads the Plan and produces an ordered TaskPlan.
    Each ImplementationTask maps 1-to-1 with a file.
    """
    plan: Plan = state["plan"]

    print(f"\n[ARCHITECT] Breaking plan into tasks for {len(plan.files)} files...")

    try:
        task_plan: TaskPlan = llm.with_structured_output(TaskPlan).invoke(
            architect_prompt(plan=plan.model_dump_json(indent=2))
        )
    except Exception as e:
        raise RuntimeError(f"[ARCHITECT] LLM call failed: {e}") from e


    #  Logging
    logger.info(f"[ARCHITECT] {len(task_plan.implementation_steps)} tasks generated:")


    return {"task_plan": task_plan}



# Node: Coder

def coder_agent(state: GraphState) -> GraphState:
    """
    Executes one implementation step at a time.
    Loops until all steps are complete.
    """
    #  Initialise coder state on first call
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(
            task_plan=state["task_plan"],
            current_step_idx=0,
        )

    steps = coder_state.task_plan.implementation_steps

    # Check if all tasks are done
    if coder_state.current_step_idx >= len(steps):
        logger.info("\n[CODER] All tasks complete. Status → DONE")
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]

    print(
        f"\n[CODER] Task {coder_state.current_step_idx + 1}/{len(steps)}: "
        f"{current_task.filepath}"
    )

    # Read existing file content
    existing_content = read_file.run(current_task.filepath) or ""

    # user message for task
    user_message = (
        f"File to write: {current_task.filepath}\n\n"
        f"Instructions:\n{current_task.task_description}\n\n"
        + (
            f"Dependencies already implemented:\n"
            + "\n".join(f"  - {d}" for d in current_task.depends_on)
            + "\n\n"
            if current_task.depends_on else ""
        )
        + (
            f"Existing file content (update if needed):\n"
            f"```\n{existing_content}\n```\n\n"
            if existing_content else
            "This file does not exist yet — create it from scratch.\n\n"
        )
        + "Write the COMPLETE file content using write_file(path, content)."
    )

    # Run the coder agent
    coder_tools = [read_file, write_file, list_files, get_current_directory]

    agent = create_agent(
        model=llm,
        tools=coder_tools,
        prompt=coder_system_prompt(),
    )

    agent.invoke({
        "messages": [HumanMessage(content=user_message)]
    })

    # Advance step index
    coder_state.current_step_idx += 1
    logger.info(f"[CODER] ✓ Finished {current_task.filepath}")

    return {"coder_state": coder_state, "status": "IN_PROGRESS"}



# Conditional edge: loop or finish


def should_continue_coding(state: GraphState) -> str:
    if state.get("status") == "DONE":
        return "END"
    return "coder"



# Graph assembly


graph = StateGraph(GraphState)

graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge("planner", "architect")
graph.add_edge("architect", "coder")

graph.add_conditional_edges(
    "coder",
    should_continue_coding,
    {"END": END, "coder": "coder"},
)

graph.set_entry_point("planner")

app = graph.compile()



# Entry point


if __name__ == "__main__":
    from Agents.tools import init_project_root

    init_project_root()

    result = app.invoke(
        {"user_prompt": "Build a colourful modern todo app in HTML CSS and JS"},
        {"recursion_limit": 150},
    )

