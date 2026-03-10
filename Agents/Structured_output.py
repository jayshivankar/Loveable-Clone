from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator



class ProjectAction(str, Enum):
    BUILD = "BUILD"
    MODIFY = "MODIFY"


# Planner output

class File(BaseModel):
    path: str = Field(
        description="Relative file path within the project, e.g. 'src/app.py' or 'index.html'"
    )
    purpose: str = Field(
        description="What this file does, e.g. 'main Flask app entry point'"
    )


class Plan(BaseModel):
    action: ProjectAction = Field(
        description=(
            "BUILD if the user wants to create a new project from scratch. "
            "MODIFY if the user wants to change or extend an existing project."
        )
    )
    name: str = Field(
        description="Short app name in kebab-case, e.g. 'todo-app'"
    )
    description: str = Field(
        description="One-sentence summary of what the app does"
    )
    techstack: str = Field(
        description="Comma-separated tech stack, e.g. 'HTML, CSS, JavaScript' or 'Python, Flask, SQLite'"
    )
    features: list[str] = Field(
        description="Concrete user-facing features, e.g. ['Add todo item', 'Mark as complete', 'Delete item']"
    )
    files: list[File] = Field(
        description=(
            "Every file the project needs. "
            "MUST always include 'requirements.txt' (even if empty for front-end projects) "
            "and 'README.md'."
        )
    )



# Architect output


class ImplementationTask(BaseModel):
    filepath: str = Field(
        description="Relative path of the file this task writes or modifies"
    )
    task_description: str = Field(
        description=(
            "Detailed coding instructions for this file. Must include: "
            "(1) exact functions/classes/variables to define, "
            "(2) imports needed, "
            "(3) how this file connects to other files already implemented, "
            "(4) any edge cases or validation to handle."
        )
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "List of filepaths that must be implemented BEFORE this task. "
            "Empty list means no dependencies."
        )
    )




class TaskPlan(BaseModel):
    implementation_steps: list[ImplementationTask] = Field(
        description=(
            "Ordered list of implementation tasks. "
            "Tasks with no dependencies come first. "
            "Each subsequent task may reference outputs of previous tasks."
        )
    )

    model_config = ConfigDict(extra="forbid")




# Coder runtime state (not an LLM output)


class CoderState(BaseModel):
    task_plan: TaskPlan = Field(description="The full task plan to execute")
    current_step_idx: int = Field(0, description="Index of the next step to execute")
    current_file_content: Optional[str] = Field(
        None, description="Content of the file currently being edited"
    )