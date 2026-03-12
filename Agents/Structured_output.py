from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Planner → Plan

class File(BaseModel):
    path: str = Field(description="Relative file path, e.g. 'src/app.py' or 'index.html'")
    purpose: str = Field(description="What this file does, e.g. 'main Flask entry point'")

class Plan(BaseModel):
    name: str = Field(description="Short app name in kebab-case, e.g. 'todo-app'")
    description: str = Field(description="One-sentence summary of what the app does")
    techstack: str = Field(description="Comma-separated stack, e.g. 'HTML5, CSS3, JavaScript'")
    features: list[str] = Field(description="Concrete user-visible features")
    files: list[File] = Field(
        description="Every file the project needs, including README.md and requirements.txt"
    )


# Architect → TaskPlan

class ImplementationTask(BaseModel):
    filepath: str = Field(description="Relative path of the file this task writes or modifies")
    task_description: str = Field(
        description=(
            "Complete coding instructions: exact functions/classes to define, "
            "imports needed, integration with other files, edge cases."
        )
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Filepaths that must be implemented before this task. Empty = no deps."
    )


class TaskPlan(BaseModel):
    implementation_steps: list[ImplementationTask] = Field(
        description="Ordered list of tasks. Dependency-free tasks come first."
    )
    model_config = ConfigDict(extra="forbid")



# Coder runtime state  (internal tracking — not an LLM output)

class CoderState(BaseModel):
    task_plan: TaskPlan = Field(description="The full task plan to execute")
    current_step_idx: int = Field(0, description="Index of the next step to execute")
    current_file_content: Optional[str] = Field(None, description="Content of file being edited")



# Reviewer → ReviewResult

class IssueSeverity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class CodeIssue(BaseModel):
    filepath: str = Field(description="File that has the issue")
    severity: IssueSeverity = Field(description="low | medium | high")
    description: str = Field(description="Clear description of the problem")
    suggested_fix: str  = Field(description="Concise instruction on how to fix it")


class ReviewResult(BaseModel):
    quality_score: int  = Field(description="Overall code quality score 1-10")
    passed: bool  = Field(description="True if no high-severity issues found")
    issues: list[CodeIssue] = Field(
        default_factory=list,
        description="All issues found. Empty list means the code is clean."
    )
    summary: str = Field(description="One paragraph overall assessment")