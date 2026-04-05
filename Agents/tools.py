import pathlib
import subprocess
import threading
from typing import Tuple

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Thread-local project root — each job gets its own isolated directory
# ---------------------------------------------------------------------------

_tls = threading.local()
_BASE_ROOT = pathlib.Path.cwd() / "generated_projects"

# Legacy alias kept for code that imported PROJECT_ROOT directly
PROJECT_ROOT = _BASE_ROOT


def set_job_root(job_id: str) -> pathlib.Path:
    """Set the project root for the current thread (one thread per job)."""
    root = _BASE_ROOT / job_id
    root.mkdir(parents=True, exist_ok=True)
    _tls.root = root
    return root


def get_current_root() -> pathlib.Path:
    """Return the per-thread project root, falling back to the base root."""
    return getattr(_tls, "root", _BASE_ROOT)


def init_project_root(job_id: str | None = None) -> str:
    if job_id:
        return str(set_job_root(job_id))
    root = get_current_root()
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def safe_path_for_project(path: str) -> pathlib.Path:
    root = get_current_root().resolve()
    p = (get_current_root() / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Attempt to write outside project root: {p}")
    return p


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def write_file(path: str, content: str) -> str:
    """Writes content to a file at the specified path within the project root."""
    p = safe_path_for_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return f"WROTE:{p}"


@tool
def read_file(path: str) -> str:
    """Reads content from a file at the specified path within the project root."""
    p = safe_path_for_project(path)
    if not p.exists():
        return ""
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


@tool
def get_current_directory() -> str:
    """Returns the current working directory of the project being generated."""
    return str(get_current_root())


@tool
def list_files(directory: str = ".") -> str:
    """Lists all files in the specified directory within the project root."""
    p = safe_path_for_project(directory)
    if not p.is_dir():
        return f"ERROR: {p} is not a directory"
    root = get_current_root()
    files = [str(f.relative_to(root)) for f in p.glob("**/*") if f.is_file()]
    return "\n".join(files) if files else "No files found."


@tool
def run_cmd(cmd: str, cwd: str = ".", timeout: int = 30) -> Tuple[int, str, str]:
    """Runs a shell command in the specified directory and returns (returncode, stdout, stderr)."""
    cwd_dir = safe_path_for_project(cwd)
    res = subprocess.run(
        cmd, shell=True, cwd=str(cwd_dir),
        capture_output=True, text=True, timeout=timeout,
    )
    return res.returncode, res.stdout, res.stderr


# ---------------------------------------------------------------------------
# RAG tool (singleton retriever set at startup)
# ---------------------------------------------------------------------------

_retriever = None


def set_retriever(retriever) -> None:
    global _retriever
    _retriever = retriever


@tool
def rag_query(query: str) -> str:
    """
    Query the best-practices knowledge base (Clean Code, Pragmatic Programmer,
    Good Research Code handbook).
    Call this BEFORE writing any new file to retrieve relevant coding guidelines.
    """
    if _retriever is None:
        return "RAG not initialised — skipping best-practices lookup."
    docs = _retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)