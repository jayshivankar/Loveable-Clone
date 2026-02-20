import pathlib
import subprocess
from typing import Tuple
from langchain_core.tools import tool

PROJECT_ROOT = pathlib.Path.cwd() / "Generated_Project"

def safe_path_for_project(path:str) -> pathlib.Path:
    p = (PROJECT_ROOT/path).resolve()
    if PROJECT_ROOT.resolve() not in p.parents:
        return ValueError("Attempt to write outside the Project Directory")
    return p

@tool
def write_file(path:str,content:str) -> str:
    """Writes content to a file at the specified path within the Project Directory """
    p = safe_path_for_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,'w',encoding='utf-8') as f:
        f.write(content)
    return f"Wrote to {p}"