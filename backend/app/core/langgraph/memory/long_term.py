import os
try:
    from mem0 import Memory
except ImportError:
    Memory = None

def get_memory_client():
    if not Memory:
        return None
    return Memory.from_config({
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": os.getenv("POSTGRES_DB", "codeforge"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", 5432)),
            }
        }
    })

def save_long_term_lesson(user_id: str, review_issues: list):
    """
    Convert review issues into sentence lessons and save to mem0 vector db.
    """
    m = get_memory_client()
    if not m:
        print("[Memory] mem0 not installed, skipping long-term write.")
        return
    
    for issue in review_issues:
        lesson = f"Lesson learned: {issue}"
        m.add(lesson, user_id=user_id)

def retrieve_past_lessons(user_id: str, prompt: str) -> str:
    """
    Perform semantic search against vector DB for past lessons.
    """
    m = get_memory_client()
    if not m:
        return ""
    
    results = m.search(query=prompt, user_id=user_id, limit=3)
    lessons = [res["text"] for res in results] if results else []
    return "\\n".join(lessons)

