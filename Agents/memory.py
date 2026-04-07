"""
Episodic Memory — PostgreSQL + pgvector

After every successful generation run:
  store_episode() saves each reviewer issue into the episodic_memory table.

Before planning:
  recall_past_mistakes() embeds the current prompt and does cosine similarity
  search to return relevant lessons from past builds.

Fails gracefully if PostgreSQL / pgvector is unavailable.
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_embeddings: Optional[OpenAIEmbeddings] = None
_conn: Optional[psycopg2.extensions.connection] = None
_memory_available: bool = False  # set to True after successful init


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


def _get_connection() -> psycopg2.extensions.connection:
    global _conn
    if _conn is None or _conn.closed:
        url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/codeforge",
        )
        _conn = psycopg2.connect(url)
        _conn.autocommit = True
    return _conn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_episodic_memory() -> bool:
    """
    Creates the episodic_memory table and ivfflat index if they don't exist.
    Returns True on success, False if the DB / pgvector is unavailable.
    """
    global _memory_available
    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    session_id     TEXT,
                    app_name       TEXT,
                    user_prompt    TEXT,
                    filepath       TEXT,
                    severity       TEXT,
                    description    TEXT,
                    suggested_fix  TEXT,
                    quality_score  INTEGER,
                    embedding      vector(1536),
                    created_at     TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # ivfflat index — skipped gracefully if lists param is too high
            # for small tables; recreate later in prod with VACUUM ANALYZE first
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS episodic_embedding_idx
                ON episodic_memory
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10);
                """
            )
        _memory_available = True
        print("[MEMORY] ✓ Episodic memory table ready.")
        return True
    except Exception as exc:
        print(f"[MEMORY] ⚠ Episodic memory unavailable: {exc}")
        _memory_available = False
        return False


# ---------------------------------------------------------------------------
# Write path — called after every completed run
# ---------------------------------------------------------------------------

def store_episode(
    session_id: str,
    prompt: str,
    app_name: str,
    review_result,          # ReviewResult pydantic model
) -> None:
    """Persist a completed generation run into episodic memory."""
    if not _memory_available or not review_result.issues:
        return
    try:
        conn = _get_connection()
        embeddings_api = _get_embeddings()

        # Embed the user_prompt once — the embedding represents the project
        # intent, enabling semantic search to find similar past projects.
        # The issue details (description, suggested_fix, etc.) are stored
        # in their own columns and are NOT embedded.
        prompt_embedding = embeddings_api.embed_query(prompt)
        prompt_embedding_str = str(prompt_embedding)

        with conn.cursor() as cur:
            for issue in review_result.issues:
                cur.execute(
                    """
                    INSERT INTO episodic_memory
                        (session_id, app_name, user_prompt, filepath, severity, description, suggested_fix, quality_score, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        session_id,
                        app_name,
                        prompt,
                        issue.filepath,
                        issue.severity.value,
                        issue.description,   # stored as plain text, NOT embedded
                        issue.suggested_fix,
                        review_result.quality_score,
                        prompt_embedding_str,  # embedding of the user_prompt
                    ),
                )
        print(
            f"[MEMORY] ✓ Episode stored (session={session_id[:8]}…, "
            f"score={review_result.quality_score}/10, "
            f"issues={len(review_result.issues)})"
        )
    except Exception as exc:
        print(f"[MEMORY] ✗ Could not store episode: {exc}")


# ---------------------------------------------------------------------------
# Read path — LangChain tool used by the Planner
# ---------------------------------------------------------------------------

@tool
def recall_past_mistakes(current_prompt: str) -> str:
    """
    Query episodic memory for issues encountered in similar past code-generation
    runs.  Use this BEFORE planning to avoid repeating known mistakes.

    Returns a formatted report of past high/medium-severity issues grouped by
    similarity to the current prompt.  Returns a short notice if no similar
    past runs exist yet.
    """
    if not _memory_available:
        return "[MEMORY] Episodic memory not available — skipping recall."

    try:
        conn = _get_connection()
        embedding = _get_embeddings().embed_query(current_prompt)
        vec_str = str(embedding)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    app_name,
                    filepath,
                    severity,
                    description,
                    suggested_fix,
                    quality_score,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM episodic_memory
                ORDER BY similarity DESC
                LIMIT 4
                """,
                (vec_str,),
            )
            rows = cur.fetchall()

        if not rows:
            return "[MEMORY] No similar past runs found — starting fresh."

        # Group by severity so the Planner sees the most critical issues first
        from collections import defaultdict
        by_severity: dict[str, list] = defaultdict(list)
        for row in rows:
            by_severity[row[2]].append(row)  # row[2] = severity

        lines: list[str] = [
            "╔══════════════════════════════════════════════════╗",
            "║        LESSONS FROM PAST BUILDS (MEMORY)         ║",
            "╚══════════════════════════════════════════════════╝",
            "",
        ]

        for severity_level in ("high", "medium", "low"):
            if severity_level not in by_severity:
                continue
            for app_name, filepath, severity, description, suggested_fix, quality_score, similarity in by_severity[severity_level]:
                lines.append(
                    f"[{severity.upper()}] {filepath}: {description}"
                )
                lines.append(f"  Fix: {suggested_fix}")
                lines.append(f"  (from app: {app_name}, score: {quality_score}/10)")
                lines.append("")

        lines.append("Apply these lessons during planning and architecture decisions.")
        return "\n".join(lines)

    except Exception as exc:
        return f"[MEMORY] Could not query episodic memory: {exc}"


# ---------------------------------------------------------------------------
# Utility — recent episodes (for the UI endpoint)
# ---------------------------------------------------------------------------

def get_recent_episodes(limit: int = 10) -> list[dict]:
    """Return recent episodes as plain dicts (for the /api/v1/memory endpoint)."""
    if not _memory_available:
        return []
    try:
        conn = _get_connection()
        # Aggregate the per-issue rows by session_id to match the UI format
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    session_id,
                    MAX(user_prompt) as prompt_summary,
                    MAX(app_name) as app_name,
                    MAX(quality_score) as quality_score,
                    MAX(created_at) as created_at,
                    COUNT(filepath) AS issue_count
                FROM episodic_memory
                GROUP BY session_id
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "id":            str(r[0]), # there's no PK id in the aggregated result, use session_id
                "session_id":    r[0],
                "prompt":        r[1],
                "techstack":     r[2], # using app_name instead of techstack here
                "quality_score": r[3],
                "created_at":    r[4].isoformat() if r[4] else None,
                "issue_count":   r[5],
            }
            for r in rows
        ]
    except Exception:
        return []

