"""
Episodic Memory — PostgreSQL + pgvector

After every successful generation run:
  store_episode() saves the prompt, tech stack, reviewer issues, and quality score.

Before planning:
  recall_past_mistakes() embeds the current prompt and does cosine similarity
  search to return relevant lessons from past builds.

Fails gracefully if PostgreSQL / pgvector is unavailable.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import psycopg2
from psycopg2.extras import Json
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
                    id              SERIAL PRIMARY KEY,
                    session_id      TEXT,
                    prompt_summary  TEXT,
                    prompt_embedding vector(1536),
                    techstack       TEXT,
                    issues_json     JSONB,
                    quality_score   INTEGER,
                    created_at      TIMESTAMP DEFAULT NOW()
                );
                """
            )
            # ivfflat index — skipped gracefully if lists param is too high
            # for small tables; recreate later in prod with VACUUM ANALYZE first
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS episodic_embedding_idx
                ON episodic_memory
                USING ivfflat (prompt_embedding vector_cosine_ops)
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
    techstack: str,
    review_result,          # ReviewResult pydantic model
) -> None:
    """Persist a completed generation run into episodic memory."""
    if not _memory_available:
        return
    try:
        conn = _get_connection()
        embedding = _get_embeddings().embed_query(prompt)

        issues = [
            {
                "filepath":    issue.filepath,
                "severity":    issue.severity.value,
                "description": issue.description,
                "fix":         issue.suggested_fix,
            }
            for issue in review_result.issues
        ]

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO episodic_memory
                    (session_id, prompt_summary, prompt_embedding,
                     techstack, issues_json, quality_score)
                VALUES (%s, %s, %s::vector, %s, %s, %s)
                """,
                (
                    session_id,
                    prompt[:500],
                    str(embedding),
                    techstack,
                    Json(issues),
                    review_result.quality_score,
                ),
            )
        print(
            f"[MEMORY] ✓ Episode stored (session={session_id[:8]}…, "
            f"score={review_result.quality_score}/10, "
            f"issues={len(issues)})"
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
                    prompt_summary,
                    techstack,
                    issues_json,
                    quality_score,
                    1 - (prompt_embedding <=> %s::vector) AS similarity
                FROM episodic_memory
                WHERE 1 - (prompt_embedding <=> %s::vector) > 0.55
                ORDER BY similarity DESC
                LIMIT 5
                """,
                (vec_str, vec_str),
            )
            rows = cur.fetchall()

        if not rows:
            return "[MEMORY] No similar past runs found — starting fresh."

        lines: list[str] = [
            "╔══════════════════════════════════════════════════╗",
            "║        LESSONS FROM PAST BUILDS (MEMORY)        ║",
            "╚══════════════════════════════════════════════════╝",
            "",
        ]

        for prompt_summary, techstack, issues_json, quality_score, similarity in rows:
            lines.append(
                f"▶ Past build  similarity={similarity:.2f}  "
                f"score={quality_score}/10  stack={techstack}"
            )
            lines.append(f"  Prompt: {prompt_summary[:120]}…")

            high   = [i for i in issues_json if i["severity"] == "high"]
            medium = [i for i in issues_json if i["severity"] == "medium"]

            if high:
                lines.append("  ✗ HIGH-severity issues to avoid:")
                for iss in high:
                    lines.append(f"      • {iss['filepath']}: {iss['description']}")
                    lines.append(f"          Fix → {iss['fix']}")

            if medium:
                lines.append("  ⚠ MEDIUM-severity issues to watch:")
                for iss in medium[:3]:
                    lines.append(f"      • {iss['filepath']}: {iss['description']}")

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
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, prompt_summary, techstack,
                       quality_score, created_at,
                       jsonb_array_length(issues_json) AS issue_count
                FROM episodic_memory
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "id":            r[0],
                "session_id":    r[1],
                "prompt":        r[2],
                "techstack":     r[3],
                "quality_score": r[4],
                "created_at":    r[5].isoformat() if r[5] else None,
                "issue_count":   r[6],
            }
            for r in rows
        ]
    except Exception:
        return []
