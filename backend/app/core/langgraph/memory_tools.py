import json
from langchain_core.tools import tool
from openai import OpenAI
from backend.app.core.config import settings

# Must be synchronous since existing code is sync
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

def make_recall_tool(database_service, user_id: str):
    @tool
    def recall_past_projects(query: str) -> str:
        """
        Search your past code generation history for episodes relevant
        to what you are about to plan. Call this BEFORE generating the
        project plan so you can avoid repeating past mistakes and
        understand this user's history and preferences.

        Args:
            query: A short description of what you are about to build.
                   Example: "Flask REST API with SQLite and user authentication"
        """
        try:
            # Embed the query
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query[:2000],
            )
            query_vector = response.data[0].embedding

            # Search episodic memory
            rows = database_service.search_episodic_memory(
                user_id=user_id,
                query_vector=query_vector,
                limit=4,
            )

            if not rows:
                return (
                    "No past projects found for this user. "
                    "This appears to be their first generation."
                )

            lines = [
                f"Found {len(rows)} relevant past project(s) for this user:\n"
            ]

            for row in rows:
                score_str = f"{row.quality_score}/10" if row.quality_score else "no score"
                passed_str = "passed review" if row.passed else "failed review"
                
                # created_at formatting
                date_str = row.created_at.strftime('%d %b %Y') if row.created_at else "Unknown Date"

                lines.append(
                    f"Project: {row.app_name}\n"
                    f"  Stack: {row.techstack}\n"
                    f"  Result: {score_str} ({passed_str})\n"
                    f"  Files: {row.file_count} | Retries: {row.retry_count}\n"
                    f"  Built: {date_str}"
                )

                if row.summary:
                    lines.append(f"  Summary: {row.summary}")

                # Show issues grouped by severity
                if row.issues_json and row.issues_json != "[]":
                    issues = json.loads(row.issues_json)
                    high = [i for i in issues if i.get("severity") == "high"]
                    med  = [i for i in issues if i.get("severity") == "medium"]

                    if high:
                        lines.append("  High-severity issues (must avoid):")
                        for i in high[:3]:
                            lines.append(
                                f"    - {i.get('filepath')}: {i.get('description')} "
                                f"→ Fix: {i.get('suggested_fix')}"
                            )
                    if med:
                        lines.append("  Medium-severity issues:")
                        for i in med[:2]:
                            lines.append(
                                f"    - {i.get('filepath')}: {i.get('description')}"
                            )

                lines.append("")  # blank line between entries

            return "\n".join(lines)

        except Exception as e:
            return f"Could not retrieve past projects — proceeding without history context. Error: {e}"

    return recall_past_projects
