from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.core.database import SessionLocal
from backend.app.models.episodic_memory import EpisodicMemory

class DatabaseService:
    def write_episodic_memory(self, **kwargs) -> None:
        """Write one episodic memory row."""
        with SessionLocal() as session:
            entry = EpisodicMemory(**kwargs)
            session.add(entry)
            session.commit()

    def search_episodic_memory(
        self,
        user_id: str,
        query_vector: list[float],
        limit: int = 4,
    ) -> list:
        """
        Find the most relevant past episodes for this user
        using cosine similarity on the embedding column.
        Only returns DONE episodes (those with a quality_score).
        """
        with SessionLocal() as session:
            result = session.execute(
                text("""
                    SELECT *,
                           1 - (embedding <=> :vec::vector) AS similarity
                    FROM episodic_memory
                    WHERE user_id = :uid
                      AND quality_score IS NOT NULL
                    ORDER BY embedding <=> :vec::vector
                    LIMIT :lim
                """),
                {
                    "uid": user_id,
                    "vec": str(query_vector),
                    "lim": limit,
                },
            )
            # return row mappings (dictionaries) so they can be accessed via dot notation or dict keys
            return result.mappings().all()

    def get_recent_episodes(
        self,
        user_id: str,
        limit: int = 5,
    ) -> list:
        """Fetch the user's most recent episodes ordered by time."""
        with SessionLocal() as session:
            result = session.execute(
                text("""
                    SELECT * FROM episodic_memory
                    WHERE user_id = :uid
                      AND quality_score IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                {"uid": user_id, "lim": limit},
            )
            return result.mappings().all()

database_service = DatabaseService()
