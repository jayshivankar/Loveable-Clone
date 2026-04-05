from sqlalchemy.orm import Session
from src.models import engine
from sqlalchemy import create_engine, exc


def get_db() -> Session:
    try:
        # Create a new session
        session = Session(bind=engine)
        return session
    except exc.SQLAlchemyError as e:
        raise Exception(f"Error occurred while connecting to the database: {e}")
