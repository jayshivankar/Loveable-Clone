from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from fastapi import HTTPException

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"

    @staticmethod
    def validate_username(username):
        if not username or len(username) < 3:
            raise HTTPException(status_code=400, detail='Username must be at least 3 characters long.')

    @staticmethod
    def validate_password(password):
        if not password or len(password) < 6:
            raise HTTPException(status_code=400, detail='Password must be at least 6 characters long.')
