from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Movie(BaseModel):
    id: int
    title: str
    genres: str

class MovieCreate(BaseModel):
    id: int
    title: str
    genres: str

class MovieUpdate(BaseModel):
    title: str
    genres: str

class MovieInDB(Movie):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
