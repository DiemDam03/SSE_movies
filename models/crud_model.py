from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class Movie(BaseModel):
    id: int
    title: str
    genres: str