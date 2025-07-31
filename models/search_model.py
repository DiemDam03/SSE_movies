from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class SearchResult(BaseModel):
    index: int
    score: float
    text: str