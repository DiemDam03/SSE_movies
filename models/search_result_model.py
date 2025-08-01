import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class SearchResult(BaseModel):
    index: int
    score: float
    text: str