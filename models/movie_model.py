import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pydantic import BaseModel

class Movie(BaseModel):
    id: int
    title: str
    genres: str