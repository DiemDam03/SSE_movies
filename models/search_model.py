from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class SearchResult(BaseModel):
    index: int
    score: float
    text: str

# đổi lại thành vector service, search chỉ là một phần của vector, còn update, add, ...