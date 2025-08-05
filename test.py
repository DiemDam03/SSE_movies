
def filter_by_genre(self, genre: str) -> list[Movie]: # list movie hay list dict
    pass


def get_all_genre(self) -> list[str]:
    conn = self.postgres_repo.connect_to_postgres()
    cursor.conn.cursor
    cursor.excute("SELECT genre FROM movies ORDER BY  movieId")
 all_genre = cursor.fetchall()
    cursor.close()
    conn.close()
 return all_genre 

def convert_list_to_enum(self, list: list[str]) -> Enum:
    return genre_enum = Enum('genre_enum', list)

# ko enum được, genre có thể phải được thêm sửa xoá nếu cần, dù hiếm khi
# có thể chuyển type từ str sang enum? xong khi cần thì clear và update?



# tham khảo bên dưới
import uvicorn
from fastapi import FastAPI, Query
app = FastAPI()
@app.get('/get_countries')
async def get_countries(_q: str = Query("eu", enum=["eu", "us", "cn", "ru"])):
    return {"selected": _q}
if __name__ == '__main__':
    uvicorn.run("test_swagger_dropdown:app", reload=True)


from fastapi import FastAPI
from enum import Enum
class Country(str, Enum):
    eu = "eu"
    us = "us"
    cn = "cn"
    ru = "ru"
app = FastAPI()
@app.get("/")
def get_something(country: Country = Country.eu):
    return {"country": country.value}  