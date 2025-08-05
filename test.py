from enum import Enum
from repositories.concrete.postgres_repo import PostgresREPO
from models.movie_model import Movie

# thay vì tạo class mới có thể add vào class cũ đã có search service, nhớ update abstractmethod
# Movie model : id, title, genre.

class Filter: 
    def __init__(self):
        self.postgres = PostgresREPO()
        self._genre = None

    def filter_by_genre(self, _genre = Query("action", all_genre_enum), top_k) -> list[Movie]: # list movie hay list dict
        conn = self.postgres_repo.connect_to_postgres()
        cursor = conn.cursor()
        cursor.execute( "SELECT movieId, title, genres FROM movies WHERE genres LIKE %s ORDER BY movieId LIMIT %s",(f'%{_genre}%',top_k))
        movies = cursor.fetchall()
        cursor.close()
        conn.close()

        return [{"id": row['movieid'], "title": row['title'], "genres": row['genres'] or ''} for row in movies]

    def get_all_genre(self) -> list[str]:
        conn = self.postgres_repo.connect_to_postgres()
        cursor.conn.cursor()
        cursor.execute("SELECT DISTINCT genre FROM movies ORDER BY genre")
        all_genre = cursor.fetchall()
        #self._genre = [row[0] for row in all_genre] if all_genre else []
        genre_set = set()
        for row in all_genres:
            if row[0]:
              genres = row[0].split('|') if '|' in row[0]
              for genre in genres:
               genre_set.add(genres)
        cursor.close()
        conn.close()
        self._genre = sort(list(genre_set))
        return self._genre

    def convert_list_to_enum(self, list: list[str]) -> Enum:
        return Enum('genre_enum', list)
    
    def update_genre_list(self, genre: ) -> Enum:
        all_genre = self.get_all_genre()
        genre_dict = {all_genre.replace('','_').replace('-',''): genre for genre in all_genre}
        
        return Enum(genre_dict)

    

# ko enum được, genre có thể phải được thêm sửa xoá nếu cần, dù hiếm khi
# có thể chuyển type từ str sang enum? xong khi cần thì clear và update?



@app.get("/filter_by_genre")
def filter_by_genre(self, genre = Query("action", all_genre_enum), top_k) -> list[Movie]:
     return [insert.path].filter_by_genre(genre, top_k)






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