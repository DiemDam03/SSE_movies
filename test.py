def filter_by_genre(self, genre: [insert enum]):
    pass



def get_all_genre(self) -> enum:
    pass


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