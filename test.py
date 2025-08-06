from repositories.concrete.milvus_repo import MilvusREPO

m = MilvusREPO()

corpus = ["batman", "manbat", "manman", "batbat"]

movie = ["bat"]

a = m.add_movie(movie, corpus)

print(a)