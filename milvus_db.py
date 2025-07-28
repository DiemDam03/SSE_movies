from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import tfidf
import pandas as pd
import os
import json

connections.connect("default", uri="tcp://localhost:19530")

COLLECTION_NAME = "movie_vectors"


def create_collection():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=300),  # giả sử vector dài 300
    ]
    schema = CollectionSchema(fields, description="Movie vector collection")
    if COLLECTION_NAME not in list_collections():
        Collection(name=COLLECTION_NAME, schema=schema)

def list_collections():
    from pymilvus import list_collections
    return list_collections()

def insert_data(texts: list[str], vectors: list[dict]):
    from pymilvus import Collection
    import numpy as np

    coll = Collection(COLLECTION_NAME)

    float_vectors = []
    for v in vectors:
        dim = 300
        vec = [0.0] * dim
        for k, val in v.items():
            idx = hash(k) % dim
            vec[idx] += val
        float_vectors.append(vec)

    data = [list(range(len(texts))), texts, float_vectors]
    coll.insert(data)

def search(query_vec: dict, top_k: int = 5):
    from pymilvus import Collection
    import numpy as np

    dim = 300
    vec = [0.0] * dim
    for k, val in query_vec.items():
        idx = hash(k) % dim
        vec[idx] += val

    coll = Collection(COLLECTION_NAME)
    coll.load()
    results = coll.search(
        data=[vec],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    return [(hit.id, hit.distance, hit.entity.get("text")) for hit in results[0]]
