from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

def connect_milvus():
    connections.connect("default", host="localhost", port="19530")

def create_collection():
    fields = [
        FieldSchema(name="textbook_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]

    schema = CollectionSchema(fields, description="RAPTOR indexed textbook content")
    collection = Collection(name="textbook_collection", schema=schema)
    return collection

def insert_data(collection, embeddings, textbook_id):
    data = [
        [textbook_id] * len(embeddings),
        list(range(len(embeddings))),
        embeddings
    ]
    collection.insert(data)

if __name__ == "__main__":
    connect_milvus()
    collection = create_collection()
    # Assuming embeddings is available
    # insert_data(collection, embeddings, 1)
