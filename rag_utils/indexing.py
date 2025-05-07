import faiss
import numpy as np
import chromadb
from qdrant_client.models import PointStruct
import uuid

# import pickle
from typing import List, Dict, Tuple, Optional, Any


# FAISS Functions
def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def save_index(index: faiss.IndexFlatL2, path: str):
    faiss.write_index(index, f"{path}.index")


def load_index(path: str) -> faiss.IndexFlatL2:
    return faiss.read_index(f"{path}.index")


def search_index(
    index: faiss.IndexFlatL2, query_embedding: List[float], k: int = 5
) -> Tuple[List[int], List[float]]:
    query = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query, k)
    return indices[0].tolist(), distances[0].tolist()


# Chroma Functions


def get_chroma_collection(persist_dir="chroma_store", collection_name="ragxplorer"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name)


def add_to_chroma(
    collection,
    texts: List[str],
    embeddings: List[List[float]],
    metadatas: Optional[List[Dict]] = None,
):
    ids = [f"doc-{i}" for i in range(len(texts))]
    # Ensure each metadata dict is non-empty
    if metadatas is None:
        metadatas = [{"source": "generated"} for _ in texts]
    else:
        metadatas = [
            m if m and len(m) > 0 else {"source": "default"} for m in metadatas
        ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )


def query_chroma(collection, query_text: str, k: int = 5):
    results = collection.query(query_texts=[query_text], n_results=k)
    return results  # contains documents, distances, metadatas, ids


# FAISS
def get_faiss_index(index_path: str = "faiss_index") -> Any:
    from faiss import IndexFlatL2
    import os

    if os.path.exists(index_path):
        # Load saved index (mock)
        return IndexFlatL2(384)  # assuming 384-dim embeddings
    return IndexFlatL2(384)


def add_to_faiss(index, embeddings: List[List[float]]) -> None:
    import numpy as np

    index.add(np.array(embeddings).astype("float32"))


# Qdrant
def get_qdrant_collection(
    collection_name="rag_collection", host="localhost", port=6333
):
    from qdrant_client import QdrantClient

    client = QdrantClient(host=host, port=port)
    try:
        client.get_collection(collection_name)
    except Exception as e:
        print(e)
        client.recreate_collection(
            collection_name=collection_name, vector_size=384, distance="Cosine"
        )
    return client


def add_to_qdrant(
    client, collection_name, embeddings: List[List[float]], texts: List[str]
) -> None:
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
        for i in range(len(texts))
    ]
    client.upsert(collection_name=collection_name, points=points)


# Weaviate
def get_weaviate_collection(class_name="RAGDocument", url="http://localhost:8080"):
    import weaviate

    client = weaviate.Client(url=url)
    if not client.schema.contains({"class": class_name}):
        client.schema.create_class(
            {
                "class": class_name,
                "vectorizer": "none",
                "properties": [{"name": "text", "dataType": ["text"]}],
            }
        )
    return client


def add_to_weaviate(
    client, class_name: str, texts: List[str], embeddings: List[List[float]]
):
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        client.data_object.create(
            data_object={"text": text}, class_name=class_name, vector=emb
        )


def get_faiss_index_local(index_path="faiss.index"):
    import faiss
    import os

    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return faiss.IndexFlatL2(384)  # default dim


def add_to_faiss_local(index, embeddings):
    index.add(embeddings)
    return index


def get_qdrant_collection_local():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(path="./local_qdrant")
    client.recreate_collection(
        collection_name="rag_collection",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return client


def add_to_qdrant_local(client, embeddings, texts):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"text": text},
        )
        for emb, text in zip(embeddings, texts)
    ]
    client.upload_points(collection_name="rag_collection", points=points)


def get_weaviate_collection_local():
    import weaviate

    client = weaviate.Client()  # or your own instance
    return client


def add_to_weaviate_local(client, embeddings, texts):
    for emb, text in zip(embeddings, texts):
        client.data_object.create(
            data={"text": text},
            class_name="RAGText",
            vector=emb,
        )
