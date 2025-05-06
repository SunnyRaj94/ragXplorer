# rag_utils/indexing.py

import faiss
import numpy as np
import chromadb

# import pickle
from typing import List, Dict, Tuple, Optional


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
