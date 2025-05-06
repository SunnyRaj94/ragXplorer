# rag_utils/retrieval.py

from typing import List, Dict, Tuple, Literal, Union

# import numpy as np
from .indexing import search_index, query_chroma


def retrieve_from_faiss(
    index, query_embedding: List[float], documents: List[Dict], k: int = 5
) -> List[Dict]:
    ids, scores = search_index(index, query_embedding, k)
    return [
        {
            "text": documents[i]["text"],
            "metadata": documents[i]["metadata"],
            "score": scores[idx],
        }
        for idx, i in enumerate(ids)
    ]


def retrieve_from_chroma(collection, query_text: str, k: int = 5) -> List[Dict]:
    result = query_chroma(collection, query_text, k)
    return [
        {"text": doc, "metadata": meta, "score": dist}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def retrieve_documents(
    query: Union[str, List[float]],
    source: Union[object, Tuple[object, List[Dict]]],
    method: Literal["faiss", "chroma"] = "faiss",
    k: int = 5,
) -> List[Dict]:
    """
    Unified document retrieval interface.

    Parameters:
        query: Either text (for Chroma) or embedding vector (for FAISS)
        source:
            - For FAISS: tuple (faiss_index, documents)
            - For Chroma: chroma_collection
        method: "faiss" or "chroma"
        k: Number of results

    Returns:
        List of dicts with text, metadata, and similarity score
    """
    if method == "faiss":
        index, docs = source
        return retrieve_from_faiss(index, query, docs, k)
    elif method == "chroma":
        return retrieve_from_chroma(source, query, k)
    else:
        raise ValueError(f"Unknown method: {method}")
