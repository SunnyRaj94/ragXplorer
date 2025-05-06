from typing import List, Union

# import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import openai
except ImportError:
    openai = None


def get_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    api_key: Union[str, None] = None,
) -> List[List[float]]:
    """
    Generate embeddings from a list of texts using either SentenceTransformer or OpenAI.

    Parameters:
        texts (List[str]): Texts to embed.
        model_name (str): Name of the embedding model.
                          Use 'openai' for OpenAI embeddings or SBERT model name.
        api_key (str): Required if using OpenAI.

    Returns:
        List[List[float]]: Embeddings.
    """
    if model_name == "openai":
        if openai is None:
            raise ImportError("OpenAI module not installed.")
        if api_key is None:
            raise ValueError("API key required for OpenAI embeddings.")

        openai.api_key = api_key
        response = openai.Embedding.create(input=texts, model="text-embedding-3-small")
        return [item["embedding"] for item in response["data"]]

    else:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers module not installed.")

        model = SentenceTransformer(model_name)
        return model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
