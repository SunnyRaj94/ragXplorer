from typing import List, Dict, Literal, Optional
import requests
import openai
from rag_utils.indexing import query_chroma


def query_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> str:
    if api_key is None:
        raise ValueError("OpenAI API key required.")
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return ""


def query_ollama(
    messages: List[Dict[str, str]],
    model: str = "mistral",
    base_url: str = "http://localhost:11434",
) -> str:
    try:
        payload = {"model": model, "messages": messages, "stream": False}
        response = requests.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"[Ollama Error] {e}")
        return ""


def query_llm(
    query: str,
    context: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    provider: Literal["openai", "ollama"] = "ollama",
    model: str = "mistral",
    api_key: Optional[str] = None,
) -> str:
    """
    Unified LLM query function for OpenAI and Ollama.

    Parameters:
        query (str): User input.
        context (str): Optional system prompt.
        chat_history (List[Dict]): Prior chat turns.
        provider (str): "openai" or "ollama"
        model (str): Model name.
        api_key (str): Required for OpenAI.

    Returns:
        str: LLM-generated response.
    """
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    if provider == "openai":
        return query_openai(messages, model=model, api_key=api_key)
    elif provider == "ollama":
        return query_ollama(messages, model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def format_context(results: Dict) -> str:
    """Extract documents and format them into a prompt-ready context string."""
    docs = results.get("documents", [[]])[0]  # top-k documents
    return "\n\n".join(docs)


def ask_openai(
    query: str,
    retriever,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
) -> str:
    results = query_chroma(retriever, query)
    context = format_context(results)
    return query_llm(
        query, context=context, provider="openai", model=model, api_key=api_key
    )


def ask_ollama(
    query: str,
    retriever,
    model: str = "mistral",
) -> str:
    results = query_chroma(retriever, query)
    context = format_context(results)
    return query_llm(query, context=context, provider="ollama", model=model)


def ask_groq(
    query: str,
    retriever,
    model: str = "llama3-8b-8192",
    api_key: Optional[str] = None,
) -> str:
    results = query_chroma(retriever, query)
    context = format_context(results)

    # POST to Groq chat endpoint
    import requests

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": query},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Groq Error] {e}")
        return ""


def ask_together(
    query: str,
    retriever,
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key: Optional[str] = None,
) -> str:
    results = query_chroma(retriever, query)
    context = format_context(results)

    import requests

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": query},
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        res = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[TogetherAI Error] {e}")
        return ""
