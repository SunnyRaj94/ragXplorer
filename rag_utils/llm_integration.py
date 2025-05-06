# rag_utils/llm_integration.py

from typing import List, Dict, Literal, Optional
import requests
import openai


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
