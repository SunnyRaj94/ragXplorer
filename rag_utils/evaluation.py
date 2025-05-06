from typing import List, Optional  # , Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# import numpy as np
import re


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and normalize whitespace."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def f1_score(pred: str, target: str) -> float:
    """Compute simple token-overlap F1 between predicted and reference answer."""
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()
    common = set(pred_tokens) & set(target_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def cosine_sim(text1: str, text2: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    vec = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]


def evaluate_generation(
    prediction: str, reference: str, retrieved_docs: Optional[List[str]] = None
) -> dict:
    """
    Evaluate a generated answer with basic metrics.

    Parameters:
        prediction (str): LLM output
        reference (str): Ground truth answer
        retrieved_docs (List[str], optional): Context used

    Returns:
        dict with f1, similarity, context_recall (if docs provided)
    """
    scores = {
        "f1": f1_score(prediction, reference),
        "similarity": cosine_sim(prediction, reference),
    }

    if retrieved_docs:
        context_combined = " ".join(retrieved_docs)
        scores["context_recall"] = cosine_sim(reference, context_combined)

    return scores
