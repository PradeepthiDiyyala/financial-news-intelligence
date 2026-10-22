from __future__ import annotations

from functools import lru_cache
from transformers import pipeline

MODEL_NAME = "ProsusAI/finbert"

@lru_cache(maxsize=1)
def get_finbert():
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None,
    )

def predict_finbert(text: str) -> list[dict]:
    classifier = get_finbert()
    result = classifier(text[:4000])[0]
    return sorted(result, key=lambda x: x["score"], reverse=True)
