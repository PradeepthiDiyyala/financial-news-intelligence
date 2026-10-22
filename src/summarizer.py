from __future__ import annotations

from functools import lru_cache
from transformers import pipeline

MODEL_NAME = "google/flan-t5-small"

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )

def _chunks(text: str, max_chars: int = 2500) -> list[str]:
    words = text.split()
    chunks, current, size = [], [], 0

    for word in words:
        if size + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks

def summarize(text: str) -> str:
    generator = get_summarizer()
    parts = []

    for chunk in _chunks(text):
        prompt = (
            "Summarize the following financial news in 2 concise sentences. "
            "Preserve important numbers, companies, events, and financial context. "
            f"Text: {chunk}"
        )
        output = generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,
        )[0]["generated_text"]
        parts.append(output.strip())

    return " ".join(parts)
