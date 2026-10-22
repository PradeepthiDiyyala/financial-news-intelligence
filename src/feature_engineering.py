from __future__ import annotations

from collections import Counter
import re

import pandas as pd


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """Extract frequent non-stopword tokens as simple exploratory keywords."""
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "was", "were",
        "are", "has", "have", "had", "its", "into", "their", "they", "will",
        "but", "not", "been", "also", "than", "about", "after", "before", "over",
        "under", "said", "company", "year", "quarter"
    }
    tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
    counts = Counter(t for t in tokens if t not in stopwords)
    return [word for word, _ in counts.most_common(top_n)]


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight text statistics for analysis and visualization."""
    result = df.copy()
    result["char_count"] = result["text"].astype(str).str.len()
    result["word_count"] = result["text"].astype(str).str.split().str.len()
    result["exclamation_count"] = result["text"].astype(str).str.count("!")
    result["question_count"] = result["text"].astype(str).str.count(r"\?")
    return result
