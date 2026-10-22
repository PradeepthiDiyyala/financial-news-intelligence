from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted characters from financial text."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9$%.,!?\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_dataframe(df):
    """Return a cleaned copy of a dataframe containing a text column."""
    result = df.copy()
    result["text_clean"] = result["text"].map(clean_text)
    result = result.drop_duplicates(subset=["text_clean"])
    result = result[result["text_clean"].str.len() > 0].reset_index(drop=True)
    return result
