from __future__ import annotations
import re

def clean_text(text: str) -> str:
    """Lightweight text normalization."""
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()
