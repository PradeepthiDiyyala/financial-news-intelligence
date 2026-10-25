from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
DATA_DIR = Path("data/raw")
DATA_FILE = DATA_DIR / "Sentences_AllAgree.txt"
DATASET_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/"
    "598b6aad98f7c8d67be161b12a4b5f2497e07edd/data/FinancialPhraseBank-v1.0.zip"
)


def _download_dataset() -> None:
    """Download and extract the original Financial PhraseBank archive."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        target = next(
            name
            for name in archive.namelist()
            if name.endswith("/Sentences_AllAgree.txt")
        )
        DATA_FILE.write_bytes(archive.read(target))


def load_financial_phrasebank() -> pd.DataFrame:
    """Load the 100%-agreement Financial PhraseBank configuration."""
    if not DATA_FILE.exists():
        _download_dataset()

    rows = []
    with DATA_FILE.open("r", encoding="iso-8859-1") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            text, sentiment = line.rsplit("@", 1)
            rows.append(
                {
                    "text": text,
                    "sentiment": sentiment,
                    "label": LABEL_MAP[sentiment],
                }
            )

    return pd.DataFrame(rows, columns=["text", "label", "sentiment"])
