from __future__ import annotations

import pandas as pd
from datasets import load_dataset

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_financial_phrasebank() -> pd.DataFrame:
    """Load the all-agree Financial PhraseBank configuration."""
    # The dataset was moved to the `takala` namespace. Using the dataset
    # repository directly avoids the legacy `financial_phrasebank.py` loading
    # script that newer versions of `datasets` no longer execute.
    dataset = load_dataset(
        "takala/financial_phrasebank",
        "sentences_allagree",
    )
    df = dataset["train"].to_pandas()
    df = df.rename(columns={"sentence": "text"})
    df["sentiment"] = df["label"].map(LABEL_MAP)
    return df[["text", "label", "sentiment"]].dropna().reset_index(drop=True)
