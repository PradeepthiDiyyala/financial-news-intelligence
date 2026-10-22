from __future__ import annotations

from datasets import load_dataset
import pandas as pd

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

def load_financial_phrasebank() -> pd.DataFrame:
    """Load the all-agree Financial PhraseBank configuration."""
    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=False,
    )
    df = dataset["train"].to_pandas()
    df = df.rename(columns={"sentence": "text"})
    df["sentiment"] = df["label"].map(LABEL_MAP)
    return df[["text", "label", "sentiment"]].dropna().reset_index(drop=True)

if __name__ == "__main__":
    data = load_financial_phrasebank()
    print(f"Rows: {len(data):,}")
    print(data["sentiment"].value_counts())
