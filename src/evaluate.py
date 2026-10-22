from __future__ import annotations

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_loader import load_financial_phrasebank
from src.finbert import get_finbert

RANDOM_STATE = 42


def main() -> None:
    df = load_financial_phrasebank()
    _, test_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=RANDOM_STATE
    )

    model = joblib.load("models/tfidf_logreg.joblib")
    ml_predictions = model.predict(test_df["text"])

    print("\n=== TF-IDF + Logistic Regression ===")
    print(f"Accuracy: {accuracy_score(test_df['label'], ml_predictions):.4f}")
    print(classification_report(
        test_df["label"], ml_predictions,
        target_names=["negative", "neutral", "positive"], digits=4,
    ))

    print("\n=== FinBERT ===")
    classifier = get_finbert()
    true_labels = test_df["sentiment"].tolist()
    predictions = []

    for i, text in enumerate(test_df["text"].tolist(), start=1):
        result = classifier(text[:4000])[0]
        best = max(result, key=lambda x: x["score"])
        predictions.append(best["label"].lower())
        if i % 100 == 0:
            print(f"Processed {i}/{len(test_df)}")

    print(classification_report(
        true_labels, predictions,
        labels=["negative", "neutral", "positive"],
        digits=4, zero_division=0,
    ))


if __name__ == "__main__":
    main()
