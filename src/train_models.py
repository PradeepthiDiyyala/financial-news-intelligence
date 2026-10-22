from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_financial_phrasebank

RANDOM_STATE = 42
MODEL_PATH = Path("models/tfidf_logreg.joblib")


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def main() -> None:
    df = load_financial_phrasebank()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20,
        stratify=df["label"], random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset size: {len(df):,}")
    print(f"Train size: {len(X_train):,}")
    print(f"Test size: {len(X_test):,}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(
        y_test, predictions,
        target_names=["negative", "neutral", "positive"],
        digits=4,
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
