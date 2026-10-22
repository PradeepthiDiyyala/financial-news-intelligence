from __future__ import annotations

from pathlib import Path
import joblib
import streamlit as st

from src.finbert import predict_finbert
from src.summarizer import summarize

st.set_page_config(
    page_title="Financial News Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Financial News Intelligence")
st.caption("NLP sentiment analysis + finance-specific transformer + LLM summarization")

MODEL_PATH = Path("models/tfidf_logreg.joblib")

@st.cache_resource
def load_ml_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_ml_model()

if model is None:
    st.warning("Traditional ML model is not trained yet.")
    st.code("python -m src.train_ml", language="bash")

sample = (
    "The company reported stronger-than-expected quarterly revenue, "
    "while management raised its full-year guidance."
)

text = st.text_area(
    "Paste a financial news article or headline",
    value=sample,
    height=180,
)

if st.button("Analyze News", type="primary"):
    if not text.strip():
        st.error("Please enter some financial news text.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Traditional ML")
        if model is None:
            st.info("Train the TF-IDF + Logistic Regression model first.")
        else:
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            best_idx = probabilities.argmax()

            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            prediction = label_map[int(classes[best_idx])]

            st.metric(
                "Sentiment",
                prediction,
                f"{probabilities[best_idx] * 100:.1f}% confidence",
            )

            st.write("Class probabilities")
            for cls, prob in zip(classes, probabilities):
                st.progress(
                    float(prob),
                    text=f"{label_map[int(cls)]}: {prob:.1%}",
                )

    with col2:
        st.subheader("FinBERT")
        with st.spinner("Running finance-specific transformer..."):
            results = predict_finbert(text)

        top = results[0]
        st.metric(
            "FinBERT Sentiment",
            top["label"].title(),
            f"{top['score'] * 100:.1f}% confidence",
        )

        for item in results:
            st.write(f"**{item['label'].title()}** — {item['score']:.1%}")

    st.divider()
    st.subheader("LLM Summary")

    with st.spinner("Generating summary..."):
        summary = summarize(text)

    st.info(summary)

st.divider()
st.markdown(
    "**Models:** TF-IDF + Logistic Regression · ProsusAI/FinBERT · "
    "Google FLAN-T5-small"
)
