from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
import joblib
import streamlit as st

from src.api_client import get_daily_series, get_news_sentiment, get_quote
from src.finbert import predict_finbert
from src.summarizer import summarize

st.set_page_config(page_title="Financial News Intelligence", page_icon="📈", layout="wide")
st.title("📈 Financial News Intelligence")
st.caption("Live financial data + NLP sentiment + FinBERT + LLM summarization")

MODEL_PATH = Path("models/tfidf_logreg.joblib")
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

@st.cache_resource
def load_ml_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data(ttl=300)
def cached_news(tickers: str, topics: str, limit: int):
    return get_news_sentiment(tickers or None, topics or None, limit)

@st.cache_data(ttl=300)
def cached_quote(symbol: str):
    return get_quote(symbol)

@st.cache_data(ttl=900)
def cached_daily(symbol: str):
    return get_daily_series(symbol)

model = load_ml_model()
with st.sidebar:
    st.header("Market Intelligence")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    news_limit = st.slider("News articles", 5, 30, 10)
    topics = st.selectbox("News topic", ["financial_markets", "earnings", "technology", "economy_fiscal"])
    api_ready = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
    if api_ready:
        st.success("Market API configured")
    else:
        st.warning("Set ALPHAVANTAGE_API_KEY to enable live data.")

if model is None:
    st.warning("Traditional ML model is not trained yet. Run: python -m src.train_models")

tab_live, tab_manual, tab_market = st.tabs(["📰 Live News Intelligence", "🧠 Analyze Text", "📊 Market Data"])

with tab_live:
    st.subheader(f"Live financial news — {ticker}")
    if not api_ready:
        st.info("Configure ALPHAVANTAGE_API_KEY and restart Streamlit.")
    else:
        try:
            articles = cached_news(ticker, topics, news_limit)
        except Exception as exc:
            st.error(f"News API error: {exc}")
            articles = []
        if not articles:
            st.info("No live news returned for this query.")
        for i, article in enumerate(articles):
            title = article.get("title", "Untitled")
            summary_text = article.get("summary", "")
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption(f"{article.get('source', 'Unknown source')} · {article.get('time_published', '')}")
                if summary_text:
                    st.write(summary_text)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("API Sentiment", str(article.get("overall_sentiment_label", "N/A")).title())
                with c2:
                    score = article.get("overall_sentiment_score")
                    st.metric("API Score", f"{float(score):.3f}" if score is not None else "N/A")
                with c3:
                    if article.get("url"):
                        st.link_button("Open article", article["url"])
                if st.button("Run FinBERT + LLM", key=f"article_{i}"):
                    text = f"{title}. {summary_text}".strip()
                    with st.spinner("Running finance NLP models..."):
                        finbert_results = predict_finbert(text)
                        generated_summary = summarize(text)
                    top = finbert_results[0]
                    st.success(f"FinBERT: {top['label'].title()} ({top['score']:.1%} confidence)")
                    st.write("**AI Summary:**", generated_summary)

with tab_manual:
    st.subheader("Analyze financial text")
    text = st.text_area("Paste a financial news article or headline", value="The company reported stronger-than-expected quarterly revenue, while management raised its full-year guidance.", height=180)
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
                st.metric("Sentiment", LABEL_MAP[int(classes[best_idx])], f"{probabilities[best_idx] * 100:.1f}% confidence")
                for cls, prob in zip(classes, probabilities):
                    st.progress(float(prob), text=f"{LABEL_MAP[int(cls)]}: {prob:.1%}")
        with col2:
            st.subheader("FinBERT")
            with st.spinner("Running finance-specific transformer..."):
                results = predict_finbert(text)
            top = results[0]
            st.metric("FinBERT Sentiment", top["label"].title(), f"{top['score'] * 100:.1f}% confidence")
            for item in results:
                st.write(f"**{item['label'].title()}** — {item['score']:.1%}")
        st.divider()
        st.subheader("LLM Summary")
        with st.spinner("Generating summary..."):
            summary = summarize(text)
        st.info(summary)

with tab_market:
    st.subheader(f"Market data — {ticker}")
    if not api_ready:
        st.info("Configure ALPHAVANTAGE_API_KEY to retrieve market data.")
    else:
        try:
            quote = cached_quote(ticker)
            price = quote.get("05. price")
            change = quote.get("09. change")
            change_pct = quote.get("10. change percent")
            volume = quote.get("06. volume")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${float(price):,.2f}" if price else "N/A")
            c2.metric("Change", change or "N/A")
            c3.metric("Change %", change_pct or "N/A")
            c4.metric("Volume", volume or "N/A")
            daily = cached_daily(ticker)
            if daily:
                frame = pd.DataFrame.from_dict(daily, orient="index")
                frame.index = pd.to_datetime(frame.index)
                frame = frame.rename(columns={"1. open":"Open", "2. high":"High", "3. low":"Low", "4. close":"Close", "5. volume":"Volume"})
                frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
                st.line_chart(frame["Close"])
                st.dataframe(frame.tail(10), use_container_width=True)
        except Exception as exc:
            st.error(f"Market API error: {exc}")

st.divider()
st.markdown("**Pipeline:** Alpha Vantage API · TF-IDF + Logistic Regression · ProsusAI/FinBERT · Google FLAN-T5-small")
