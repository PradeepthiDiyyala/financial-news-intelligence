# Financial News Intelligence using NLP & LLMs

End-to-end financial news sentiment analysis and summarization project using traditional machine learning, FinBERT, and a local LLM summarizer.

## Features
- Financial PhraseBank sentiment dataset
- TF-IDF + Logistic Regression baseline
- FinBERT finance-specific sentiment analysis
- FLAN-T5-small local text summarization
- Accuracy, precision, recall, and F1 evaluation
- Streamlit interface

## Project structure
```text
financial-news-intelligence/
├── src/
├── scripts/
├── models/
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate
```bash
python -m src.train_ml
python scripts/evaluate.py
```

## Run the dashboard
```bash
streamlit run app.py
```

Model weights and generated binaries are intentionally excluded from Git history; they are downloaded/cached when needed.

> Performance metrics should be taken from the actual evaluation output produced by this project rather than hard-coded claims.
