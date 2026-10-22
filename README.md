# Financial News Intelligence using NLP & LLMs

End-to-end financial news analysis project combining data preprocessing, feature engineering, traditional NLP, finance-specific transformer sentiment analysis, LLM summarization, evaluation, and an interactive Streamlit application.

## Dataset
- Financial PhraseBank `sentences_allagree` configuration
- 2,264 labeled financial news sentences
- Three classes: negative, neutral, positive

## Features
- Data loading and preprocessing
- Text feature engineering and exploratory analysis
- TF-IDF + Logistic Regression baseline
- FinBERT finance-specific sentiment analysis
- FLAN-T5-small local LLM summarization
- Accuracy, precision, recall, and F1 evaluation
- Visualization utilities
- Streamlit interface

## Project structure
```text
financial-news-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_comparison.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── finbert.py
│   ├── summarizer.py
│   └── visualization.py
│
├── models/
├── reports/
│   └── figures/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate
```bash
python -m src.train_models
python -m src.evaluate
```

The first run downloads the dataset and transformer model weights, so internet access is required.

## Run the dashboard
```bash
streamlit run app.py
```

Model weights and generated binaries are intentionally excluded from Git history; they are downloaded or cached when needed.

> Performance metrics should be taken from the actual evaluation output produced by this project rather than hard-coded claims.
