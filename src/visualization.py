from __future__ import annotations

import matplotlib.pyplot as plt


def plot_sentiment_distribution(df):
    """Plot the distribution of sentiment labels."""
    counts = df["sentiment"].value_counts().reindex(
        ["negative", "neutral", "positive"], fill_value=0
    )
    ax = counts.plot(kind="bar", title="Financial News Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Articles")
    plt.tight_layout()
    return ax


def plot_text_length_distribution(df):
    """Plot distribution of article word counts."""
    ax = df["word_count"].plot(kind="hist", bins=25, title="News Word Count Distribution")
    ax.set_xlabel("Word Count")
    plt.tight_layout()
    return ax
