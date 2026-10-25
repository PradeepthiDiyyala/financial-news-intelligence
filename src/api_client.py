from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

BASE_URL = "https://www.alphavantage.co/query"

# Load local .env values for development without committing secrets.
load_dotenv()


def _request(params: dict[str, str]) -> dict[str, Any]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not configured.")

    response = requests.get(
        BASE_URL,
        params={**params, "apikey": api_key},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    if "Error Message" in data:
        raise RuntimeError(data["Error Message"])
    if "Note" in data:
        raise RuntimeError(data["Note"])
    return data


def get_news_sentiment(
    tickers: str | None = None,
    topics: str | None = "financial_markets",
    limit: int = 20,
) -> list[dict[str, Any]]:
    params = {
        "function": "NEWS_SENTIMENT",
        "limit": str(min(max(limit, 1), 1000)),
    }
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics

    data = _request(params)
    return data.get("feed", [])


def get_quote(symbol: str) -> dict[str, Any]:
    data = _request({"function": "GLOBAL_QUOTE", "symbol": symbol.upper()})
    return data.get("Global Quote", {})


def get_daily_series(symbol: str, outputsize: str = "compact") -> dict[str, Any]:
    data = _request(
        {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol.upper(),
            "outputsize": outputsize,
        }
    )
    return data.get("Time Series (Daily)", {})
