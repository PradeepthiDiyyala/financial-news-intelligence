from __future__ import annotations

import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://www.alphavantage.co/query"
MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"

# Load local .env values for development without committing secrets.
load_dotenv()


def _marketaux_session() -> requests.Session:
    """Create a session with retries for transient TLS/network failures."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "financial-news-intelligence/1.0"})
    return session


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


def get_marketaux_news(
    symbol: str | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Fetch financial news using Marketaux pagination with retry handling."""
    api_token = os.getenv("MARKETAUX_API_TOKEN")
    if not api_token:
        raise RuntimeError("MARKETAUX_API_TOKEN is not configured.")

    target = min(max(limit, 1), 30)
    articles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    page = 1
    session = _marketaux_session()

    while len(articles) < target:
        remaining = target - len(articles)
        page_limit = min(remaining, 3)
        params = {
            "api_token": api_token,
            "language": "en",
            "limit": str(page_limit),
            "page": str(page),
        }
        if symbol:
            params["symbols"] = symbol.upper()

        try:
            response = session.get(MARKETAUX_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            # If an intermittent failure happens after some pages succeeded,
            # return the articles already collected instead of discarding them.
            if articles:
                break
            raise RuntimeError(f"Marketaux request failed: {exc}") from exc

        if "error" in data:
            if articles:
                break
            raise RuntimeError(str(data["error"]))

        batch = data.get("data", [])
        if not batch:
            break

        added = 0
        for article in batch:
            article_id = str(article.get("uuid") or article.get("id") or article.get("url"))
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                articles.append(article)
                added += 1

        if added == 0:
            break

        page += 1

        # Stop when the API has no more results for this query.
        if len(batch) < page_limit:
            break

        # Small pause between paginated requests to avoid bursty traffic.
        if len(articles) < target:
            time.sleep(0.25)

    return articles[:target]


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
