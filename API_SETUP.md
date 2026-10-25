# Live API setup

This project can use Alpha Vantage for live financial news and market data.

1. Create an Alpha Vantage API key from the official Alpha Vantage website.
2. Copy `.env.example` to `.env`.
3. Replace `your_api_key_here` with your key.
4. In Windows Command Prompt, you can alternatively run:

```bash
set ALPHAVANTAGE_API_KEY=your_api_key_here
```

5. Restart Streamlit:

```bash
streamlit run app.py
```

The application uses the API for news, quotes, and daily historical prices. API responses are cached in Streamlit to reduce repeated requests.

Never commit `.env` or expose your API key in GitHub.
