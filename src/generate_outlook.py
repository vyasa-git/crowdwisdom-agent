
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

NEWS_PATH = DATA_DIR / "btc_news_scored.csv"
PRICES_PATH = DATA_DIR / "prices.csv"
EVAL_PATH = DATA_DIR / "forecast_evaluation.csv"
HISTORICAL_PATH = DATA_DIR / "merged_btc_dataset.csv"
OUTPUT_PATH = DATA_DIR / "bitcoin_sentiment_outlook.md"

required = [NEWS_PATH, PRICES_PATH, EVAL_PATH, HISTORICAL_PATH]
for path in required:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

news = pd.read_csv(NEWS_PATH).fillna("")
prices = pd.read_csv(PRICES_PATH)
evaluation = pd.read_csv(EVAL_PATH)
historical = pd.read_csv(HISTORICAL_PATH)

# Filter news using a simple keyword relevance check.
keywords = [
    "bitcoin", "btc", "crypto", "cryptocurrency",
    "blockchain", "digital asset", "bitcoin etf"
]

def is_relevant(row):
    text = (
        str(row.get("title", "")) + " " +
        str(row.get("content", ""))
    ).lower()
    return any(word in text for word in keywords)

relevant_news = news[news.apply(is_relevant, axis=1)].copy()

# Find the latest available market price.
date_col = "Date" if "Date" in prices.columns else "date"
close_col = "Close" if "Close" in prices.columns else "close"

prices[date_col] = pd.to_datetime(prices[date_col], errors="coerce")
prices[close_col] = pd.to_numeric(prices[close_col], errors="coerce")
prices = prices.dropna(subset=[date_col, close_col]).sort_values(date_col)

latest = prices.iloc[-1]
latest_date = latest[date_col].strftime("%Y-%m-%d")
latest_price = float(latest[close_col])

# Summarise news sentiment.
if not relevant_news.empty:
    relevant_news["sentiment_score"] = pd.to_numeric(
        relevant_news["sentiment_score"], errors="coerce"
    )
    relevant_news = relevant_news.dropna(subset=["sentiment_score"])

if not relevant_news.empty:
    average_sentiment = relevant_news["sentiment_score"].mean()
    positive = int((relevant_news["sentiment_label"] == "POSITIVE").sum())
    negative = int((relevant_news["sentiment_label"] == "NEGATIVE").sum())
    neutral = int((relevant_news["sentiment_label"] == "NEUTRAL").sum())
else:
    average_sentiment = None
    positive = negative = neutral = 0

# Determine whether the ML model beat the baseline.
evaluation["model_MAE"] = pd.to_numeric(evaluation["model_MAE"])
evaluation["baseline_MAE"] = pd.to_numeric(evaluation["baseline_MAE"])
wins = int((evaluation["model_MAE"] < evaluation["baseline_MAE"]).sum())

historical_start = pd.to_datetime(historical["date"]).min().date()
historical_end = pd.to_datetime(historical["date"]).max().date()

if average_sentiment is None:
    sentiment_summary = "No relevant news articles passed the keyword filter."
else:
    sentiment_summary = (
        f"Average VADER sentiment: {average_sentiment:.3f}. "
        f"Labels: {positive} positive, {neutral} neutral, {negative} negative "
        f"among {len(relevant_news)} keyword-matched results."
    )

if wins == len(evaluation):
    model_summary = "The ML model beat the baseline at every tested horizon."
elif wins == 0:
    model_summary = (
        f"The Ridge model beat the baseline at {wins} of "
        f"{len(evaluation)} horizons. It has not demonstrated an advantage."
    )
else:
    model_summary = (
        f"The Ridge model beat the baseline at {wins} of "
        f"{len(evaluation)} horizons."
    )

report = f"""# Bitcoin Sentiment Outlook

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## Market snapshot

- **Latest available BTC closing price:** ${latest_price:,.2f}
- **Price data date:** {latest_date}
- **News search:** Recent results returned by Tavily; this is not a comprehensive news feed.

## News sentiment

{sentiment_summary}

These scores are automated text-sentiment estimates, not probabilities of a price increase.
Keyword filtering is basic and does not guarantee that every article is relevant or accurately classified.

## Forecast model status

{model_summary}

The model was evaluated using chronological time-series splits for 1–5 day returns.
Lower mean absolute error (MAE) is better. Historical performance does not guarantee future results.

## Evidence and limitations

- Historical price/sentiment training data covers {historical_start} to {historical_end}.
- The historical sentiment dataset contains daily aggregates, not a complete collection of individual X posts.
- The initial X scraping attempt returned only 10 posts, which is insufficient for robust 180-day social-media analysis.
- News sentiment and historical model features are currently separate components; news sentiment has not yet been used to train the forecasting model.
- This prototype is experimental and is not financial advice.

## Current interpretation

Treat the news sentiment as a small, preliminary snapshot only. Do not use it alone to make trading decisions. More relevant news, representative recent social-media data, and further out-of-sample validation are needed before claiming predictive value.
"""

OUTPUT_PATH.write_text(report, encoding="utf-8")

print("Outlook generated successfully.")
print(f"Relevant news results: {len(relevant_news)}")
print(f"Latest available BTC price: ${latest_price:,.2f} ({latest_date})")
print(f"Model beat baseline at {wins}/{len(evaluation)} horizons.")
print(f"Saved to: {OUTPUT_PATH}")
