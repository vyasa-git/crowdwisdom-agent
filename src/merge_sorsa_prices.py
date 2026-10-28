
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

PRICE_FILE = DATA_DIR / "prices.csv"
SENTIMENT_FILE = DATA_DIR / "sorsa_daily_sentiment.csv"
OUTPUT_FILE = DATA_DIR / "merged_sorsa_btc_dataset.csv"


def main():
    # Load Bitcoin prices
    prices = pd.read_csv(PRICE_FILE)
    prices["date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    prices = prices.rename(
        columns={
            "Open": "price_open",
            "High": "price_high",
            "Low": "price_low",
            "Close": "price_close",
            "Volume": "price_volume",
        }
    )

    price_columns = [
        "date",
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "price_volume",
    ]
    prices = prices[price_columns].dropna(subset=["date"])
    prices = prices.drop_duplicates(subset=["date"]).sort_values("date")

    # Load daily sentiment from the collected tweets
    sentiment = pd.read_csv(SENTIMENT_FILE)
    sentiment["date"] = pd.to_datetime(
        sentiment["date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    sentiment = sentiment.dropna(subset=["date"])
    sentiment = sentiment.drop_duplicates(subset=["date"])

    # Keep every price date. Missing sentiment stays NaN; it is NOT zero.
    merged = prices.merge(sentiment, on="date", how="left")

    # Add future closing prices and returns for prediction experiments.
    # These targets are future information and must never be used as input features.
    for horizon in range(1, 6):
        future_close_col = f"close_n_plus_{horizon}"
        return_col = f"return_n_plus_{horizon}"

        merged[future_close_col] = merged["price_close"].shift(-horizon)
        merged[return_col] = (
            merged[future_close_col] / merged["price_close"]
        ) - 1

    merged.to_csv(OUTPUT_FILE, index=False)

    matched_dates = set(prices["date"]) & set(sentiment["date"])
    unmatched_sentiment_dates = sorted(
        set(sentiment["date"]) - set(prices["date"])
    )

    print("PRICE-SENTIMENT MERGE COMPLETE")
    print("-" * 40)
    print(f"Price rows: {len(prices)}")
    print(f"Daily sentiment rows: {len(sentiment)}")
    print(f"Dates matched: {len(matched_dates)}")
    print(f"Price dates without sentiment: {merged['mean_sentiment'].isna().sum()}")
    print(f"Sentiment dates without prices: {len(unmatched_sentiment_dates)}")

    if unmatched_sentiment_dates:
        print("\nSentiment dates without matching prices:")
        for date in unmatched_sentiment_dates:
            print(f"  {date}")

    print(f"\nSaved dataset: {OUTPUT_FILE}")
    print("\nSample rows:")
    print(
        merged[
            [
                "date",
                "price_close",
                "post_count",
                "mean_sentiment",
                "return_n_plus_1",
            ]
        ].tail(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
