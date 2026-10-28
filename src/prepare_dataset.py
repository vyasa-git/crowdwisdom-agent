
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

sentiment_path = DATA_DIR / "btc_sentiment.csv"
prices_path = DATA_DIR / "prices_2023.csv"
output_path = DATA_DIR / "merged_btc_dataset.csv"

print("Loading datasets...")

# Load the files
sentiment = pd.read_csv(sentiment_path)
prices = pd.read_csv(prices_path)

# Standardize dates
sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")

# Keep only sentiment-related columns
sentiment = sentiment[["date", "compound", "sentiment_type"]].copy()

# Rename price columns
prices = prices.rename(columns={
    "Date": "date",
    "Open": "price_open",
    "High": "price_high",
    "Low": "price_low",
    "Close": "price_close",
    "Volume": "price_volume"
})

# Keep only the required price columns
prices = prices[
    ["date", "price_open", "price_high", "price_low",
     "price_close", "price_volume"]
].copy()

# Convert numeric columns
for column in ["compound"]:
    sentiment[column] = pd.to_numeric(
        sentiment[column], errors="coerce"
    )

for column in [
    "price_open", "price_high", "price_low",
    "price_close", "price_volume"
]:
    prices[column] = pd.to_numeric(
        prices[column], errors="coerce"
    )

# Merge by matching dates
merged = pd.merge(
    sentiment,
    prices,
    on="date",
    how="inner",
    validate="one_to_one"
)

merged = merged.sort_values("date").reset_index(drop=True)

# Remove rows missing essential data
merged = merged.dropna(
    subset=["date", "compound", "price_close"]
).copy()

# Create next-day prediction targets
merged["next_day_close"] = merged["price_close"].shift(-1)

merged["next_day_return"] = (
    merged["next_day_close"] / merged["price_close"]
) - 1

# Save the merged dataset
merged.to_csv(output_path, index=False)

print("\nMerge successful!")
print("Sentiment rows:", len(sentiment))
print("Price rows:", len(prices))
print("Matched rows:", len(merged))
print(
    "Date range:",
    merged["date"].min(),
    "to",
    merged["date"].max()
)
print("Saved to:", output_path)

print("\nFirst 5 rows:")
print(merged.head().to_string(index=False))

print("\nLast 5 rows:")
print(merged.tail().to_string(index=False))
