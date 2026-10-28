
import pandas as pd
from pathlib import Path

file_path = Path(__file__).resolve().parent.parent / "data" / "btc_sentiment.csv"

df = pd.read_csv(file_path, low_memory=False)

# Remove the unnecessary index column
df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

# Convert dates and numeric columns
df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_columns = ["close", "high", "low", "open", "volume", "compound"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("===== DATA QUALITY REPORT =====")

print("\nDate range:")
print("First date:", df["date"].min())
print("Last date:", df["date"].max())

print("\nDuplicate dates:", df["date"].duplicated().sum())
print("Missing values:\n", df.isnull().sum().to_string())

print("\nSentiment label counts:")
print(df["sentiment_type"].value_counts().to_string())

print("\nSentiment score range:")
print(df["compound"].min(), "to", df["compound"].max())

# Check whether reported high/low prices are consistent
bad_high = df["high"] < df[["open", "close"]].max(axis=1)
bad_low = df["low"] > df[["open", "close"]].min(axis=1)

print("\nRows where high is below open or close:", bad_high.sum())
print("Rows where low is above open or close:", bad_low.sum())

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))
