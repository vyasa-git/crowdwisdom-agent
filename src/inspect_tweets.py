
from pathlib import Path
import pandas as pd

files = [
    Path("data/sorsa_tweets_may_01_07.csv"),
    Path("data/sorsa_tweets_oct_28.csv"),
]

for file in files:
    print("\n" + "=" * 60)
    print(f"FILE: {file}")

    if not file.exists():
        print("ERROR: File not found.")
        continue

    df = pd.read_csv(file, low_memory=False)

    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "date_utc" in df.columns:
        dates = pd.to_datetime(
            df["date_utc"], errors="coerce", utc=True
        )

        print("Valid timestamps:", dates.notna().sum())
        print("Earliest timestamp:", dates.min())
        print("Latest timestamp:", dates.max())

        print("\nPosts per day:")
        print(
            dates.dt.strftime("%Y-%m-%d")
            .value_counts()
            .sort_index()
            .to_string()
        )

    if "text" in df.columns:
        print("Missing text:", df["text"].isna().sum())
        print("Empty text:", df["text"].fillna("").str.strip().eq("").sum())

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False))
