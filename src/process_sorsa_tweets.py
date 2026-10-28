
from pathlib import Path
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Project paths
DATA_DIR = Path("data")

INPUT_FILES = [
    DATA_DIR / "sorsa_tweets_may_01_07.csv",
    DATA_DIR / "sorsa_tweets_oct_28.csv",
]

OUTPUT_TWEETS = DATA_DIR / "sorsa_tweets_scored.csv"
OUTPUT_DAILY = DATA_DIR / "sorsa_daily_sentiment.csv"

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

frames = []

# --------------------------------------------------
# 1. LOAD AND VALIDATE THE FILES
# --------------------------------------------------
for file in INPUT_FILES:
    if not file.exists():
        raise FileNotFoundError(f"Missing input file: {file}")

    df = pd.read_csv(file, low_memory=False)

    required = {"id", "date_utc", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{file.name} is missing columns: {sorted(missing)}"
        )

    # Parse Twitter-style UTC timestamps
    df["timestamp"] = pd.to_datetime(
        df["date_utc"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )

    df["text"] = df["text"].fillna("").astype(str)

    # Normalize engagement columns
    for col in ["likes", "retweets", "replies", "views"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        ).fillna(0)

    df = df.dropna(subset=["timestamp"])
    frames.append(df)

# Combine files and remove duplicate post IDs
tweets = pd.concat(frames, ignore_index=True)
tweets = tweets.drop_duplicates(subset=["id"], keep="first")

if tweets.empty:
    raise ValueError("No valid tweets remain after loading.")

# --------------------------------------------------
# 2. CALCULATE SENTIMENT
# --------------------------------------------------
scores = tweets["text"].apply(analyzer.polarity_scores)

tweets["sentiment_compound"] = scores.apply(
    lambda result: result["compound"]
)

tweets["sentiment_positive"] = scores.apply(
    lambda result: result["pos"]
)

tweets["sentiment_neutral"] = scores.apply(
    lambda result: result["neu"]
)

tweets["sentiment_negative"] = scores.apply(
    lambda result: result["neg"]
)

tweets["sentiment_label"] = tweets["sentiment_compound"].apply(
    lambda score: (
        "positive" if score >= 0.05
        else "negative" if score <= -0.05
        else "neutral"
    )
)

# Use UTC dates for alignment with UTC price data
tweets["date"] = tweets["timestamp"].dt.strftime("%Y-%m-%d")

# Save individual scored posts
tweets.to_csv(OUTPUT_TWEETS, index=False)

# --------------------------------------------------
# 3. AGGREGATE BY DAY
# --------------------------------------------------
daily = (
    tweets.groupby("date")
    .agg(
        post_count=("id", "count"),
        mean_sentiment=("sentiment_compound", "mean"),
        positive_posts=(
            "sentiment_label",
            lambda values: (values == "positive").sum(),
        ),
        neutral_posts=(
            "sentiment_label",
            lambda values: (values == "neutral").sum(),
        ),
        negative_posts=(
            "sentiment_label",
            lambda values: (values == "negative").sum(),
        ),
        total_likes=("likes", "sum"),
        total_retweets=("retweets", "sum"),
        total_replies=("replies", "sum"),
        total_views=("views", "sum"),
    )
    .reset_index()
    .sort_values("date")
)

daily.to_csv(OUTPUT_DAILY, index=False)

# --------------------------------------------------
# 4. PRINT SUMMARY
# --------------------------------------------------
print("\nSENTIMENT PROCESSING COMPLETE")
print("--------------------------------")
print("Unique tweets:", len(tweets))
print("Valid timestamps:", tweets["timestamp"].notna().sum())
print("Date range:", tweets["date"].min(), "to", tweets["date"].max())

print("\nSentiment distribution:")
print(tweets["sentiment_label"].value_counts().to_string())

print("\nDaily sentiment:")
print(daily.to_string(index=False))

print("\nSaved files:")
print(OUTPUT_TWEETS)
print(OUTPUT_DAILY)
