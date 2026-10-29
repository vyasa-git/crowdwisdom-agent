
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

input_file = "data/apify_tweets.csv"
output_file = "data/apify_tweets_scored.csv"
daily_file = "data/apify_daily_sentiment.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"Missing input file: {input_file}")

df = pd.read_csv(input_file)

if "text" not in df.columns:
    raise ValueError("The CSV must contain a 'text' column.")

df["text"] = df["text"].fillna("").astype(str)

analyzer = SentimentIntensityAnalyzer()

df["compound"] = df["text"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

df["sentiment"] = df["compound"].apply(
    lambda score: (
        "positive" if score >= 0.05
        else "negative" if score <= -0.05
        else "neutral"
    )
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    format="%a %b %d %H:%M:%S %z %Y",
    errors="coerce",
    utc=True
)

os.makedirs("data", exist_ok=True)
df.to_csv(output_file, index=False)

valid = df.dropna(subset=["created_at"]).copy()

if not valid.empty:
    valid["date"] = valid["created_at"].dt.strftime("%Y-%m-%d")

    daily = valid.groupby("date").agg(
        tweet_count=("text", "count"),
        compound=("compound", "mean"),
        positive_count=("sentiment", lambda s: (s == "positive").sum()),
        neutral_count=("sentiment", lambda s: (s == "neutral").sum()),
        negative_count=("sentiment", lambda s: (s == "negative").sum()),
        likes=("likes", "sum"),
        retweets=("retweets", "sum"),
    ).reset_index()

    daily.to_csv(daily_file, index=False)
    print(f"Daily sentiment saved to {daily_file}")
else:
    print("No valid timestamps; daily aggregation was skipped.")

print(f"Scored {len(df)} posts.")
print(f"Scored posts saved to {output_file}")
print(df[["text", "compound", "sentiment"]].head(5).to_string(index=False))
