
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "btc_news.csv"
OUTPUT_PATH = BASE_DIR / "data" / "btc_news_scored.csv"

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "News file not found. Run src/collect_news.py first."
    )

df = pd.read_csv(INPUT_PATH).fillna("")

analyzer = SentimentIntensityAnalyzer()

# Analyse each headline together with its available description.
df["text_to_analyze"] = (
    df["title"].astype(str) + ". " + df["content"].astype(str)
)

df["sentiment_score"] = df["text_to_analyze"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

def classify_sentiment(score):
    if score >= 0.05:
        return "POSITIVE"
    elif score <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"

df["sentiment_label"] = df["sentiment_score"].apply(
    classify_sentiment
)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\nBITCOIN NEWS SENTIMENT ANALYSIS")
print("=" * 40)
print(f"Articles analysed: {len(df)}")

if len(df) > 0:
    print(f"Average sentiment: {df['sentiment_score'].mean():.3f}")
    print("\nSentiment breakdown:")
    print(df["sentiment_label"].value_counts().to_string())

    print("\nIndividual results:")
    for _, row in df.iterrows():
        print(f"\n{row['sentiment_label']} ({row['sentiment_score']:.3f})")
        print(row["title"])

print(f"\nSaved scored news to: {OUTPUT_PATH}")
print("Note: Automated sentiment labels can be wrong; review important news.")
