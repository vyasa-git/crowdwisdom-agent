
import os
import pandas as pd
from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

token = os.getenv("APIFY_API_TOKEN")
if not token:
    raise ValueError("APIFY_API_TOKEN is missing from .env")

client = ApifyClient(token)

# Keep the limit low while using the free plan.
run_input = {
    "searchTerms": ["bitcoin lang:en"],
    "maxItems": 10,
    "sort": "Latest",
    "tweetLanguage": "en",
}

# Reuse the earlier successful dataset.
# This does not start a new Apify run.
dataset_id = "DugdrkyWebFGSiK4p"

print(f"Reading existing Apify dataset: {dataset_id}")
tweets = list(client.dataset(dataset_id).iterate_items())

# Reject empty or invalid posts before saving anything.
tweets = [
    tweet for tweet in tweets
    if isinstance(tweet, dict) and tweet.get("text")
]

if not tweets:
    raise ValueError(
        "No valid posts found. Existing output was not overwritten."
    )

rows = []

for tweet in tweets:
    author = tweet.get("author") or {}

    rows.append({
        "tweet_id": tweet.get("id"),
        "text": tweet.get("text", ""),
        "created_at": tweet.get("createdAt"),
        "username": author.get("userName"),
        "likes": tweet.get("likeCount", 0),
        "retweets": tweet.get("retweetCount", 0),
        "replies": tweet.get("replyCount", 0),
        "url": tweet.get("url") or tweet.get("twitterUrl"),
    })

os.makedirs("data", exist_ok=True)
output_file = "data/apify_tweets.csv"

df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print(f"Collected {len(df)} posts.")
print(f"Saved to {output_file}")
print(df[["text", "created_at"]].head(3).to_string(index=False))
