
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient

# Find the main project folder.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY was not found. Check your project's .env file."
    )

client = TavilyClient(api_key=api_key)

print("Searching for recent Bitcoin news...")

response = client.search(
    query="Bitcoin BTC latest news market price regulation adoption",
    topic="news",
    time_range="week",
    max_results=10,
    include_answer=False,
)

results = response.get("results", [])

if not results:
    print("No news results returned. Try again later.")
else:
    rows = []

    for item in results:
        rows.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score", 0),
            "published_date": item.get("published_date", ""),
        })

    df = pd.DataFrame(rows)

    output_path = BASE_DIR / "data" / "btc_news.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nCollected {len(df)} news results.")
    print(f"Saved to: {output_path}\n")

    for index, item in enumerate(rows, start=1):
        print(f"{index}. {item['title']}")
        print(f"   URL: {item['url']}")
        print()
