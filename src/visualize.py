
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "merged_sorsa_btc_dataset.csv"
OUTPUT_FILE = DATA_DIR / "sorsa_sentiment_vs_price.png"


def main():
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(13, 6))

    # Bitcoin closing price
    ax1.plot(
        df["date"],
        df["price_close"],
        label="Bitcoin closing price (USD)",
        linewidth=1.8,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Bitcoin closing price (USD)")
    ax1.grid(True, alpha=0.25)

    # Sentiment is plotted only where actual tweet data exists
    ax2 = ax1.twinx()
    sentiment_rows = df.dropna(subset=["mean_sentiment"])

    ax2.scatter(
        sentiment_rows["date"],
        sentiment_rows["mean_sentiment"],
        label="Mean tweet sentiment",
        marker="o",
        s=65,
    )
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_ylabel("Mean sentiment (-1 to +1)")
    ax2.set_ylim(-1, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Bitcoin Price and Collected X-Tweet Sentiment")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=160)
    plt.close(fig)

    print(f"Chart saved to: {OUTPUT_FILE}")
    print(f"Price dates plotted: {df['price_close'].notna().sum()}")
    print(f"Dates with sentiment plotted: {len(sentiment_rows)}")
    print("Note: sentiment points are sparse and do not represent daily coverage.")


if __name__ == "__main__":
    main()
