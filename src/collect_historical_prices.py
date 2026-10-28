
import yfinance as yf
from pathlib import Path

output_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "prices_2023.csv"
)

print("Downloading Bitcoin prices for the sentiment dataset period...")

btc = yf.download(
    "BTC-USD",
    start="2023-01-01",
    end="2023-06-14",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if btc.empty:
    print("No data downloaded. Check your internet connection.")
else:
    # Handle possible multi-level column names from yfinance
    if hasattr(btc.columns, "nlevels") and btc.columns.nlevels > 1:
        btc.columns = btc.columns.get_level_values(0)

    btc.index.name = "Date"
    btc = btc.reset_index()
    btc.to_csv(output_path, index=False)

    print("Download successful!")
    print("Rows:", len(btc))
    print("Saved to:", output_path)
    print(btc.head().to_string(index=False))
