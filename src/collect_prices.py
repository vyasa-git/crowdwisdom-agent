
import yfinance as yf
from pathlib import Path

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Provisional asset: Bitcoin priced in US dollars
TICKER = "BTC-USD"

print("Downloading 180 days of historical prices...")

prices = yf.download(
    TICKER,
    period="180d",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if prices.empty:
    raise RuntimeError(
        "No price data received. Check your internet connection."
    )

# Handle the column format returned by yfinance
if hasattr(prices.columns, "nlevels") and prices.columns.nlevels > 1:
    prices.columns = prices.columns.get_level_values(0)

prices = prices.reset_index()
prices.to_csv(DATA_DIR / "prices.csv", index=False)

print("\nDownload successful!")
print("Rows:", len(prices))
print("\nLatest records:")
print(prices.tail())

print("\nSaved to:", DATA_DIR / "prices.csv")
