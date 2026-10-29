# X Sentiment Price Prediction Agent

## Model Evaluation Report

### 1. Project Overview

This project investigates whether cryptocurrency sentiment and historical market information can help predict Bitcoin returns over multiple forecast horizons.

The pipeline includes:
- Historical Bitcoin price collection using yfinance.
- News collection using Tavily.
- X post collection using Sorsa and Apify.
- Sentiment scoring using VADER.
- Data preparation and feature engineering.
- Ridge regression model training and time-series evaluation.
- Visualization and preliminary trading-strategy backtesting.

### 2. Data Sources and Limitations

- Historical Bitcoin prices were collected using yfinance.
- Historical sentiment data was used for exploratory model training.
- Historical X data was collected using Sorsa.
- Additional X posts were collected using the Apify `apidojo/tweet-scraper` actor.
- News articles were collected using Tavily and scored for exploratory analysis.
- The separately collected Sorsa X data covered May 1–7 and October 28, 2026, with seven dates matching available Bitcoin prices.
- The Apify integration was tested on 10 Bitcoin-related X posts.

The collected X samples do not provide continuous daily sentiment coverage. The historical model evaluation and recent X-data collection are separate experiments. The reported model results should not be interpreted as validation of the recent Apify sample.

Apify's monthly run limit was reached during testing. The current Apify collector therefore reuses an existing dataset rather than starting a new actor run. Fresh collection requires available actor usage and a valid Apify API token.

### 3. Predictive Model Evaluation

A Ridge regression model with standardized features was compared against a mean-prediction baseline. Evaluation used chronological time-series splits with a gap between training and test samples.

| Forecast horizon | Ridge MAE | Baseline MAE | Ridge direction accuracy | Baseline direction accuracy |
|---|---:|---:|---:|---:|
| 1 day | 0.02130 | 0.02008 | 43.8% | 44.6% |
| 2 days | 0.03100 | 0.02871 | 51.5% | 52.3% |
| 3 days | 0.04050 | 0.04106 | 41.5% | 43.8% |
| 4 days | 0.05339 | 0.05218 | 42.3% | 42.3% |
| 5 days | 0.06456 | 0.06183 | 42.3% | 42.3% |

Lower MAE is better. The Ridge model achieved a lower MAE than the baseline at only one of the five forecast horizons. Directional accuracy also did not consistently outperform the baseline.

### 4. Preliminary Trading-Strategy Backtest

An expanding-window walk-forward simulation generated out-of-sample one-day return predictions. The strategy went long when the predicted return was positive and otherwise held cash.

| Strategy | Test days | Total return | Sharpe ratio | Sortino ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Ridge long/cash | 110 | -3.03% | 0.019 | 0.030 | -22.50% |
| Baseline long/cash | 110 | 7.08% | 0.702 | 1.145 | -16.54% |
| Buy and hold | 110 | 7.08% | 0.702 | 1.145 | -16.54% |

The baseline and buy-and-hold results were identical because the baseline's positive mean prediction resulted in continuous market exposure under the chosen trading rule.

The Ridge strategy underperformed both comparison strategies in this simulation. Transaction costs, slippage, and execution constraints were not included. These historical results do not indicate future profitability.

### 5. Sentiment Analysis and Visualizations

The project uses VADER to calculate compound sentiment scores and classify posts as positive, negative, or neutral.

The Apify test produced:
- 10 collected X posts.
- 10 posts with valid timestamps.
- 6 negative, 3 positive, and 1 neutral classification.
- Mean compound sentiment of -0.20992 for the test sample.

This is a small, single-day sample and should not be treated as representative of overall Bitcoin market sentiment.

The project also generates charts comparing Bitcoin prices with collected tweet sentiment and showing sentiment distributions. Missing sentiment dates are not treated as neutral sentiment.

### 6. Conclusion

This project demonstrates an end-to-end prototype for collecting and processing sentiment data, preparing historical features, evaluating multi-horizon forecasts, and measuring preliminary strategy performance.

The current results do not establish a consistent predictive advantage for the Ridge model. The limited and discontinuous recent X sample, small historical dataset, and absence of transaction costs in the backtest are important limitations.

The Apify integration demonstrates collection from a saved dataset, sentiment scoring, and daily aggregation. Further work is needed to establish continuous data collection and evaluate whether the new data improves forecasting performance.

### 7. Future Improvements

- Collect more relevant X posts with consistent historical coverage.
- Improve filtering for irrelevant posts and news.
- Investigate sentiment features and engagement-weighted signals.
- Compare additional models using strictly chronological validation.
- Integrate recent Apify sentiment into the modeling pipeline when sufficient overlapping data is available.
- Evaluate alternative trading rules and include transaction costs and slippage.
- Test robustness across different periods and market conditions.

### 8. Reproducibility

Important outputs include:

- `data/forecast_evaluation.csv`
- `data/backtest_predictions.csv`
- `data/backtest_metrics.csv`
- `data/merged_btc_dataset.csv`
- `data/merged_sorsa_btc_dataset.csv`
- `data/sorsa_sentiment_vs_price.png`
- `data/apify_tweets.csv`
- `data/apify_tweets_scored.csv`
- `data/apify_daily_sentiment.csv`

Relevant scripts are located in `src/`.


### 9. Setup and Commands

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file containing your own API credentials:

```env
TAVILY_API_KEY=your_tavily_api_key
APIFY_API_TOKEN=your_apify_api_token
```

Never commit `.env` or publish API tokens to GitHub.

Run the Apify sentiment pipeline:

```powershell
python src/run_apify_pipeline.py
```

The current pipeline reads the configured existing Apify dataset, saves the posts, calculates VADER sentiment, and generates the daily summary. It does not start a new Apify run.

Run the existing model evaluation and visualization scripts:

```powershell
python src/train_models.py
python src/backtest_metrics.py
python src/visualize.py
```

**Note:** Fresh Apify collection depends on the actor's usage limits and the collector's configuration. The current collector is configured to reuse a saved dataset.

