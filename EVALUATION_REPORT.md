# Evaluation Report — X Sentiment Price Prediction Agent

## 1. Project Overview

This project explores whether Bitcoin market sentiment, combined with historical price and trading-volume features, can help predict future returns. It includes sentiment analysis, news analysis, dataset preparation, multi-horizon machine learning, visualizations, and a historical walk-forward backtest.

## 2. Data Sources

- **Bitcoin prices:** Yahoo Finance data covering approximately April–October 2026, plus historical prices from January–June 2023.
- **Historical sentiment:** A public dataset containing daily Bitcoin sentiment observations from January–June 2023.
- **X posts:** Sampled Bitcoin-related posts collected for May 1–7 and October 28, 2026.
- **News:** Articles retrieved using Tavily and analyzed with VADER sentiment scoring.

### Data Limitations

The historical sentiment dataset and current market data cover different periods. The collected X posts provide sparse daily coverage rather than continuous observations across the full 180-day period. Some retrieved news articles may be irrelevant or incorrectly classified by automated sentiment analysis.

Consequently, the available data does not establish a fully integrated, continuously updated prediction system.

## 3. Model and Validation

A Ridge regression model with feature standardization was evaluated against a DummyRegressor baseline. Features included sentiment, daily returns, daily price range, and volume changes.

For multi-horizon prediction, TimeSeriesSplit walk-forward validation was used with five splits and a gap corresponding to the forecast horizon.

### Multi-Horizon Results

| Horizon | Ridge MAE | Baseline MAE | Ridge Direction Accuracy | Baseline Direction Accuracy |
|---|---:|---:|---:|---:|
| 1 day | 0.02130 | 0.02008 | 43.8% | 44.6% |
| 2 days | 0.03100 | 0.02871 | 51.5% | 52.3% |
| 3 days | 0.04050 | 0.04106 | 41.5% | 43.8% |
| 4 days | 0.05339 | 0.05218 | 42.3% | 42.3% |
| 5 days | 0.06456 | 0.06183 | 42.3% | 42.3% |

MAE measures the mean absolute error in predicted returns. Lower values are better. Direction accuracy measures how often the predicted direction matched the actual direction.

The Ridge model achieved lower MAE than the baseline for only one of the five forecast horizons. These results do not demonstrate consistent predictive improvement.

## 4. Historical Walk-Forward Backtest

A one-day prediction strategy was evaluated on 110 out-of-sample test days. The strategy held Bitcoin when the predicted return was positive and otherwise held cash.

| Strategy | Total Return | Sharpe Ratio | Sortino Ratio | Maximum Drawdown |
|---|---:|---:|---:|---:|
| Ridge long/cash | -3.03% | 0.019 | 0.030 | -22.50% |
| Baseline long/cash | 7.08% | 0.702 | 1.145 | -16.54% |
| Buy and hold | 7.08% | 0.702 | 1.145 | -16.54% |

The backtest used an expanding training window and 365-day annualization. Transaction costs and slippage were not included.

The Ridge strategy underperformed both the baseline and buy-and-hold strategies over this test period. The baseline and buy-and-hold results were identical because the baseline remained invested throughout the test period.

These are historical simulation results, not evidence of future profitability.

## 5. News and Sentiment Analysis

The collected news sample contained eight articles. Automated scoring classified five as positive and three as negative, with an average compound sentiment score of approximately 0.217.

The sample included potentially irrelevant articles, demonstrating the need for stronger topic filtering and manual relevance checks before news sentiment is used in a trading model.

The sampled X posts were scored and aggregated by date. However, sentiment coverage was too sparse to represent daily sentiment reliably across the entire price dataset.

## 6. Visualizations and Outputs

The project generates:

- `data/btc_sentiment_vs_price.png` — historical sentiment and price visualization.
- `data/sentiment_distribution.png` — historical sentiment distribution.
- `data/sorsa_sentiment_vs_price.png` — sampled X sentiment and Bitcoin price visualization.
- `data/bitcoin_sentiment_outlook.md` — a preliminary sentiment outlook.
- `data/forecast_evaluation.csv` — multi-horizon model evaluation.
- `data/backtest_metrics.csv` — historical strategy performance metrics.
- `data/backtest_predictions.csv` — out-of-sample predictions.

## 7. Conclusion

This project demonstrates a reproducible prototype for sentiment processing, feature engineering, time-series model evaluation, visualization, and historical strategy testing.

The current results do not show that the Ridge model consistently outperforms a simple baseline. Sparse X data, imperfect news relevance, differences between dataset periods, and omitted transaction costs limit the conclusions that can be drawn.

### Future Improvements

1. Collect a consistent, sufficiently large dataset of timestamped X posts and market prices.
2. Improve news relevance filtering and sentiment classification.
3. Evaluate stronger models and compare them against simple baselines.
4. Produce explicit forecasts for price horizons n+0 through n+5.
5. Test alternative assets and forecast periods.
6. Include transaction costs, slippage, and realistic execution assumptions.
7. Repeat evaluation over multiple out-of-sample periods.

## 8. Reproducibility

Python scripts are provided in `src/`, with package dependencies listed in `requirements.txt`. Data and generated outputs are stored in `data/`.

API credentials and other secrets must be supplied through local environment variables and must not be committed to the repository.
