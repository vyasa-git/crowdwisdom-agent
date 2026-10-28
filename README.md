# X Sentiment Price Prediction Agent
## Model Evaluation Report

### 1. Project overview

This project investigates whether cryptocurrency sentiment and historical market information can help predict Bitcoin returns over multiple forecast horizons.

The pipeline includes historical price collection, tweet collection, sentiment scoring, data preparation, model training, time-series evaluation, visualization, and a preliminary trading-strategy backtest.

### 2. Data sources and limitations

- Historical Bitcoin prices were collected using yfinance.
- Historical sentiment data was used for exploratory model training.
- Additional X posts were collected and scored using VADER sentiment analysis.
- News articles were collected using Tavily and scored for exploratory analysis.
- The separately collected X data covered May 1–7 and October 28, 2026, with seven dates matching available Bitcoin prices.

The collected X sample does not provide continuous daily sentiment coverage. The historical model evaluation and the recent X-data collection are separate experiments; the reported model results should not be interpreted as validation of the recent X sentiment sample.

### 3. Predictive model evaluation

A Ridge regression model with standardized features was compared against a mean-prediction baseline. Evaluation used chronological time-series splits with a gap between training and test samples.

| Forecast horizon | Ridge MAE | Baseline MAE | Ridge direction accuracy | Baseline direction accuracy |
|---|---:|---:|---:|---:|
| 1 day | 0.02130 | 0.02008 | 43.8% | 44.6% |
| 2 days | 0.03100 | 0.02871 | 51.5% | 52.3% |
| 3 days | 0.04050 | 0.04106 | 41.5% | 43.8% |
| 4 days | 0.05339 | 0.05218 | 42.3% | 42.3% |
| 5 days | 0.06456 | 0.06183 | 42.3% | 42.3% |

Lower MAE is better. The Ridge model achieved a lower MAE than the baseline at only one of the five forecast horizons. Directional accuracy also did not consistently outperform the baseline.

### 4. Preliminary trading-strategy backtest

An expanding-window walk-forward simulation generated out-of-sample one-day return predictions. The strategy went long when the predicted return was positive and otherwise held cash.

| Strategy | Test days | Total return | Sharpe ratio | Sortino ratio | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| Ridge long/cash | 110 | -3.03% | 0.019 | 0.030 | -22.50% |
| Baseline long/cash | 110 | 7.08% | 0.702 | 1.145 | -16.54% |
| Buy and hold | 110 | 7.08% | 0.702 | 1.145 | -16.54% |

The baseline and buy-and-hold results were identical because the baseline's positive mean prediction resulted in continuous market exposure under the chosen trading rule.

The Ridge strategy underperformed both comparison strategies in this simulation. Transaction costs, slippage, and execution constraints were not included. The results are historical and do not indicate future profitability.

### 5. Visualizations

The project generates charts comparing Bitcoin prices with collected tweet sentiment and showing sentiment distributions.

The recent tweet visualization contains only the dates with collected sentiment. Missing sentiment dates are not treated as neutral sentiment.

### 6. Conclusion

This experiment demonstrates an end-to-end prototype for collecting and processing sentiment data, preparing historical features, evaluating multi-horizon forecasts, and measuring preliminary strategy performance.

The current results do not establish a consistent predictive advantage for the Ridge model. The limited and discontinuous recent X sample, small historical dataset, and absence of transaction costs in the backtest are important limitations.

### 7. Future improvements

- Collect more relevant X posts with consistent historical coverage.
- Improve filtering for irrelevant posts and news.
- Investigate sentiment features and engagement-weighted signals.
- Compare additional models using strictly chronological validation.
- Evaluate alternative trading rules and include transaction costs and slippage.
- Test robustness across different periods and market conditions.

### 8. Reproducibility

Key outputs include:

- `data/forecast_evaluation.csv`
- `data/backtest_predictions.csv`
- `data/backtest_metrics.csv`
- `data/merged_btc_dataset.csv`
- `data/merged_sorsa_btc_dataset.csv`
- `data/sorsa_sentiment_vs_price.png`

Scripts for data collection, preparation, modeling, visualization, and backtesting are located in `src/`.

## 9.Commands to Run :
-py -m venv .venv
-.\.venv\Scripts\Activate.ps1
-python -m pip install -r requirements.txt
-python src/train_models.py
-python src/backtest_metrics.py
-python src/visualize.py
