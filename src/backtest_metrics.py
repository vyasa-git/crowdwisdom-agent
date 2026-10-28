
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_btc_dataset.csv"

PREDICTIONS_PATH = BASE_DIR / "data" / "backtest_predictions.csv"
METRICS_PATH = BASE_DIR / "data" / "backtest_metrics.csv"

ANNUALIZATION_DAYS = 365  # Bitcoin trades every day


def calculate_metrics(returns):
    """Calculate performance and risk metrics from daily strategy returns."""
    returns = pd.Series(returns, dtype=float).dropna()

    if returns.empty:
        return {
            "total_return": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "max_drawdown": np.nan,
        }

    equity = np.concatenate(
        [[1.0], np.cumprod(1.0 + returns.to_numpy())]
    )

    total_return = equity[-1] - 1.0
    drawdowns = equity / np.maximum.accumulate(equity) - 1.0
    max_drawdown = drawdowns.min()

    daily_std = returns.std(ddof=1)
    if pd.notna(daily_std) and daily_std > 0:
        sharpe = (
            returns.mean() / daily_std
        ) * np.sqrt(ANNUALIZATION_DAYS)
    else:
        sharpe = np.nan

    # Downside deviation relative to a zero daily return.
    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))

    if downside_deviation > 0:
        sortino = (
            returns.mean() / downside_deviation
        ) * np.sqrt(ANNUALIZATION_DAYS)
    else:
        sortino = np.nan

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    features = [
        "compound",
        "daily_return",
        "daily_range",
        "volume_change",
    ]

    # Features must use information available at the prediction date.
    df["daily_return"] = df["price_close"].pct_change()
    df["daily_range"] = (
        (df["price_high"] - df["price_low"]) / df["price_open"]
    )
    df["volume_change"] = df["price_volume"].pct_change()

    # Predict the return from today's close to the next day's close.
    df["actual_next_day_return"] = (
        df["price_close"].shift(-1) / df["price_close"] - 1
    )

    df[features + ["actual_next_day_return"]] = df[
        features + ["actual_next_day_return"]
    ].replace([np.inf, -np.inf], np.nan)

    data = df.dropna(
        subset=features + ["actual_next_day_return"]
    ).reset_index(drop=True)

    # Require an initial training window before making predictions.
    minimum_training_rows = 50

    if len(data) <= minimum_training_rows:
        raise ValueError(
            f"Not enough usable rows ({len(data)}). "
            f"Need more than {minimum_training_rows}."
        )

    records = []

    # Expanding-window walk-forward predictions.
    for i in range(minimum_training_rows, len(data)):
        train = data.iloc[:i]
        test = data.iloc[[i]]

        X_train = train[features]
        y_train = train["actual_next_day_return"]
        X_test = test[features]

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        )
        model.fit(X_train, y_train)
        ridge_prediction = float(model.predict(X_test)[0])

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_prediction = float(baseline.predict(X_test)[0])

        actual_return = float(test["actual_next_day_return"].iloc[0])

        records.append({
            "date": test["date"].iloc[0],
            "actual_next_day_return": actual_return,
            "ridge_prediction": ridge_prediction,
            "baseline_prediction": baseline_prediction,
            # Long when forecast is positive; otherwise remain in cash.
            "ridge_strategy_return": (
                actual_return if ridge_prediction > 0 else 0.0
            ),
            "baseline_strategy_return": (
                actual_return if baseline_prediction > 0 else 0.0
            ),
            "buy_and_hold_return": actual_return,
        })

    predictions = pd.DataFrame(records)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    strategies = {
        "Ridge long/cash": predictions["ridge_strategy_return"],
        "Baseline long/cash": predictions["baseline_strategy_return"],
        "Buy and hold": predictions["buy_and_hold_return"],
    }

    results = []

    for name, returns in strategies.items():
        metrics = calculate_metrics(returns)
        metrics["strategy"] = name
        metrics["test_days"] = len(returns)
        results.append(metrics)

    results_df = pd.DataFrame(results)[
        [
            "strategy",
            "test_days",
            "total_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
        ]
    ]
    results_df.to_csv(METRICS_PATH, index=False)

    print("\nOUT-OF-SAMPLE BACKTEST")
    print("=" * 65)
    print(f"Historical data rows used: {len(data)}")
    print(f"Out-of-sample test days: {len(predictions)}")
    print("Trading rule: long Bitcoin if predicted return > 0; otherwise cash.")
    print("Annualization: 365 days.")
    print("Transaction costs and slippage: NOT included.")
    print("\nPerformance metrics:")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "total_return": "{:.2%}".format,
                "sharpe_ratio": "{:.3f}".format,
                "sortino_ratio": "{:.3f}".format,
                "max_drawdown": "{:.2%}".format,
            },
        )
    )

    print(f"\nSaved predictions: {PREDICTIONS_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(
        "\nWarning: Historical simulation only. Results do not guarantee "
        "future performance."
    )


if __name__ == "__main__":
    main()
