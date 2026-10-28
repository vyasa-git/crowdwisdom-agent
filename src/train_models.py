

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "merged_btc_dataset.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

# Create features using information available on each date.
df["daily_return"] = df["price_close"].pct_change()
df["daily_range"] = (
    (df["price_high"] - df["price_low"]) / df["price_open"]
)
df["volume_change"] = df["price_volume"].pct_change()

FEATURES = [
    "compound",
    "daily_return",
    "daily_range",
    "volume_change",
]

df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

# Forecast returns over the next 1 to 5 days.
results = []

print("\nBITCOIN MULTI-HORIZON FORECAST EVALUATION")
print("=" * 55)

for horizon in range(1, 6):
    target = f"return_next_{horizon}d"

    # Future return measured from today's close.
    df[target] = (
        df["price_close"].shift(-horizon) / df["price_close"] - 1
    )

    data = df[FEATURES + [target]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    X = data[FEATURES]
    y = data[target]

    model_errors = []
    baseline_errors = []
    model_directions = []
    baseline_directions = []

    # Leave a gap to reduce overlap between training and test targets.
    splitter = TimeSeriesSplit(n_splits=5, gap=horizon)

    for train_idx, test_idx in splitter.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        baseline = DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        baseline_predictions = baseline.predict(X_test)

        model_errors.append(
            mean_absolute_error(y_test, predictions)
        )
        baseline_errors.append(
            mean_absolute_error(y_test, baseline_predictions)
        )

        model_directions.append(
            np.mean(np.sign(predictions) == np.sign(y_test))
        )
        baseline_directions.append(
            np.mean(np.sign(baseline_predictions) == np.sign(y_test))
        )

    model_mae = np.mean(model_errors)
    baseline_mae = np.mean(baseline_errors)
    model_direction = np.mean(model_directions)
    baseline_direction = np.mean(baseline_directions)

    results.append({
        "horizon_days": horizon,
        "model_MAE": model_mae,
        "baseline_MAE": baseline_mae,
        "model_direction_accuracy": model_direction,
        "baseline_direction_accuracy": baseline_direction,
    })

    print(f"\nForecast horizon: {horizon} day(s)")
    print(f"Ridge MAE:           {model_mae:.5f}")
    print(f"Baseline MAE:        {baseline_mae:.5f}")
    print(f"Ridge direction acc: {model_direction:.1%}")
    print(f"Baseline direction:  {baseline_direction:.1%}")

# Save evaluation results.
results_df = pd.DataFrame(results)
output_path = BASE_DIR / "data" / "forecast_evaluation.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 55)
print(f"Evaluation saved to: {output_path}")
print("Lower MAE is better.")
print("These are historical test results, not guaranteed future performance.")
