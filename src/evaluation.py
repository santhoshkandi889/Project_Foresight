"""
Project FORESIGHT – Evaluation Module
========================================
Forecasting metrics: WAPE, MAPE, MAE, RMSE, Bias, MdAPE
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from src.utils import logger


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (preferred over MAPE for zeros)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / denom)


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (excludes zeros in actuals)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual > epsilon
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Forecast Bias (positive = over-forecast, negative = under-forecast)."""
    return float(np.mean(np.asarray(predicted) - np.asarray(actual)))


def mdape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-8) -> float:
    """Median Absolute Percentage Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual > epsilon
    if mask.sum() == 0:
        return 0.0
    return float(np.median(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute full suite of evaluation metrics."""
    return {
        "WAPE":  round(wape(actual, predicted) * 100, 4),
        "MAPE":  round(mape(actual, predicted) * 100, 4),
        "MAE":   round(mae(actual, predicted), 4),
        "RMSE":  round(rmse(actual, predicted), 4),
        "Bias":  round(bias(actual, predicted), 4),
        "MdAPE": round(mdape(actual, predicted) * 100, 4),
    }


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create a comparison DataFrame sorted by WAPE."""
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name, **metrics}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def select_best_model(results: Dict[str, Any], metric: str = "WAPE") -> str:
    """Return the model name with the lowest validation metric."""
    best = min(results, key=lambda m: results[m].get(metric, float("inf")))
    logger.info(f"🏆 Best model: {best} | {metric}={results[best].get(metric):.2f}%")
    return best


if __name__ == "__main__":
    # Quick sanity check
    y_true = np.array([100, 200, 150, 300, 0])
    y_pred = np.array([90, 210, 160, 280, 5])
    metrics = compute_all_metrics(y_true, y_pred)
    print(metrics)
